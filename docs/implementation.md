# Implementation

## Files

| File | Role |
|---|---|
| `sol_attn_flex.py` | Routing, correction, and `flex_attention` execution. The whole algorithm. |
| `morton.py` | Z-order permutations for video token grids. |
| `morton_h3.py` | H3 layout hooks: publishes the video span, optionally reorders it. |
| `patch.py` | ComfyUI hook and node definitions. |
| `__init__.py` | Registration and compile warmup. |

## How it works

The routing algorithm is [Sol-Attn](https://arxiv.org/abs/2607.24027) by NVlabs
([NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine), `sol-engine` branch). What follows
describes this implementation, not their kernel.

**Routing.** Mean-pool K and Q into 128-token blocks (`[B, H, N, D]`, computed in fp32 — the
variance term is numerically poor in bf16 and the tensor is small). Score each query block against
each KV block with `q̄ @ k̄ᵀ`, and keep blocks whose score exceeds `mean + tau·std` of the
distribution. A `|i-j| <= 1` diagonal window is always kept so local attention is exact.

The threshold is derived in closed form from the moments of the block centroids under a
diagonal-independence assumption, so no score histogram is materialized. It is computed *unscaled*:
`mean`, `std` and the pilot score are all linear in the softmax scale, so the comparison is
scale-invariant and applying the scale would be wasted work.

**Packing.** `flex_attention`'s `BlockMask` wants, per query block, the selected KV block ids packed
to the front plus a count. An int32 prefix sum assigns each selected block its output position and
`scatter_add_` writes the ids in ascending order. This is linear in the number of blocks per row;
the old implementation sorted every row.

**Execution.** `BlockMask.from_kv_blocks(...)`, then `torch.compile(flex_attention)`. The real
sequence length is passed directly, so FlexAttention handles the final partial block without
padding Q/K/V. Q-side mask metadata is disabled because it is only needed for backward.

**Approximate correction.** Routing drops every block below the threshold. Individually those are
negligible; summed over the ~80% of blocks that get dropped they are not, and discarding them biases
the softmax denominator low and pulls the output toward the kept blocks. Sol-Attn recovers the tail
from the proxy scores routing already computed: approximate every key in a dropped block by the
block centroid, so with `s_ib = scale · qᵢ·k̄_b`,

```
Σ_{j∈b} exp(s_ij)·v_j  ≈  exp(s_ib) · (Σ_{j∈b} v_j)
Σ_{j∈b} exp(s_ij)      ≈  exp(s_ib) · len(b)
```

Both are one matmul against precomputed block value *sums* (`_pool_blocks(v, reduce="sum")`, with
the block length appended as an extra column so numerator and denominator come out of the same
matmul). Merging that with the exact result is the standard flash-attention rescale, so
`flex_attention`'s logsumexp is the only extra state needed.

Two details matter. The pilot score here is per *row* (`q @ k̄ᵀ`, `[B,H,S,N]`), not per query block
as in routing — that is a much larger intermediate, so it is chunked over heads on a tighter budget
(`_MAX_CORRECT_ELEMS`). And torch normalizes `flex_attention`'s base-2 statistics to natural log on
the way out; a silent base mismatch would corrupt every corrected output by a factor of `ln 2` in
the exponent, so `_probe_lse` verifies the convention against a reference `logsumexp` once at warmup
and disables the correction rather than trusting it.

**Cornish–Fisher thresholds** (optional). Pilot scores are a weighted sum over D dimensions and are
measurably skewed, so the Gaussian quantile `tau` systematically misplaces the threshold. The
expansion uses the third and fourth cumulants of the same centroid distribution to correct the
quantile, clamped to within one sigma of the Gaussian answer so a badly conditioned head cannot
swing it arbitrarily. Cost is two extra moment reductions over the (tiny) centroid tensor. It
changes *which* blocks are kept at a given `tau`, so `tau` needs re-tuning after enabling it.

## Morton reordering

Video latents are packed frame-major then row-major, so a contiguous 128-token block is a thin
horizontal strip of a single frame — at a 60×104 latent grid that is about 1.2 rows. Attention in a
video DiT is locally 3D, so that strip borders many blocks that each carry little mass. Z-ordering
makes each block a compact neighbourhood instead, which concentrates the same mass into fewer blocks
and lets a given `tau` hold quality.

Measured mean distance from block centroid, in latent cells:

| grid (t, h, w) | linear | `2d_frame` | `3d` |
|---|---|---|---|
| 7×32×64 | 16.02 | 4.73 | 2.83 |
| 7×30×52 | 13.23 | 7.05 | 3.05 |
| 5×60×104 | 25.67 | 7.02 | 2.86 |

`3d` is tighter spatially but spans ~5 latent frames per block. H3's `FRAME_PER_TOKEN` is
`(1, 4, 4, 4, 4)`, so index-adjacent frames are 1 or 4 real frames apart and those 5 index-frames can
be ~17 real frames — which is why `2d_frame` is the default.

This is exactly a no-op for dense attention. Attention is permutation-equivariant and rope is applied
per row from a table, so permuting tokens and their rope rows together permutes the output;
everything else in a block is row-wise; the adaln modulation is applied per layout segment and the
video segment is a single uniform entry, so reordering *within* it changes nothing. The permutation
is undone after the last block, before `final_layer` slices the video rows out.

Two things are easy to get wrong here:

- **The permute-or-not decision must be made in one place.** `morton_h3.py` makes it in the block-0
  pre-hook, the only point that sees the hidden states and the rope table together. Deciding
  separately for the two lets the guards disagree when a reference image or extra context changes the
  token count, which permutes positions without permuting tokens — that corrupts output outright
  rather than degrading it.
- **The span must be realigned to the block grid.** Blocks are cut from absolute position 0, so a
  video span starting at a non-multiple of 128 would split every Z-order cell across two blocks,
  joining opposite ends of the volume. `aligned_morton_perm` rotates the permutation by the
  misalignment; the ragged group that displaces lands in the block shared with the conditioning
  prefix, which prefix protection already keeps exact.

The layout hooks also publish the exact `[start, stop)` of the video segment, which is a better
source for prefix protection than deriving it from the latent shape. They are installed whenever
`protect_prefix` or `morton` is on, and are inert without the gate flag.

## How the hook attaches

ComfyUI invokes an override as `override(func, *args, **kwargs)`
(`comfy/ldm/modules/attention.py:159`), where `func` is the attention implementation that would
otherwise run — so the fallback path lands on SageAttention when ComfyUI was started with
`--use-sage-attention`.

MiniMax H3's self-attention (`comfy/ldm/minimax/model.py:178-182`) passes `q/k/v` as
`[1, heads, S, 128]` with `skip_reshape=True`, and expects `[1, S, heads*128]` back.

The node clones the ModelPatcher and installs the override into `transformer_options`. A lightweight
diffusion wrapper publishes the sampling sigma and the target-video token count; the layout hooks
publish the exact video segment boundary when available, which the override prefers. No weights are
touched.

**Block index.** The dense-block guards need to know which transformer block is running. Counting
attention calls and taking the result modulo a hardcoded 50 guesses wrong the moment anything else
issues an attention call or the model has a different depth, so a forward pre-hook on each block
publishes the true index into `transformer_options`. The counter remains as a fallback for models
without a `.blocks` list. This is what makes `dense_blocks` able to name the *last* block — as
approximation-sensitive as the first, since its error reaches the output with no later block to
absorb it.

## Design notes

**BHSD throughout.** H3 hands the hook `[B, H, S, D]` and `flex_attention` wants the same layout, so
nothing is permuted or made contiguous. PyTorch FlexAttention accepts these last-dimension-contiguous
strided views. Avoiding Q/K/V padding and repacking removes three full-sequence copies per call.

**Head-chunked routing.** The routing intermediates are `[B, H, N, N]`; at 84k tokens N=656 and a
full-width workspace runs into the hundreds of MB. Heads are processed in chunks to bound it.

**Index packing is sort-free.** `BlockMask` needs selected block ids packed to the front of each row.
An int32 cumulative count gives every selected id a unique destination. Unselected entries scatter
zero, so collisions are harmless, including when block 0 is selected.

**Density is lazy.** The density stays as a CUDA scalar until a periodic log line or the stats node
actually asks for it. The old `.item()` in every layer synchronized the GPU with Python even when
logging was disabled.

**Reference H3 quality guards.** The first portion of sampling and a named set of blocks per step can
stay dense, and sampling can return to dense near the end (`end_percent`). The complete
`[text | conditioning/reference | audio]` prefix is forced exact as KV — read from the layout hooks
when they are installed, otherwise derived from the target video length. Prefix query rows can
optionally be dense as well.

**fp16 accepted** alongside bf16. H3 is bf16-only in ComfyUI (`supported_models.py:973` lists
`[bfloat16, float32]`), but the check costs nothing and `flex_attention` handles fp16 fine.

**`min_seq_len` gate.** Below the threshold the call falls through to the normal attention path.
Sol-Attn is only ~1.5x at 4k tokens and routing overhead is real, so short sequences are better
served by whatever ComfyUI is already using.

**No `thresh_type` control.** The routing here always uses the diagonal threshold. An exact
full-covariance variant would add cost for negligible routing benefit at these block sizes, so
rather than expose a parameter that does nothing, there isn't one.

## Interaction with quantized transformers

GGUF and INT8 checkpoints are **weight-only** quantized — the loader dequantizes weights inside
`cast_bias_weight` and runs an ordinary `F.linear`. Activations are never quantized, so q/k/v reach
the hook in the model's compute dtype and nothing here needs to change.

`torch.compile(flex_attention)` is a leaf compile over plain tensors *after* the linear layers; it
never traces the quantized ops, so there are no graph breaks or recompile storms. This would not
hold under a whole-model `TorchCompileModel` node — avoid stacking that on top.

## Scope of interception

Only H3 self-attention: `skip_reshape=True`, 4-D input, `head_dim == 128`, `mask is None`, bf16/fp16,
and `S >= min_seq_len`. Everything else returns `func(...)` unchanged. Kernel exceptions are caught
and fall back per-call, so a failure degrades performance rather than aborting a generation. The
`STATS` counters in `patch.py` distinguish all of these cases.
