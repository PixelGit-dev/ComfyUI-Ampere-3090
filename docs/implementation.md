# Implementation

## Files

| File | Role |
|---|---|
| `sol_attn_flex.py` | Routing + `flex_attention` execution. The whole algorithm. |
| `patch.py` | ComfyUI hook and node definitions. |
| `__init__.py` | Registration and compile warmup. |

## How it works

The routing algorithm is [Sol-Attn](https://arxiv.org/abs/2607.24027) by NVlabs
([NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine), `sol-engine` branch). What follows
describes this implementation, not their kernel.

One deliberate gap: Sol-Attn also specifies an **approximate-correction** step — reusing
below-threshold proxy scores to recover the long-tail contribution of skipped blocks — which is not
implemented here. Blocks below threshold are dropped outright. That is a quality cost not present in
the published method, and the most substantive thing missing.

**Routing.** Mean-pool K and Q into 128-token blocks (`[B, H, N, D]`, computed in fp32 — the
variance term is numerically poor in bf16 and the tensor is small). Score each query block against
each KV block with `q̄ @ k̄ᵀ`, and keep blocks whose score exceeds `mean + tau·std` of the
distribution. A `|i-j| <= 1` diagonal window is always kept so local attention is exact.

**Packing.** `flex_attention`'s `BlockMask` wants, per query block, the selected KV block ids packed
to the front plus a count. We rank selected blocks as `idx + 1` and unselected as `0`, sort
descending, and subtract 1 — the trailing garbage sits past `kv_num_blocks` and is never read.

**Execution.** `BlockMask.from_kv_blocks(...)` with a boundary `mask_mod`, then
`torch.compile(flex_attention)`. 128 is the kernel's minimum block size.

## How the hook attaches

ComfyUI invokes an override as `override(func, *args, **kwargs)`
(`comfy/ldm/modules/attention.py:159`), where `func` is the attention implementation that would
otherwise run — so the fallback path lands on SageAttention when ComfyUI was started with
`--use-sage-attention`.

MiniMax H3's self-attention (`comfy/ldm/minimax/model.py:178-182`) passes `q/k/v` as
`[1, heads, S, 128]` with `skip_reshape=True`, and expects `[1, S, heads*128]` back.

The node clones the ModelPatcher and installs the override into `transformer_options`. No weights
are touched.

## Design notes

**BHSD throughout.** H3 hands the hook `[B, H, S, D]` and `flex_attention` wants the same layout, so
nothing is permuted. Converting to BTHD and back would cost two full contiguous copies of q/k/v per
call — ~1.2 GB each at 84k tokens — for no benefit.

**Head-chunked routing.** The routing intermediates are `[B, H, N, N]`; at 84k tokens N=656 and a
full-width sort workspace runs into the hundreds of MB. Heads are processed in chunks to bound it.

**Index packing avoids `argsort`.** `BlockMask` needs the selected block ids packed to the front of
each row. The obvious formulation ranks blocks as `selected * block_idx` and recovers ids with
`argsort` + `gather`, but that has two problems: a *selected* block 0 ranks as `0`, identical to
every *unselected* block, so its survival depends on tie-breaking — and `torch.argsort` does not
guarantee stability without `stable=True`.

In practice the tie resolves favourably (tested across 6 selection patterns on torch 2.12 / CUDA,
block 0 survived every time, because ties fall back to original order). It is still relying on
unspecified behaviour. Ranking as `idx + 1` and reading ids out of `sort(...).values` removes the
dependency, and skips materialising the int64 `order` tensor that `gather` would need — 193 MB at
N=656.

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
