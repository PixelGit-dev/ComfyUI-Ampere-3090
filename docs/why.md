# Why this exists

## Sparsity is architecture-neutral

Sol-Attn is training-free: no retraining, no calibration pass, no quantized weights. It is two steps
of ordinary PyTorch:

1. Mean-pool K and Q into 128-token blocks, score `q̄ @ k̄ᵀ`, keep blocks above `mean + tau·std`
   plus a diagonal window.
2. Hand that selection to `torch.compile(flex_attention)` as a `BlockMask`.

Nothing in that requires a particular tensor-core generation. Routing is `mean`, `var`, and a
matmul. Execution is `flex_attention`, which lowers to a Triton-generated FlashAttention-2-style
kernel and has supported Ampere (sm_80) since PyTorch 2.5. No CUTLASS, no CuTe DSL, no FP8/FP4, no
`wgmma`/TMA, no inline PTX.

It is worth being precise about one thing often claimed for sparse-attention kernels: **FlashAttention-3
is Hopper-only** — it depends on `wgmma` and TMA. Consumer cards do not run FA3 regardless of
generation, so "FA3-class performance" is not a reason to require newer hardware here.

The speedup is **arithmetic**: at density 0.10 you skip ~90% of the attention blocks. That helps any
GPU. If anything Ampere benefits more, being more compute-starved relative to its bandwidth than
later parts.

## SageAttention 2.2 gives Ampere its weak path

The incumbent on this hardware is SageAttention. On sm_86 it is not doing what its version number
suggests. From `sageattention/core.py`:

```python
elif arch in {"sm80", "sm86", "sm87"}:
    return sageattn_qk_int8_pv_fp16_cuda(..., pv_accum_dtype="fp32")
elif arch == "sm89":
    ...
    pv_accum_dtype = "fp32+fp16"   # SageAttention2++
    return sageattn_qk_int8_pv_fp8_cuda(...)
```

Ampere gets **INT8 QK, but FP16 PV with FP32 accumulate** — the same PV path as FlashAttention-2.
The SageAttention2++ accumulator that names the 2.2 release is gated to sm89/sm90/sm100/sm120.

That said, it is not weak in absolute terms. Measured on a 3090 it delivers **2.2–2.3x over SDPA**
— at 301 ms for 32k tokens that is ~102 TFLOPS, above the card's bf16 peak, so the INT8 tensor cores
are doing real work. An earlier estimate of 1.2–1.4x for this path was wrong.

## Why sparsity wins anyway

Quantization and sparsity attack different terms. SageAttention makes each operation cheaper;
Sol-Attn removes operations entirely. At density 0.20 that is a 5x arithmetic reduction, which
compounds against SageAttention's fixed ~2.3x rather than competing with it.

They are not composable in practice: `flex_attention` runs bf16 Triton kernels and cannot call
SageAttention's INT8 kernel on the gathered blocks. So it is one or the other per attention call —
hence the `min_seq_len` gate, which hands short sequences back to SageAttention.

## Where this actually helps

MiniMax H3 packs everything into **one self-attention sequence** — `[text | cond rows | audio | video]`
(`comfy/ldm/minimax/model.py:6`), a single `optimized_attention` call per block (`model.py:181`).

Compression is 16x spatial from the VAE (`space_down=(2,2,2,2,1,1)`) and 4x temporal
(`time_down=(1,2,2,1,1,1)`), plus the DiT's `patch_size=(1,2,2)` — **32x spatial / 4x temporal**
total. So:

| Output | Approx. tokens |
|---|---|
| 480p, 5s | ~12k |
| 480p, 15s | ~35k |
| 720p, 5s | ~28k |
| 720p, 15s | ~84k |

Attention is O(N²) while the FFN is O(N), so at these lengths attention dominates step time and
sparsity has the most room to work. Below ~8k tokens the picture inverts — routing overhead is not
free and sparsity is weak — which is what `min_seq_len` exists for.

## What limits the end-to-end gain

Attention carries no weights, so quantized transformers (GGUF, INT8) add dequant overhead to every
linear layer while leaving attention untouched. That *lowers* attention's share of step time, and
the end-to-end speedup is correspondingly below the attention-level number.

The same applies to weight streaming. On a 24 GB card running a ~20 GB model, ComfyUI stages weights
dynamically and a real share of each step is transfer, not compute — which Sol-Attn cannot touch.
