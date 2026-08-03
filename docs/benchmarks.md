# Benchmarks

All measurements on the same machine.

**Environment:** RTX 3090 (sm_86, 25.8 GB) · torch 2.12.1+cu130 · triton-windows 3.7.1 ·
sageattention 2.2.0 · ComfyUI 0.30.1

**Shape:** MiniMax H3's real attention config — `B=1, H=56, D=128`, bf16.

---

## Kernel only (synthetic mask)

Feasibility check. Random block mask at ~0.10 density; routing cost **excluded**.

| Tokens | SDPA | SageAttention 2.2 | flex_attention | vs SDPA | vs sage |
|---|---|---|---|---|---|
| 12,288 | 98.04 ms | 44.40 ms | 9.64 ms (d=0.128) | 10.2x | 4.6x |
| 32,768 | 702.72 ms | 301.63 ms | 58.08 ms (d=0.111) | 12.1x | 5.2x |

Compile + warmup: 3.6 s first call, one-time.

## Full path (routing included)

Real Sol-Attn routing, still on random tensors.

| Tokens | SDPA | SageAttention 2.2 | Sol-Attn | Density | vs SDPA | vs sage | Peak VRAM |
|---|---|---|---|---|---|---|---|
| 12,288 | 117.6 ms | 52.8 ms | 16.6 ms | 0.143 | 7.08x | **3.18x** | 1.12 GB |
| 32,768 | 848.4 ms | 368.5 ms | 93.3 ms | 0.125 | 9.09x | **3.95x** | 3.59 GB |

Comparing against the kernel-only run: 58 ms → 93 ms at 32k means **routing costs ~30%** of total
runtime, dominated by the `torch.sort` over `[B, hc, N, N]`. That is the obvious remaining
optimisation — a cumsum+scatter pack, or caching routing across adjacent denoise steps.

## Density vs tau

T = 32,768, random tensors.

| tau | Density | Time |
|---|---|---|
| 0.6 | 0.283 | 203.7 ms |
| 0.8 | 0.221 | 161.3 ms |
| 1.0 | 0.169 | 123.6 ms |
| 1.2 | 0.125 | 94.0 ms |
| 1.5 | 0.078 | 61.2 ms |

## Correctness

Forcing near-full density (`tau=-50`) must reproduce SDPA.

| Tokens | Density | Max abs err | Rel err |
|---|---|---|---|
| 2,048 | 1.000 | 0.00098 | 0.0036 |
| 4,096 | 1.000 | 0.00098 | 0.0055 |

Within bf16 tolerance. This is also the test that would catch a broken index-packing path.

---

## Real generation

FL2VA, 480x672, 124 frames, 20 steps, `tau=1.2`, INT8 transformer + Q4_0 GGUF text encoder.

```
S=17627  density=0.210 -> 0.192 over the run
sol_attn=1000  skipped_short=0  skipped_shape=0  failed=0  last_density=0.1689
20/20 [01:59, ~6.0 s/it]   total 168.67 s cold
```

**`sol_attn=1000` is exactly 50 blocks x 20 steps** — every attention call intercepted, zero
fallbacks, zero errors. It also confirms guidance-free sampling: one forward per step, no CFG
doubling.

### The important correction

**Real density at `tau=1.2` is 0.17–0.21, against 0.125 on random tensors — roughly 60% higher.**
Real H3 activations route less sparsely than random data. The synthetic speedup table above
therefore **overstates** what this delivers in practice. Still well under the 0.35 threshold at
which the approach stops being worthwhile.

Density *falls* as denoising progresses (0.210 at step 0 → 0.192 by step 16): attention sharpens as
structure emerges, so later steps get cheaper. `last_density=0.1689` is the final call, whereas the
logged values sample every 200th.

### Still missing

**No A/B baseline.** Nothing here establishes what SageAttention would have done at S=17627, so the
real-world end-to-end speedup is unmeasured. To get it: set `enabled` to false on the Sol-Attn node
and rerun the same seed — everything else identical, so the delta is clean.

Note also `MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged` — with a ~20 GB model on a
25.8 GB card, part of each 6.0 s step is weight transfer that Sol-Attn cannot affect. Expect the
end-to-end figure to compress well below the attention-level one.

## Reproducing

The benchmark and verification scripts are not shipped with the pack. To rebuild them, time
`F.scaled_dot_product_attention`, `sageattn(..., tensor_layout="HND")`, and `sol_attn_flex(...)`
against `[1, 56, T, 128]` bf16 tensors, and read density back via `sol_attn_flex.last_density()`.
