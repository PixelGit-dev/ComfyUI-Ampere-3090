# ComfyUI-SolAttn-Ampere

Block-sparse attention for **MiniMax H3** on **Ampere and newer** (RTX 3090 / sm_80+).

Sol-Attn routes attention at 128-token granularity — it mean-pools K and Q into blocks, scores them
cheaply, and runs exact attention only on the blocks that matter. On long video sequences that skips
roughly 80% of the work.

## Why this exists

Sol-Attn is training-free. There is no retraining, no calibration pass, and no dependency on a
particular tensor-core generation — the saving is arithmetic. Routing is plain tensor ops and the
sparse kernel is `torch.compile(flex_attention)`, which has supported Ampere since PyTorch 2.5.

Ampere is also where it helps most. On sm_86, SageAttention 2.2 quietly runs its weakest path —
INT8 QK but FP16 PV with FP32 accumulate, the same PV path as FlashAttention-2 — because the
SageAttention2++ accumulator is gated to sm89 and above. Sparsity attacks a different term
entirely: it removes operations rather than making each one cheaper.

See [docs/why.md](docs/why.md) for the full analysis.

## Install

Drop the folder into `ComfyUI/custom_nodes/`. No dependencies beyond PyTorch — `flex_attention`
requires **torch >= 2.5** and a working Triton.

## Use

Insert **Sol-Attn MiniMax H3** on the MODEL path, feeding *both* the scheduler and the guider:

```
UNETLoader -> MiniMax H3 Sigma Shift -> Sol-Attn MiniMax H3 -+-> BasicScheduler
                                                             +-> BasicGuider
```

A complete FL2VA workflow is in
[`example_workflows/minimax_h3_fl2va_solattn_3090.json`](example_workflows/minimax_h3_fl2va_solattn_3090.json).

### Nodes

**Sol-Attn MiniMax H3** — patches the model.

| Parameter                | Default    | Meaning                                                                             |
| ------------------------ | ---------- | ----------------------------------------------------------------------------------- |
| `tau`                    | 1.2        | Sparsity. Higher = faster, less accurate. 1.0–1.5 is speed-first.                   |
| `min_seq_len`            | 8192       | Below this, fall through to normal attention (SageAttention wins there).            |
| `protect_prefix`         | true       | Keep the complete text/conditioning/reference/audio prefix exact automatically.     |
| `dense_prefix_queries`   | false      | Also run prefix query rows densely for the strongest conditioning guard.            |
| `dense_first_percent`    | 0.2        | Keep the first 20% of denoising dense, matching the reference H3 policy.            |
| `dense_first_blocks`     | 2          | Keep H3's first two transformer blocks dense on every step.                         |
| `end_percent`            | 1.0        | Return to dense after this fraction of denoising. 1.0 disables; try 0.9.            |
| `dense_blocks`           | `""`       | Blocks to keep dense, e.g. `0-2,-1`. Overrides `dense_first_blocks`.                |
| `approx_correction`      | true       | Fold skipped blocks back in via their pilot scores instead of dropping them.        |
| `cornish_fisher`         | false      | Correct the routing threshold for score skew/kurtosis instead of assuming Gaussian. |
| `morton`                 | false      | Z-order the video tokens so each block is a compact 3D neighbourhood.               |
| `morton_curve`           | `2d_frame` | `2d_frame` orders within each frame; `3d` interleaves t/h/w.                        |
| `preserve_prefix_blocks` | 0          | Optional manual minimum when automatic prefix detection is unavailable.             |
| `log_every`              | 200        | Log measured density every N calls.                                                 |

The last six are new and default to preserving previous behaviour, except
`approx_correction`, which is on because dropping the long tail was a quality
regression against the published method. See [docs/tuning.md](docs/tuning.md).

**Sol-Attn Stats** — reports whether the kernel actually ran. Wire the decoded `IMAGE` into `trigger`
so it reads *after* sampling. `sol_attn=0` means it never engaged; `last_density > 0.35` means little
speedup.

Tuning guidance and troubleshooting: [docs/tuning.md](docs/tuning.md).

## Scope

Intercepts MiniMax H3 self-attention only — pre-reshaped BHSD, head_dim 128, unmasked, bf16/fp16.
Cross-attention, other dtypes, and short sequences pass through untouched. Any kernel failure falls
back rather than aborting the generation. If another attention override was installed earlier in
the MODEL chain, it remains the fallback.

## Docs

- [why.md](docs/why.md) — why sparsity is architecture-neutral, SageAttention on sm_86, when this helps
- [benchmarks.md](docs/benchmarks.md) — all measurements, including the real-generation validation
- [implementation.md](docs/implementation.md) — internals and design notes
- [tuning.md](docs/tuning.md) — parameters, density, troubleshooting

## Credit

The algorithm is **Sol-Attn**, by NVlabs — training-free on-the-fly attention sparsification for
video generation. The block routing implemented here (128-token block means, the `q̄ @ k̄ᵀ` pilot
score, and the `mean + tau·std` threshold) is theirs.

- Paper: [Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification](https://arxiv.org/abs/2607.24027) (arXiv:2607.24027)
- Project page: <https://nvlabs.github.io/Sana/Sol-Attn/>
- Code: [NVlabs/Sana](https://github.com/NVlabs/Sana/tree/sol-engine), `sol-engine` branch

Sol-Attn's approximate-correction step — reusing below-threshold proxy scores to recover the
long-tail contribution of skipped blocks — is now implemented (`approx_correction`, on by default).

## License

[Apache License 2.0](LICENSE) — the same license as NVlabs/Sana. Attribution in [NOTICE](NOTICE).
