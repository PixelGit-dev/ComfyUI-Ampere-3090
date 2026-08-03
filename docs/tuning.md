# Tuning and troubleshooting

## Read the stats node first

Wire the decoded `IMAGE` into **Sol-Attn Stats** → `trigger` so it reads after sampling. Without
that input ComfyUI may execute it before the sampler and report zeros.

```
sol_attn=1000  skipped_short=0  skipped_shape=0  failed=0  last_density=0.1689
```

| Field | Meaning |
|---|---|
| `sol_attn` | Calls that used the sparse kernel. Should equal *blocks x steps* (H3 has 50 blocks). |
| `skipped_short` | Fell through because `S < min_seq_len`. |
| `skipped_shape` | Not H3 self-attention — cross-attention, wrong dtype, masked. Some is normal. |
| `failed` | Kernel raised and fell back. Should be 0. |
| `last_density` | Fraction of KV blocks kept on the last call. **The number that matters.** |

## Density is the headline number

Speedup tracks `1 / density` more than anything else. Check it against your run, not against a
benchmark table — **real activations route less sparsely than random tensors**, by roughly 60% in
measured practice.

| Observed density | Reading |
|---|---|
| < 0.10 | Very sparse. Watch quality. |
| 0.10 – 0.25 | Normal working range. |
| 0.25 – 0.35 | Modest speedup. Raise `tau`. |
| > 0.35 | Little benefit — the overhead is eating the gain. |

Density naturally *falls* across a run as denoising sharpens attention, so later steps are cheaper
than early ones. `last_density` is the final call; the `log_every` lines are periodic samples.

## Parameters

**`tau`** — sparsity temperature, the main knob. Higher skips more blocks.

| tau | Density (random tensors, 32k) |
|---|---|
| 0.6 | 0.283 |
| 0.8 | 0.221 |
| 1.0 | 0.169 |
| 1.2 | 0.125 |
| 1.5 | 0.078 |

Real densities run higher. `1.0–1.5` is the speed-first range; drop to `0.6–0.8` if you see
artifacts. Change one variable at a time and read the density log rather than trusting these ratios.

**`min_seq_len`** (default 8192) — below this, hand the call back to normal attention. Sol-Attn is
only ~1.5x at 4k tokens and routing overhead is real, so SageAttention wins on short sequences.
Lower it only if you have measured that it helps.

**`preserve_prefix_blocks`** (default 0) — force-keep the first N 128-token blocks. H3 packs the
sequence as `[text | cond | audio | video]`, so this protects the text prefix from pruning. Try
`8–16` if prompt adherence degrades. Costs a little speed.

**`log_every`** (default 200) — density logging interval. 0 disables.

## Quality

Sol-Attn is lossy in a way SageAttention is not — at density 0.20 you are dropping 80% of attention
blocks. The failure mode specific to H3 is **audio-video sync**: video, audio and text share one
sequence, so block pruning can sever cross-modal attention in a way no speed benchmark reveals.

If output degrades, in order: lower `tau`, then set `preserve_prefix_blocks` to `8–16`, then compare
against a bypassed run at the same seed.

## Troubleshooting

### `sol_attn=0` — never engaged

Sequence is below `min_seq_len`, or the shape check rejected everything. Check the logged `S` value
against your resolution. Raise resolution/length, or lower `min_seq_len`.

### `failed > 0`

The kernel raised and fell back. The warning line names the exception. Most likely a
`torch.compile`/Triton problem — confirm `flex_attention` imports and compiles standalone.

### Long first-run pause

One-time `torch.compile` cost, 3–4 s. `__init__.py` warms it up at startup to keep it out of the
first generation.

### Crash in `patchify_video` — not this node

```
RuntimeError: shape '[1, 24, 1, 1, 21, 2, 15, 2]' is invalid for input of size 30960
```

This is stock ComfyUI, raised before any attention call, and unrelated to Sol-Attn or SageAttention.

**Width and height must be multiples of 32, not 16.** The VAE downsamples 16x and the DiT patchifies
2x2. A height of 688 gives `688/16 = 43` — odd — and `patchify_video`
(`comfy/ldm/minimax/model.py:56-57`) floors to 21 blocks, then fails the reshape.

It only bites with **keyframes connected**: `model.py:508` pads the main video latent to patch size,
but `_cond_video_rows` (`model.py:477`) does not pad the keyframe latent. The same resolution runs
fine for t2va.

The `step=32` on the width/height widgets only constrains the spinner arrows — typed values, and
anything linked in from a resolution selector or `GetImageSize`, bypass it entirely.

Valid sizes: `864x480`, `1344x768`, `480x672`, `480x704`, `608x1088`.

## Getting a real baseline

Benchmarks here are per-call attention timings. For the end-to-end number on your hardware, set
`enabled` to false on the Sol-Attn node and rerun the same seed. Everything else stays identical, so
the delta is clean.

Expect it to be well below the attention-level speedup: quantized weights add dequant overhead to
the linear layers, and on a card where the model is staged for dynamic VRAM loading, part of each
step is weight transfer that no attention kernel can touch.
