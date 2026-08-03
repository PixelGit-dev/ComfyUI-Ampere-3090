# SPDX-License-Identifier: Apache-2.0
# See NOTICE for attribution.

"""ComfyUI hook + node wiring Sol-Attn into MiniMax H3.

Comfy invokes an override as ``override(func, *args, **kwargs)`` (see
``comfy/ldm/modules/attention.py:159``), where ``func`` is the attention
implementation that would otherwise run -- so the fallback path naturally
lands on SageAttention when ComfyUI was started with ``--use-sage-attention``.

MiniMax H3's self-attention (``comfy/ldm/minimax/model.py:178-182``) hands us
``q/k/v`` as ``[1, heads, S, 128]`` with ``skip_reshape=True`` and expects
``[1, S, heads*128]`` back.
"""

from __future__ import annotations

import logging

import torch

from sol_attn_flex import sol_attn_flex, last_density

logger = logging.getLogger(__name__)

_H3_HEAD_DIM = 128

# Verification counters -- distinguishes "ran the kernel" from "silently fell
# back", which absence-of-errors cannot.
STATS = {"sol_attn": 0, "skipped_short": 0, "skipped_shape": 0, "failed": 0}


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def make_override(tau: float, min_seq_len: int, preserve_prefix_blocks: int,
                  log_every: int):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def fallback():
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        # Only intercept H3 self-attention: pre-reshaped BHSD, head_dim 128,
        # unmasked, half precision.  Everything else (cross-attn, other dtypes)
        # goes to the normal path.
        if (not skip_reshape or mask is not None or q.ndim != 4
                or q.shape[-1] != _H3_HEAD_DIM
                or q.dtype not in (torch.bfloat16, torch.float16)):
            STATS["skipped_shape"] += 1
            return fallback()

        B, H, S, D = q.shape
        if S < min_seq_len:
            # Routing overhead is not free and sparsity is weak at short
            # lengths; SageAttention wins below roughly 8k.
            STATS["skipped_short"] += 1
            return fallback()

        try:
            out = sol_attn_flex(
                q, k, v,
                scale=kwargs.get("scale"),
                tau=tau,
                preserve_prefix_blocks=preserve_prefix_blocks,
            )
        except Exception as exc:
            STATS["failed"] += 1
            logger.warning(
                f"[Sol-Attn] kernel failed ({type(exc).__name__}: {exc}); "
                f"falling back for this call"
            )
            return fallback()

        STATS["sol_attn"] += 1
        if log_every and STATS["sol_attn"] % log_every == 1:
            d = last_density()
            logger.info(
                f"[Sol-Attn] S={S} density={d:.3f} "
                f"(~{1.0 / max(d, 1e-6):.1f}x fewer blocks) tau={tau}"
            )

        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(B, S, H * D)

    return override


class SolAttnMiniMaxH3:
    """Patch MiniMax H3 to use Sol-Attn block-sparse attention."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {
                    "default": 1.2, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Sparsity temperature. Higher = skip more KV "
                               "blocks = faster, less accurate. 1.0-1.5 is the "
                               "speed-first range; drop to 0.6-0.8 if you see "
                               "artifacts.",
                }),
                "min_seq_len": ("INT", {
                    "default": 8192, "min": 0, "max": 200000, "step": 1024,
                    "tooltip": "Below this token count, fall through to the "
                               "normal attention path (SageAttention). Sol-Attn "
                               "only wins on long sequences.",
                }),
            },
            "optional": {
                "preserve_prefix_blocks": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Force-keep the first N 128-token blocks, which "
                               "in H3's [text | cond | audio | video] layout "
                               "protects the text/cond prefix from pruning. 0 "
                               "is speed-first; try 8-16 if prompt adherence "
                               "degrades.",
                }),
                "log_every": ("INT", {
                    "default": 200, "min": 0, "max": 100000, "step": 50,
                    "tooltip": "Log measured density every N calls. 0 disables.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = ("Block-sparse Sol-Attn attention for MiniMax H3. "
                   "Works on Ampere (RTX 3090) and newer.")

    def patch(self, model, enabled, tau, min_seq_len,
              preserve_prefix_blocks=0, log_every=200):
        if not enabled:
            return (model,)

        patched = model.clone()
        opts = patched.model_options.get("transformer_options", {}).copy()
        opts["optimized_attention_override"] = make_override(
            tau=tau, min_seq_len=min_seq_len,
            preserve_prefix_blocks=preserve_prefix_blocks,
            log_every=log_every,
        )
        patched.model_options["transformer_options"] = opts
        reset_stats()
        logger.info(
            f"[Sol-Attn] patched | tau={tau} min_seq_len={min_seq_len} "
            f"preserve_prefix_blocks={preserve_prefix_blocks}"
        )
        return (patched,)


class SolAttnStats:
    """Report how many attention calls actually used Sol-Attn.

    Takes an optional ``trigger`` so it can be sequenced *after* sampling --
    without it ComfyUI is free to execute this node before the sampler runs,
    which would report zeros.  Wire the decoded IMAGE into ``trigger``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("IMAGE", {
                    "tooltip": "Optional ordering input. Connect the decoded "
                               "IMAGE so stats are read after sampling.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "report"
    CATEGORY = "model_patches/optimization"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def report(self, trigger=None):
        d = last_density()
        ran = STATS["sol_attn"]
        txt = (f"sol_attn={ran} "
               f"skipped_short={STATS['skipped_short']} "
               f"skipped_shape={STATS['skipped_shape']} "
               f"failed={STATS['failed']} "
               f"last_density={d if d is None else round(d, 4)}")
        if ran == 0:
            txt += "  <- Sol-Attn never ran; check min_seq_len vs your resolution"
        elif d is not None and d > 0.35:
            txt += f"  <- density {d:.3f} is high; little speedup, raise tau"
        logger.info(f"[Sol-Attn] {txt}")
        return {"ui": {"text": [txt]}, "result": (txt,)}


NODE_CLASS_MAPPINGS = {
    "SolAttnMiniMaxH3": SolAttnMiniMaxH3,
    "SolAttnStats": SolAttnStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SolAttnMiniMaxH3": "Sol-Attn MiniMax H3",
    "SolAttnStats": "Sol-Attn Stats",
}
