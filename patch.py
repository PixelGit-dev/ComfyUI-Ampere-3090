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
import math

import torch

from sol_attn_flex import sol_attn_flex, last_density

logger = logging.getLogger(__name__)

_H3_HEAD_DIM = 128

# Verification counters -- distinguishes "ran the kernel" from "silently fell
# back", which absence-of-errors cannot.
STATS = {
    "sol_attn": 0,
    "skipped_short": 0,
    "skipped_shape": 0,
    "skipped_early": 0,
    "skipped_block": 0,
    "prefix_blocks": 0,
    "failed": 0,
}

_VIDEO_TOKENS_KEY = "sol_attn_ampere_video_tokens"
_SIGMA_KEY = "sol_attn_ampere_sigma"
_LAYOUT_WRAPPER_KEY = "sol_attn_ampere_layout"


def reset_stats():
    for key in STATS:
        STATS[key] = 0


class _RoutingPolicy:
    """Track H3 sampling progress and block position without patching the model."""

    def __init__(self, dense_first_percent, dense_first_blocks,
                 sigma_hi, sigma_lo, blocks_per_step=50):
        self.dense_first_percent = float(dense_first_percent)
        self.dense_first_blocks = int(dense_first_blocks)
        self.sigma_hi = float(sigma_hi)
        self.sigma_lo = float(sigma_lo)
        self.sigma_span = max(self.sigma_hi - self.sigma_lo, 1e-8)
        self.blocks_per_step = int(blocks_per_step)
        self.last_sigma = None
        self.block_index = 0

    def reset(self):
        self.last_sigma = None
        self.block_index = 0

    def _sigma(self, transformer_options):
        cached = (transformer_options or {}).get(_SIGMA_KEY)
        if isinstance(cached, (float, int)):
            return float(cached)
        sigmas = (transformer_options or {}).get("sigmas")
        if torch.is_tensor(sigmas) and sigmas.numel():
            return float(sigmas.flatten()[0])
        return None

    def decide(self, transformer_options):
        sigma = self._sigma(transformer_options)
        if sigma is not None and sigma != self.last_sigma:
            self.last_sigma = sigma
            self.block_index = 0

        index = self.block_index
        self.block_index = (self.block_index + 1) % self.blocks_per_step

        if sigma is None:
            # Older ComfyUI builds do not publish sigma. Preserve performance
            # instead of accidentally treating every call as the first step.
            progress = 1.0
        else:
            progress = min(max((self.sigma_hi - sigma) / self.sigma_span, 0.0), 1.0)

        if progress < self.dense_first_percent:
            return False, "skipped_early", index, progress
        if index < self.dense_first_blocks:
            return False, "skipped_block", index, progress
        return True, None, index, progress


def _capture_video_tokens(executor, x, timestep, context,
                          transformer_options={}, minimax_payload=None, **kwargs):
    """Publish once-per-model-call values used by the attention override."""
    if isinstance(transformer_options, dict):
        sigmas = transformer_options.get("sigmas")
        if torch.is_tensor(sigmas) and sigmas.numel():
            transformer_options[_SIGMA_KEY] = float(sigmas.flatten()[0])
        else:
            transformer_options.pop(_SIGMA_KEY, None)
        if isinstance(x, (tuple, list)) and x:
            video = x[0]
            if torch.is_tensor(video) and video.ndim == 5:
                # H3 patchifies the video latent with (1, 2, 2), padding H/W up.
                transformer_options[_VIDEO_TOKENS_KEY] = (
                    int(video.shape[2])
                    * math.ceil(int(video.shape[3]) / 2)
                    * math.ceil(int(video.shape[4]) / 2)
                )
    return executor(
        x, timestep, context, transformer_options,
        minimax_payload=minimax_payload, **kwargs
    )


def make_override(tau: float, min_seq_len: int, preserve_prefix_blocks: int,
                  log_every: int, policy: _RoutingPolicy,
                  protect_prefix: bool, dense_prefix_queries: bool,
                  fallback_override=None):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):

        def fallback():
            if fallback_override is not None:
                return fallback_override(
                    func, q, k, v, heads, mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape, **kwargs
                )
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

        transformer_options = kwargs.get("transformer_options") or {}
        use_sparse, skip_reason, _, _ = policy.decide(transformer_options)
        if not use_sparse:
            STATS[skip_reason] += 1
            return fallback()

        prefix_blocks = int(preserve_prefix_blocks)
        if protect_prefix:
            video_tokens = transformer_options.get(_VIDEO_TOKENS_KEY)
            if isinstance(video_tokens, int) and 0 < video_tokens <= S:
                prefix_tokens = S - video_tokens
                prefix_blocks = max(
                    prefix_blocks,
                    (prefix_tokens + 127) // 128,
                )

        try:
            out = sol_attn_flex(
                q, k, v,
                scale=kwargs.get("scale"),
                tau=tau,
                preserve_prefix_blocks=prefix_blocks,
                dense_prefix_queries=dense_prefix_queries,
            )
        except Exception as exc:
            STATS["failed"] += 1
            logger.warning(
                f"[Sol-Attn] kernel failed ({type(exc).__name__}: {exc}); "
                f"falling back for this call"
            )
            return fallback()

        STATS["sol_attn"] += 1
        STATS["prefix_blocks"] = prefix_blocks
        if log_every and STATS["sol_attn"] % log_every == 1:
            d = last_density()
            logger.info(
                f"[Sol-Attn] S={S} density={d:.3f} "
                f"(~{1.0 / max(d, 1e-6):.1f}x fewer blocks) "
                f"prefix_blocks={prefix_blocks} tau={tau}"
            )

        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(B, S, H * D)

    override._sol_attn_ampere = True
    override._sol_attn_ampere_fallback = fallback_override
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
                    "tooltip": "Manual minimum number of exact 128-token prefix "
                               "blocks. Normally leave this at 0 and use automatic "
                               "prefix protection.",
                }),
                "log_every": ("INT", {
                    "default": 200, "min": 0, "max": 100000, "step": 50,
                    "tooltip": "Log measured density every N calls. 0 disables.",
                }),
                "protect_prefix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically keep H3's complete text, visual "
                               "conditioning, reference, and audio prefix exact. "
                               "Recommended for prompt adherence and A/V sync.",
                }),
                "dense_prefix_queries": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also make prefix query rows fully dense. This is "
                               "the strongest quality guard, at additional cost.",
                }),
                "dense_first_percent": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Use normal dense attention for the first fraction "
                               "of denoising. The reference H3 policy uses 0.2.",
                }),
                "dense_first_blocks": ("INT", {
                    "default": 2, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Use normal dense attention for the first N of "
                               "H3's 50 transformer blocks on every step. The "
                               "reference H3 policy uses 2.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = ("Block-sparse Sol-Attn attention for MiniMax H3. "
                   "Works on Ampere (RTX 3090) and newer.")

    def patch(self, model, enabled, tau, min_seq_len,
              preserve_prefix_blocks=0, log_every=200,
              protect_prefix=True, dense_prefix_queries=False,
              dense_first_percent=0.2, dense_first_blocks=2):
        if not enabled:
            return (model,)

        patched = model.clone()
        opts = patched.model_options.get("transformer_options", {}).copy()
        prior_override = opts.get("optimized_attention_override")
        if getattr(prior_override, "_sol_attn_ampere", False):
            prior_override = prior_override._sol_attn_ampere_fallback

        try:
            model_sampling = patched.get_model_object("model_sampling")
            sigma_hi = float(model_sampling.percent_to_sigma(0.0))
            sigma_lo = float(model_sampling.percent_to_sigma(1.0))
        except Exception:
            sigma_hi, sigma_lo = 1.0, 0.0

        policy = _RoutingPolicy(
            dense_first_percent=dense_first_percent,
            dense_first_blocks=dense_first_blocks,
            sigma_hi=sigma_hi,
            sigma_lo=sigma_lo,
        )
        opts["optimized_attention_override"] = make_override(
            tau=tau, min_seq_len=min_seq_len,
            preserve_prefix_blocks=preserve_prefix_blocks,
            log_every=log_every,
            policy=policy,
            protect_prefix=protect_prefix,
            dense_prefix_queries=dense_prefix_queries,
            fallback_override=prior_override,
        )
        patched.model_options["transformer_options"] = opts

        try:
            import comfy.patcher_extension
            patched.remove_wrappers_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                _LAYOUT_WRAPPER_KEY,
            )
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                _LAYOUT_WRAPPER_KEY,
                _capture_video_tokens,
            )
        except Exception as exc:
            logger.warning(
                f"[Sol-Attn] runtime context capture unavailable "
                f"({type(exc).__name__}: {exc}); using compatibility fallbacks"
            )

        reset_stats()
        policy.reset()
        logger.info(
            f"[Sol-Attn] patched | tau={tau} min_seq_len={min_seq_len} "
            f"protect_prefix={protect_prefix} "
            f"dense_first={dense_first_percent:.0%} "
            f"dense_blocks={dense_first_blocks}"
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
               f"skipped_early={STATS['skipped_early']} "
               f"skipped_block={STATS['skipped_block']} "
               f"prefix_blocks={STATS['prefix_blocks']} "
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
