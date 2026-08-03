# SPDX-License-Identifier: Apache-2.0
#
# The routing algorithm is Sol-Attn by NVlabs (arXiv:2607.24027,
# https://github.com/NVlabs/Sana tree/sol-engine), Apache-2.0.  See NOTICE.

"""Sol-Attn block-sparse attention via torch flex_attention.

Works on any GPU flex_attention supports (sm_80 Ampere and newer) -- there is
nothing architecture-specific here.  Routing is done in pure torch at 128-token
granularity, then executed by ``torch.compile``d ``flex_attention``.

Note this implements Sol-Attn's routing and sparse execution only.  The paper's
approximate-correction step (reusing below-threshold proxy scores to recover the
long-tail contribution of skipped blocks) is not implemented; blocks below the
threshold are dropped outright.

Implementation notes:

* **Operates natively in BHSD** ``[B, H, S, D]`` -- the layout MiniMax H3 hands
  us and the one flex_attention wants, so q/k/v are never permuted.
* **Index packing ranks selected blocks as ``idx + 1``** rather than
  ``selected * idx``, so a selected block 0 stays distinguishable from the
  unselected zeros and survival does not depend on sort tie-breaking.  Reading
  ids from ``sort(...).values`` also avoids an int64 ``argsort`` index tensor
  (193 MB at N=656).
* **Head-chunked routing** so the transient sort workspace stays bounded at
  long sequences.
* **fp16 tolerated** in addition to bf16.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# flex_attention's compiled kernel requires 128-token blocks.
FLEX_BLOCK = 128

# Cap on heads processed per routing chunk; keeps the [b, hc, N, N] sort
# workspace small at long sequences.
_MAX_ROUTE_ELEMS = 8 * 656 * 656

_compiled_flex = None
_last_density = None


def _get_compiled_flex():
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


def last_density() -> float | None:
    """Fraction of KV blocks kept on the most recent call (None if unused)."""
    return _last_density


def _build_routing(q_h, k_h, tau, preserve_prefix_blocks):
    """Sol-Attn diag routing.

    q_h, k_h: [B, H, T_pad, D].  Returns (kv_num_blocks, kv_indices, density).

    A KV block is kept when its pilot score ``q_bar @ kc`` exceeds
    ``mean + tau * std`` of the score distribution, plus a local diagonal
    window that is always exact.
    """
    B, H, T_pad, D = q_h.shape
    N = T_pad // FLEX_BLOCK
    dev = q_h.device

    # 128-token block means, in fp32 -- the variance term is numerically poor
    # in bf16 and this tensor is small ([B, H, N, D]).
    kc_all = k_h.view(B, H, N, FLEX_BLOCK, D).mean(dim=3).float()
    qb_all = q_h.view(B, H, N, FLEX_BLOCK, D).mean(dim=3).float()

    blk = torch.arange(N, device=dev, dtype=torch.int32)
    diag = (blk.view(N, 1) - blk.view(1, N)).abs() <= 1  # [N, N]

    kv_num_blocks = torch.empty(B, H, N, dtype=torch.int32, device=dev)
    kv_indices = torch.empty(B, H, N, N, dtype=torch.int32, device=dev)

    chunk = max(1, min(H, _MAX_ROUTE_ELEMS // max(1, N * N)))
    for h0 in range(0, H, chunk):
        h1 = min(H, h0 + chunk)
        kc = kc_all[:, h0:h1]                     # [B, hc, N, D]
        qb = qb_all[:, h0:h1]

        kc_mean = kc.mean(dim=2, keepdim=True)
        kc_var = kc.var(dim=2, unbiased=False, keepdim=True)
        mean_term = (qb @ kc_mean.transpose(-1, -2)).squeeze(-1)          # [B,hc,N]
        var_term = ((qb * qb) @ kc_var.transpose(-1, -2)).squeeze(-1)
        thresh = mean_term + tau * torch.sqrt(var_term.clamp_min(0.0) + 1e-6)

        score = qb @ kc.transpose(-1, -2)                                  # [B,hc,N,N]
        sel = score > thresh.unsqueeze(-1)
        sel |= diag
        if preserve_prefix_blocks > 0:
            # Text (and, at larger values, cond/audio) rows are a contiguous
            # prefix in H3's packed layout: [text | cond | audio | video].
            sel[..., :preserve_prefix_blocks] = True

        kv_num_blocks[:, h0:h1] = sel.sum(-1).to(torch.int32)

        # Pack selected block ids to the front.  rank = idx+1 for selected,
        # 0 otherwise, so a selected block 0 (rank 1) still outranks every
        # unselected block.  Sorting descending then subtracting 1 recovers
        # the ids; trailing garbage sits past kv_num_blocks and is never read.
        rank = torch.where(sel, blk + 1, torch.zeros_like(blk))
        vals = torch.sort(rank, dim=-1, descending=True).values
        kv_indices[:, h0:h1] = (vals - 1).clamp_min_(0)

        del score, sel, rank, vals

    density = kv_num_blocks.float().mean().item() / N
    return kv_num_blocks, kv_indices, density


def sol_attn_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.2,
    preserve_prefix_blocks: int = 0,
) -> torch.Tensor:
    """Block-sparse attention over BHSD tensors ``[B, H, S, D]``.

    Returns the same shape.  Raises on unsupported inputs so the caller can
    fall back.
    """
    global _last_density

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape [B, H, S, D]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Sol-Attn requires bf16/fp16, got {q.dtype}")
    if q.device.type != "cuda":
        raise ValueError("Sol-Attn requires CUDA tensors")

    B, H, T, D = q.shape
    scale = D ** -0.5 if scale is None else float(scale)

    if T == 0:
        return q.clone()

    T_pad = ((T + FLEX_BLOCK - 1) // FLEX_BLOCK) * FLEX_BLOCK
    if T_pad != T:
        pad = (0, 0, 0, T_pad - T)  # pad the S dim of [B, H, S, D]
        qp, kp, vp = F.pad(q, pad), F.pad(k, pad), F.pad(v, pad)
    else:
        qp, kp, vp = q, k, v

    qp = qp.contiguous()
    kp = kp.contiguous()
    vp = vp.contiguous()

    kv_num_blocks, kv_indices, density = _build_routing(
        qp, kp, tau, preserve_prefix_blocks
    )
    _last_density = density

    from torch.nn.attention.flex_attention import BlockMask

    def boundary(b, h, q_idx, kv_idx):
        # Zero-padded rows/cols would otherwise score 0 (not -inf) and leak in.
        return (q_idx < T) & (kv_idx < T)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices,
        BLOCK_SIZE=FLEX_BLOCK,
        mask_mod=boundary,
        seq_lengths=(T_pad, T_pad),
    )

    out = _get_compiled_flex()(qp, kp, vp, block_mask=block_mask, scale=scale)
    return out[:, :, :T, :]


def warmup(device=None) -> bool:
    """Pay the one-time torch.compile cost up front. Returns success."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return False
    try:
        from torch.nn.attention.flex_attention import create_block_mask
        B, H, T, D = 1, 1, 256, 128
        q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        def noop(b, h, qi, kv):
            return qi >= 0

        bm = create_block_mask(noop, B, H, T, T, device=device, BLOCK_SIZE=FLEX_BLOCK)
        _get_compiled_flex()(q, k, v, block_mask=bm)
        torch.cuda.synchronize(device)
        logger.info("[Sol-Attn] flex_attention compiled (warmup done)")
        return True
    except Exception as exc:
        logger.warning(f"[Sol-Attn] warmup failed: {type(exc).__name__}: {exc}")
        return False


__all__ = ["sol_attn_flex", "warmup", "last_density", "FLEX_BLOCK"]
