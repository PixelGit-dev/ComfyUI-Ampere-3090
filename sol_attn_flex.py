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
* **Linear-time index packing** uses an int32 prefix sum and scatter-add rather
  than sorting every routing row.
* **Head-chunked routing** so dense pilot-score intermediates stay bounded at
  long sequences.
* **fp16 tolerated** in addition to bf16.
* **No full-tensor repacking or padding.** FlexAttention accepts H3's strided
  BHSD views and arbitrary sequence lengths; only the small block summaries
  are materialized.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

# flex_attention's compiled kernel requires 128-token blocks.
FLEX_BLOCK = 128

# Cap on heads processed per routing chunk; keeps the [b, hc, N, N] pilot
# score and packing workspace small at long sequences.
_MAX_ROUTE_ELEMS = 8 * 656 * 656

_compiled_flex = None
_last_density = None
_geometry_cache = OrderedDict()
_MAX_GEOMETRIES = 4


def _get_compiled_flex():
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


def last_density() -> float | None:
    """Fraction of KV blocks kept on the most recent call (None if unused)."""
    if _last_density is None:
        return None
    # Keep this synchronization out of the hot path.  The caller only asks on
    # an occasional log line or after sampling has completed.
    return float(_last_density.detach().item())


def _routing_geometry(device, blocks):
    """Return cached block ids and the exact local window for this shape."""
    key = (device.type, device.index, blocks)
    cached = _geometry_cache.get(key)
    if cached is not None:
        _geometry_cache.move_to_end(key)
        return cached

    ids = torch.arange(blocks, device=device, dtype=torch.int32)
    local = (ids.view(blocks, 1) - ids.view(1, blocks)).abs() <= 1
    _geometry_cache[key] = (ids, local)
    if len(_geometry_cache) > _MAX_GEOMETRIES:
        _geometry_cache.popitem(last=False)
    return ids, local


def _pool_blocks(x):
    """Mean-pool a possibly strided BHSD tensor without padding/copying it."""
    B, H, T, D = x.shape
    full = T // FLEX_BLOCK
    pieces = []
    if full:
        body = x[:, :, :full * FLEX_BLOCK, :]
        pieces.append(
            body.view(B, H, full, FLEX_BLOCK, D).mean(
                dim=3, dtype=torch.float32
            )
        )
    if full * FLEX_BLOCK != T:
        pieces.append(
            x[:, :, full * FLEX_BLOCK:, :].mean(
                dim=2, dtype=torch.float32, keepdim=True
            )
        )
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)


def _pack_selected(selected, block_ids):
    """Pack selected KV ids in ascending order using O(N) row operations."""
    counts = selected.sum(dim=-1, dtype=torch.int32)
    positions = selected.cumsum(dim=-1, dtype=torch.int32).sub_(1).clamp_min_(0)
    source = torch.where(selected, block_ids, 0)
    packed = torch.zeros_like(positions)
    # Selected ids have unique destinations. Unselected entries only add zero,
    # including the block-0 case, so collisions are harmless and deterministic.
    packed.scatter_add_(-1, positions, source)
    return counts, packed


def _build_routing(q_h, k_h, tau, preserve_prefix_blocks,
                   dense_prefix_queries):
    """Sol-Attn diag routing.

    q_h, k_h: [B, H, T, D].  Returns (kv_num_blocks, kv_indices, density).

    A KV block is kept when its pilot score ``q_bar @ kc`` exceeds
    ``mean + tau * std`` of the score distribution, plus a local diagonal
    window that is always exact.
    """
    B, H, T, D = q_h.shape
    N = (T + FLEX_BLOCK - 1) // FLEX_BLOCK
    dev = q_h.device

    # Direct fp32 accumulation avoids bf16 variance noise. The partial tail is
    # divided by its real token count rather than by a zero-padded 128.
    kc_all = _pool_blocks(k_h)
    qb_all = _pool_blocks(q_h)

    blk, diag = _routing_geometry(dev, N)

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
            if dense_prefix_queries:
                sel[..., :preserve_prefix_blocks, :] = True

        counts, packed = _pack_selected(sel, blk)
        kv_num_blocks[:, h0:h1] = counts
        kv_indices[:, h0:h1] = packed

        del score, sel, counts, packed

    # Leave density on-device. Calling .item() here serialized every attention
    # layer with the CPU even when density logging was infrequent.
    density = kv_num_blocks.sum(dtype=torch.float32) / float(B * H * N * N)
    return kv_num_blocks, kv_indices, density


def sol_attn_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.2,
    preserve_prefix_blocks: int = 0,
    dense_prefix_queries: bool = False,
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

    kv_num_blocks, kv_indices, density = _build_routing(
        q, k, tau, preserve_prefix_blocks, dense_prefix_queries
    )
    _last_density = density

    from torch.nn.attention.flex_attention import BlockMask

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices,
        BLOCK_SIZE=FLEX_BLOCK,
        seq_lengths=(T, T),
        # Q-side metadata is only used by backward. Building it transposes and
        # repacks the complete sparse map, which is wasted during inference.
        compute_q_blocks=False,
    )

    return _get_compiled_flex()(q, k, v, block_mask=block_mask, scale=scale)


def warmup(device=None) -> bool:
    """Pay the one-time torch.compile cost up front. Returns success."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return False
    try:
        from torch.nn.attention.flex_attention import BlockMask
        B, H, T, D = 1, 1, 256, 128
        qkv = torch.randn(B, T, 3 * H * D, device=device, dtype=torch.bfloat16)
        q, k, v = (
            part.view(B, T, H, D).transpose(1, 2)
            for part in qkv.split(H * D, dim=-1)
        )
        blocks = T // FLEX_BLOCK
        counts = torch.full((B, H, blocks), blocks, device=device, dtype=torch.int32)
        indices = torch.arange(blocks, device=device, dtype=torch.int32)
        indices = indices.view(1, 1, 1, blocks).expand(
            B, H, blocks, blocks
        ).contiguous()
        bm = BlockMask.from_kv_blocks(
            counts, indices, BLOCK_SIZE=FLEX_BLOCK, seq_lengths=(T, T),
            compute_q_blocks=False,
        )
        _get_compiled_flex()(q, k, v, block_mask=bm)
        torch.cuda.synchronize(device)
        logger.info("[Sol-Attn] flex_attention compiled (warmup done)")
        return True
    except Exception as exc:
        logger.warning(f"[Sol-Attn] warmup failed: {type(exc).__name__}: {exc}")
        return False


__all__ = ["sol_attn_flex", "warmup", "last_density", "FLEX_BLOCK"]
