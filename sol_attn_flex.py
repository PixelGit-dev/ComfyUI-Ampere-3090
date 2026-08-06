# SPDX-License-Identifier: Apache-2.0
#
# The routing algorithm is Sol-Attn by NVlabs (arXiv:2607.24027,
# https://github.com/NVlabs/Sana tree/sol-engine), Apache-2.0.  See NOTICE.

"""Sol-Attn block-sparse attention via torch flex_attention.

Works on any GPU flex_attention supports (sm_80 Ampere and newer) -- there is
nothing architecture-specific here.  Routing is done in pure torch at 128-token
granularity, then executed by ``torch.compile``d ``flex_attention``.

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
* **Approximate correction** (optional, on by default) recovers the long-tail
  contribution of skipped blocks instead of dropping them outright -- see
  ``_correct_dropped``.
* **Cornish-Fisher thresholds** (optional) replace the Gaussian quantile in the
  routing threshold with one corrected for the score distribution's skew and
  excess kurtosis.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

# flex_attention's compiled kernel requires 128-token blocks.
FLEX_BLOCK = 128

# Cap on heads processed per routing chunk; keeps the [b, hc, N, N] pilot
# score and packing workspace small at long sequences.
_MAX_ROUTE_ELEMS = 8 * 656 * 656

# Cap on elements per correction chunk.  The correction works on [b, hc, S, N]
# rather than [b, hc, N, N], so it needs a much tighter bound: 8M fp32 elements
# is a 32 MB working set per intermediate.
_MAX_CORRECT_ELEMS = 8 * 1024 * 1024

_compiled_flex = None
_last_density = None
_geometry_cache = OrderedDict()
_MAX_GEOMETRIES = 4

# None = not yet probed, True/False = flex_attention's logsumexp is usable.
_lse_supported = None

try:  # return_lse is deprecated in favour of return_aux from torch 2.10
    from torch.nn.attention.flex_attention import AuxRequest as _AuxRequest

    _AUX_LSE = _AuxRequest(lse=True)
except Exception:
    _AUX_LSE = None


def _get_compiled_flex():
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex = torch.compile(flex_attention)
    return _compiled_flex


def _flex(q, k, v, block_mask, scale, want_lse):
    """Run flex_attention, optionally also returning the per-row logsumexp."""
    flex = _get_compiled_flex()
    if not want_lse:
        return flex(q, k, v, block_mask=block_mask, scale=scale), None
    if _AUX_LSE is not None:
        out, aux = flex(q, k, v, block_mask=block_mask, scale=scale,
                        return_aux=_AUX_LSE)
        return out, aux.lse
    return flex(q, k, v, block_mask=block_mask, scale=scale, return_lse=True)


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
    rows = torch.arange(blocks * FLEX_BLOCK, device=device) // FLEX_BLOCK
    _geometry_cache[key] = (ids, local, rows)
    if len(_geometry_cache) > _MAX_GEOMETRIES:
        _geometry_cache.popitem(last=False)
    return ids, local, rows


def _pool_blocks(x, reduce="mean"):
    """Reduce a possibly strided BHSD tensor per 128-token block, no padding.

    ``reduce="mean"`` gives block centroids (the routing pilot uses these);
    ``reduce="sum"`` gives the block value sums the correction term needs.  The
    partial tail block is reduced over its real token count, never a padded 128.
    """
    B, H, T, D = x.shape
    full = T // FLEX_BLOCK
    pieces = []
    if full:
        body = x[:, :, :full * FLEX_BLOCK, :].view(B, H, full, FLEX_BLOCK, D)
        pieces.append(
            body.mean(dim=3, dtype=torch.float32) if reduce == "mean"
            else body.sum(dim=3, dtype=torch.float32)
        )
    if full * FLEX_BLOCK != T:
        tail = x[:, :, full * FLEX_BLOCK:, :]
        pieces.append(
            tail.mean(dim=2, dtype=torch.float32, keepdim=True) if reduce == "mean"
            else tail.sum(dim=2, dtype=torch.float32, keepdim=True)
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


def _threshold(qb, kc, tau, cornish_fisher):
    """Per-query-block routing threshold over the pilot score distribution.

    Sol-Attn keeps a KV block when its pilot score exceeds ``mean + tau * std``
    of the score distribution, computed in closed form from the moments of the
    block centroids under a diagonal-independence assumption.

    ``cornish_fisher`` replaces the Gaussian quantile ``tau`` with a
    Cornish-Fisher expansion of the same quantile that also uses the third and
    fourth cumulants.  Pilot scores are a weighted sum over D dimensions and are
    noticeably skewed in practice, so the Gaussian quantile systematically
    misplaces the threshold; the expansion costs two extra moment reductions
    over the (tiny) block centroids.
    """
    kc_mean = kc.mean(dim=2, keepdim=True)                       # [B,hc,1,D]
    centred = kc - kc_mean
    kc_var = centred.pow(2).mean(dim=2)                          # [B,hc,D]

    mean_term = (qb @ kc_mean.transpose(-1, -2)).squeeze(-1)     # [B,hc,N]
    qb2 = qb * qb
    var_term = (qb2 @ kc_var.unsqueeze(-1)).squeeze(-1)          # [B,hc,N]
    std = torch.sqrt(var_term.clamp_min(0.0) + 1e-6)

    if not cornish_fisher:
        return mean_term + tau * std

    kc_k3 = centred.pow(3).mean(dim=2)
    kc_k4 = centred.pow(4).mean(dim=2) - 3.0 * kc_var.pow(2)
    # Cumulants add under the same diagonal-independence assumption used above.
    raw_sd = torch.sqrt(var_term.clamp_min(0.0) + 1e-12)
    g1 = ((qb2 * qb) @ kc_k3.unsqueeze(-1)).squeeze(-1) / raw_sd.pow(3)
    g2 = ((qb2 * qb2) @ kc_k4.unsqueeze(-1)).squeeze(-1) / raw_sd.pow(4)
    g1 = g1.clamp(-2.0, 2.0)
    g2 = g2.clamp(-5.0, 5.0)

    z = float(tau)
    offset = (z
              + (z * z - 1.0) * g1 / 6.0
              + (z * z * z - 3.0 * z) * g2 / 24.0
              - (2.0 * z * z * z - 5.0 * z) * g1 * g1 / 36.0)
    # An asymptotic expansion: keep it within one sigma of the Gaussian answer
    # so a badly conditioned head cannot swing the threshold arbitrarily.
    offset = offset.clamp(z - 1.0, z + 1.0)
    return mean_term + offset * std


def _build_routing(q_h, kc_all, tau, preserve_prefix_blocks,
                   dense_prefix_queries, cornish_fisher, want_sel):
    """Sol-Attn diag routing.

    Returns ``(kv_num_blocks, kv_indices, density, sel)``.  ``sel`` is the
    ``[B, H, N, N]`` boolean keep-map, returned only when the correction term
    needs it.
    """
    B, H, T, D = q_h.shape
    N = kc_all.shape[2]
    dev = q_h.device

    qb_all = _pool_blocks(q_h)
    blk, diag, _rows = _routing_geometry(dev, N)

    kv_num_blocks = torch.empty(B, H, N, dtype=torch.int32, device=dev)
    kv_indices = torch.empty(B, H, N, N, dtype=torch.int32, device=dev)
    sel_all = torch.empty(B, H, N, N, dtype=torch.bool, device=dev) if want_sel else None

    chunk = max(1, min(H, _MAX_ROUTE_ELEMS // max(1, N * N)))
    for h0 in range(0, H, chunk):
        h1 = min(H, h0 + chunk)
        kc = kc_all[:, h0:h1]                     # [B, hc, N, D]
        qb = qb_all[:, h0:h1]

        thresh = _threshold(qb, kc, tau, cornish_fisher)

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
        if sel_all is not None:
            sel_all[:, h0:h1] = sel

        del score, sel, counts, packed

    # Leave density on-device. Calling .item() here serialized every attention
    # layer with the CPU even when density logging was infrequent.
    density = kv_num_blocks.sum(dtype=torch.float32) / float(B * H * N * N)
    return kv_num_blocks, kv_indices, density, sel_all


def _correct_dropped(out, lse, q, kc_all, vc_aug, sel, scale):
    """Fold the skipped blocks back in as a mean-field correction, in place.

    Sol-Attn's routing drops every KV block whose pilot score falls below the
    threshold.  Individually those blocks are negligible; summed over the ~80%
    of blocks that get dropped, their contribution is not, and discarding it
    biases the softmax denominator low and the output toward the kept blocks.

    The paper recovers it from the proxy scores that routing already computed.
    For a dropped block ``b`` and query row ``i`` this approximates every key in
    the block by the block centroid, so with ``s_ib = scale * q_i . kc_b``::

        sum_{j in b} exp(s_ij) v_j  ~  exp(s_ib) * (sum_{j in b} v_j)
        sum_{j in b} exp(s_ij)      ~  exp(s_ib) * len(b)

    which is one matmul against the precomputed block value sums.  Merging that
    with the exact partial softmax is the standard flash-attention rescale, so
    flex_attention's logsumexp is all the extra state required.

    Unlike routing, this works on ``[B, H, S, N]`` (one score per *row*, not per
    query block), so it is chunked over heads on a much tighter budget.
    """
    B, H, S, D = q.shape
    N = kc_all.shape[2]
    dev = q.device

    _blk, _diag, rows = _routing_geometry(dev, N)
    rows = rows[:S]

    chunk = max(1, min(H, _MAX_CORRECT_ELEMS // max(1, S * N)))
    for h0 in range(0, H, chunk):
        h1 = min(H, h0 + chunk)

        # Pilot score per query row against every block centroid.  fp32: this
        # feeds an exp, and bf16 rounding here is worth several percent on the
        # recovered weight.
        s = torch.matmul(q[:, h0:h1].float(),
                         kc_all[:, h0:h1].transpose(-1, -2)).mul_(scale)

        # Exclude the blocks flex_attention already did exactly.
        s.masked_fill_(sel[:, h0:h1].index_select(2, rows), -math.inf)

        # The diagonal window guarantees every query block keeps at least one KV
        # block, so lse is always finite and merged_max cannot be -inf even when
        # a row has nothing left to approximate.
        lse_c = lse[:, h0:h1].float()
        merged_max = torch.maximum(lse_c, s.amax(dim=-1))            # [B,hc,S]
        s.sub_(merged_max.unsqueeze(-1)).exp_()                      # weights

        # vc_aug is [.., N, D+1]: block value sums with the block length
        # appended, so the numerator and the denominator are one matmul.
        acc = torch.matmul(s, vc_aug[:, h0:h1])                      # [B,hc,S,D+1]
        exact_w = torch.exp(lse_c - merged_max).unsqueeze(-1)        # [B,hc,S,1]

        chunk_out = out[:, h0:h1]
        numerator = chunk_out.float() * exact_w + acc[..., :D]
        denominator = exact_w + acc[..., D:]
        chunk_out.copy_(numerator / denominator)

        del s, acc, numerator, denominator

    return out


def _probe_lse(device) -> bool:
    """Check that flex_attention's logsumexp means what the correction assumes.

    torch normalizes the kernel's base-2 statistics to natural log on the way
    out, but that conversion is a moving part across versions and backends, and
    a silent base mismatch would corrupt every corrected output by a factor of
    ln 2 in the exponent.  One tiny dense call settles it.
    """
    from torch.nn.attention.flex_attention import BlockMask

    # head_dim is marked static inside flex_attention, so the lse variant
    # compiles per head_dim. Probe at H3's 128 to warm the graph the run needs.
    B, H, T, D = 1, 1, 256, 128
    blocks = T // FLEX_BLOCK
    scale = D ** -0.5
    q, k, v = (torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
               for _ in range(3))
    counts = torch.full((B, H, blocks), blocks, device=device, dtype=torch.int32)
    indices = torch.arange(blocks, device=device, dtype=torch.int32)
    indices = indices.view(1, 1, 1, blocks).expand(B, H, blocks, blocks).contiguous()
    bm = BlockMask.from_kv_blocks(counts, indices, BLOCK_SIZE=FLEX_BLOCK,
                                  seq_lengths=(T, T), compute_q_blocks=False)

    _out, lse = _flex(q, k, v, bm, scale, want_lse=True)
    reference = torch.logsumexp(
        (q.float() @ k.float().transpose(-1, -2)) * scale, dim=-1
    )
    error = (lse.float() - reference).abs().max().item()
    if error > 5e-2:
        logger.warning(
            f"[Sol-Attn] flex_attention logsumexp disagrees with the reference "
            f"by {error:.3f}; approximate correction disabled"
        )
        return False
    return True


def _correction_available(device) -> bool:
    global _lse_supported
    if _lse_supported is None:
        try:
            _lse_supported = _probe_lse(device)
        except Exception as exc:
            logger.warning(
                f"[Sol-Attn] logsumexp unavailable ({type(exc).__name__}: {exc}); "
                f"approximate correction disabled"
            )
            _lse_supported = False
    return _lse_supported


def sol_attn_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.2,
    preserve_prefix_blocks: int = 0,
    dense_prefix_queries: bool = False,
    approx_correction: bool = True,
    cornish_fisher: bool = False,
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

    correct = approx_correction and _correction_available(q.device)

    kc_all = _pool_blocks(k)
    kv_num_blocks, kv_indices, density, sel = _build_routing(
        q, kc_all, tau, preserve_prefix_blocks, dense_prefix_queries,
        cornish_fisher, want_sel=correct,
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

    out, lse = _flex(q, k, v, block_mask, scale, want_lse=correct)
    if not correct:
        return out

    N = kc_all.shape[2]
    lengths = torch.full((B, H, N, 1), float(FLEX_BLOCK),
                         device=q.device, dtype=torch.float32)
    tail = T - (N - 1) * FLEX_BLOCK
    if tail != FLEX_BLOCK:
        lengths[:, :, -1] = float(tail)
    vc_aug = torch.cat([_pool_blocks(v, reduce="sum"), lengths], dim=-1)

    return _correct_dropped(out, lse, q, kc_all, vc_aug, sel, scale)


def warmup(device=None, approx_correction: bool = True) -> bool:
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
        _flex(q, k, v, bm, D ** -0.5, want_lse=False)
        # The lse variant compiles to its own graph; probing here keeps that
        # cost off the first sampled step too.
        if approx_correction:
            _correction_available(device)
        torch.cuda.synchronize(device)
        logger.info(
            f"[Sol-Attn] flex_attention compiled (warmup done, "
            f"correction={'ready' if _lse_supported else 'off'})"
        )
        return True
    except Exception as exc:
        logger.warning(f"[Sol-Attn] warmup failed: {type(exc).__name__}: {exc}")
        return False


__all__ = ["sol_attn_flex", "warmup", "last_density", "FLEX_BLOCK"]
