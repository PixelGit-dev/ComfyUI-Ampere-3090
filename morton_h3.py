# SPDX-License-Identifier: Apache-2.0
# See NOTICE for attribution.

"""Morton (Z-order) reordering of MiniMax H3's video span.

H3 packs one self-attention sequence, in this order (``PackedLayout`` in
``comfy/ldm/minimax/model.py``)::

    [text][cond][ref_audio][ref_img][audio][video]

Only the target video segment is reordered.  Text, audio and reference rows keep
their positions, so the conditioning streams are untouched and only the video
tokens gain 3D-compact attention blocks.  The video segment is contiguous,
always last, and laid out t-major then h then w (``_video_grid`` /
``unpatchify_video``), so a ``(t, h//2, w//2)`` Z-order maps straight onto it.

Why this is safe:

* Attention is permutation-equivariant, and rope is applied per row from a
  table.  Permuting tokens and their rope rows together permutes the output.
* Everything else in a block (norms, MLP, the adaln gates) is row-wise.  The
  adaln modulation is applied per layout segment, and the video segment is a
  single uniform ``mod_segments`` entry, so reordering *within* it changes
  nothing.
* The permutation is undone after the last block, before ``final_layer`` slices
  the video rows back out.

So for dense attention this is exactly a no-op.  It only changes what the
*block router* sees.

Three coordinated pieces, all gated on ``transformer_options["sol_attn_morton"]``:

* ``PackedLayout.__init__``  -- registers each layout's video span
* ``_forward``               -- reads the gate for this sampling call
* ``rope_freqs``             -- resolves the span, publishes it for prefix protection
* block pre/post hooks       -- permute hidden states + rope in, restore out

The permute-or-not decision is made in exactly one place: the block-0 pre-hook,
which is the only point that sees the hidden states and the rope table together.
Deciding separately for the two (for instance in ``rope_freqs`` for the table and
in the hook for the tokens) lets the two guards disagree when a reference image
or extra context changes the token count -- which permutes positions without
permuting tokens and corrupts the output outright rather than degrading it.
Never split this decision.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import torch

from morton import aligned_morton_perm

logger = logging.getLogger(__name__)

# id(position_ids) -> (layout, bounds, span).  The layout is deliberately kept
# alive so CPython cannot recycle the id underneath us -- which means this map
# owns memory, so it is bounded: one entry per distinct sequence shape, oldest
# evicted.  A session that sweeps many resolutions would otherwise accumulate a
# position_ids tensor per shape forever.
_SPANS = OrderedDict()
_MAX_SPANS = 8

_PATCHED_LAYOUTS = set()
_INDEX_CACHE = {}
_logged = set()

_INSTALL_FLAG = "_sol_attn_hooks_installed"

VIDEO_SPAN_KEY = "sol_attn_video_span"
MORTON_KEY = "sol_attn_morton"
CURVE_KEY = "sol_attn_morton_curve"


def _log_once(message):
    if message not in _logged:
        _logged.add(message)
        logger.info(f"[Sol-Attn] H3 Morton: {message}")


def _video_span(layout):
    """``(bounds, span)`` for the target video segment.

    ``bounds`` is ``(start, stop)`` and is used for prefix protection even when
    reordering is off.  ``span`` adds the ``(t, h, w)`` grid and is None when the
    grid does not account for the segment exactly -- the grid is stored rather
    than a permutation so the curve can change between runs.
    """
    segments = getattr(layout, "segments", None) or ()
    bounds = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    if bounds is None:
        return None, None

    signature = getattr(layout, "signature", None)
    if not signature or len(signature) < 4:
        return bounds, None
    _text_len, latent_t, latent_h, latent_w = signature[:4]

    start, stop = bounds
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    if grid[0] * grid[1] * grid[2] != stop - start:
        _log_once(f"video segment of {stop - start} rows does not match grid {grid}; "
                  f"reordering inactive")
        return bounds, None
    return bounds, (start, stop, grid)


def _patch_packed_layout(module):
    """Register the video span of every PackedLayout built, without mutating it.

    The span is registered from ``__init__`` because ComfyUI builds layouts in
    two places -- ``model_base`` prebuilds one per sampling run, and the model
    rebuilds it inline on a signature mismatch -- and both run ``__init__``.
    """
    layout_cls = getattr(module, "PackedLayout", None)
    if layout_cls is None:
        raise RuntimeError(f"{module.__name__} has no PackedLayout")
    if id(layout_cls) in _PATCHED_LAYOUTS:
        return
    original_init = layout_cls.__init__

    def __init__(self, *args, **kwargs):
        # Read the finished object rather than mirroring the constructor
        # signature, which has changed across model revisions.
        original_init(self, *args, **kwargs)
        try:
            bounds, span = _video_span(self)
            if bounds is not None and torch.is_tensor(getattr(self, "position_ids", None)):
                _SPANS[id(self.position_ids)] = (self, bounds, span)
                _SPANS.move_to_end(id(self.position_ids))
                while len(_SPANS) > _MAX_SPANS:
                    _SPANS.popitem(last=False)
        except Exception as exc:  # never break model construction
            logger.info(f"[Sol-Attn] H3 span registration failed: "
                        f"{type(exc).__name__}: {exc}")

    layout_cls.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_cls))


def _index_vectors(span, curve, device, block):
    """Full-length gather indices for the forward and inverse permutation."""
    start, stop, grid = span
    key = (start, stop, tuple(grid), curve, str(device), int(block))
    hit = _INDEX_CACHE.get(key)
    if hit is None:
        perm, inverse = aligned_morton_perm(grid, device, curve, start, block)
        forward = torch.arange(stop, device=device, dtype=torch.int64)
        forward[start:stop] = perm + start
        backward = torch.arange(stop, device=device, dtype=torch.int64)
        backward[start:stop] = inverse + start
        hit = (forward, backward)
        _INDEX_CACHE[key] = hit
    return hit


def _gather(x, dim, index):
    """index_select along ``dim``, leaving any rows past ``index`` in place.

    The index vector covers rows 0..stop; anything after the video segment (there
    is nothing, video is last, but padding would land here) is passed through.
    """
    if x.shape[dim] == index.numel():
        return x.index_select(dim, index)
    head = x.index_select(dim, index)
    tail = x.narrow(dim, index.numel(), x.shape[dim] - index.numel())
    return torch.cat([head, tail], dim=dim)


def install_h3_hooks(model, block_size: int = 128):
    """Idempotently install span publication and the reorder hooks.

    Inert unless ``transformer_options[MORTON_KEY]`` is set; the span is
    published regardless, because prefix protection wants it either way.
    """
    # Marked on the model rather than in an id() set: a freed model can be
    # reallocated at the same address, and an id-keyed guard would then skip
    # installation on a model that has no hooks.
    if getattr(model, _INSTALL_FLAG, False):
        return
    for attr in ("rope_freqs", "_forward", "blocks"):
        if not hasattr(model, attr):
            raise RuntimeError(f"H3 hooks need .{attr} on the diffusion model")

    _patch_packed_layout(sys.modules[type(model).__module__])

    original_forward = model._forward
    original_rope_freqs = model.rope_freqs
    first, last = model.blocks[0], model.blocks[-1]

    def _forward(x, timestep, context, transformer_options={}, **kwargs):
        previous = getattr(model, "_sol_active", False)
        model._sol_active = bool((transformer_options or {}).get(MORTON_KEY))
        model._sol_curve = (transformer_options or {}).get(CURVE_KEY, "2d_frame")
        model._sol_options = transformer_options
        try:
            return original_forward(x, timestep, context,
                                    transformer_options=transformer_options, **kwargs)
        finally:
            model._sol_active = previous
            model._sol_span = None
            model._sol_state = None
            model._sol_options = None
            if isinstance(transformer_options, dict):
                transformer_options.pop(VIDEO_SPAN_KEY, None)

    def rope_freqs(position_ids, device):
        """Resolve the layout and publish its span. Nothing is permuted here."""
        model._sol_span = None
        model._sol_state = None
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            _log_once("no layout registered for this position_ids tensor; "
                      "reordering and automatic prefix protection are inactive")
            return original_rope_freqs(position_ids, device)

        _layout, bounds, span = entry
        options = getattr(model, "_sol_options", None)
        if isinstance(options, dict):
            options[VIDEO_SPAN_KEY] = bounds
        if getattr(model, "_sol_active", False) and span is not None:
            model._sol_span = span
        return original_rope_freqs(position_ids, device)

    def pre_hook(module, args):
        # Blocks are called block(h, t_emb, mod_segments, rope_freqs, ...).
        if len(args) < 4:
            return None

        if module is not first:
            state = getattr(model, "_sol_state", None)
            if state is None:
                return None
            # Hidden states are already reordered; reuse the table built once.
            return args[:3] + (state[1],) + tuple(args[4:])

        model._sol_state = None
        span = getattr(model, "_sol_span", None)
        if span is None:
            return None
        start, stop, _grid = span
        h, rope = args[0], args[3]

        # One decision, both tensors in hand -- see the module docstring.
        if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] < stop:
            _log_once(f"hidden states {tuple(h.shape) if torch.is_tensor(h) else type(h)} "
                      f"do not cover the video span {(start, stop)}; inactive")
            return None
        if not torch.is_tensor(rope) or rope.ndim < 2 or rope.shape[1] != h.shape[0]:
            _log_once(f"rope table {tuple(rope.shape) if torch.is_tensor(rope) else type(rope)} "
                      f"does not match {h.shape[0]} tokens; inactive")
            return None

        curve = getattr(model, "_sol_curve", "2d_frame")
        forward, backward = _index_vectors(span, curve, h.device, block_size)
        h = _gather(h, 0, forward)
        # The rope rotation table is elementwise per row, so permuting the table
        # is identical to permuting position_ids -- and keeps the decision here.
        rope = _gather(rope, 1, forward)

        model._sol_state = (backward, rope)
        return (h,) + args[1:3] + (rope,) + tuple(args[4:])

    def post_hook(_module, _args, output):
        state = getattr(model, "_sol_state", None)
        model._sol_state = None
        if state is None:
            return None
        backward, _rope = state
        if not torch.is_tensor(output) or output.shape[0] < backward.numel():
            logger.error("[Sol-Attn] H3 Morton: the block stack changed the token count "
                         f"({tuple(output.shape) if torch.is_tensor(output) else type(output)}); "
                         f"cannot restore order")
            return None
        return _gather(output, 0, backward)

    model._forward = _forward
    model.rope_freqs = rope_freqs
    for block in model.blocks:
        block.register_forward_pre_hook(pre_hook)
    last.register_forward_hook(post_hook)
    setattr(model, _INSTALL_FLAG, True)
    logger.info("[Sol-Attn] H3 layout hooks installed")


__all__ = ["install_h3_hooks", "VIDEO_SPAN_KEY", "MORTON_KEY", "CURVE_KEY"]
