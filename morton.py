# SPDX-License-Identifier: Apache-2.0
# See NOTICE for attribution.

"""Morton (Z-order) permutations for video token sequences.

Video latents are packed frame-major, then row-major within a frame.  A
contiguous 128-token attention block is therefore a thin horizontal strip of a
single frame -- for a 60x104 latent grid at 2x2 patching, 128 tokens is about
2.5 rows of one frame.  Attention in a video DiT is locally 3D, so that strip
borders many blocks that each carry only a little probability mass.

Z-ordering makes each 128-token block a roughly 5x5x5 neighbourhood instead.
The same attention mass then lands in far fewer blocks, so block routing keeps
more of it at any given density.  This does not make attention cheaper by
itself; it makes the *routing decision* sharper, which is what lets a higher
``tau`` hold quality.

The permutation is applied once before the transformer stack and inverted after
it, with the rope table permuted to match, so it is exactly a no-op for dense
attention.
"""

from __future__ import annotations

import torch

_PERM_CACHE = {}


def _part1by2(v: torch.Tensor) -> torch.Tensor:
    """Spread the low 21 bits of ``v`` so each occupies every third bit."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton_perm(grid, device, curve: str = "2d_frame"):
    """Return ``(perm, inverse)`` for a ``(t, h, w)`` token grid.

    ``curve="3d"`` interleaves t/h/w equally.

    ``curve="2d_frame"`` Z-orders within each frame and leaves frame order
    alone.  Use it when the temporal axis is not uniformly spaced: MiniMax H3's
    ``FRAME_PER_TOKEN`` is ``(1, 4, 4, 4, 4)``, so index-adjacent latent frames
    are either 1 or 4 real frames apart and a 3D curve would group temporally
    distant tokens into the same block.
    """
    key = (tuple(int(g) for g in grid), curve)
    hit = _PERM_CACHE.get(key)
    if hit is None:
        frames, height, width = key[0]
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        t = linear // area
        rem = linear - t * area
        h = rem // width
        w = rem - h * width

        if curve == "2d_frame":
            # Frame index stays the most significant key, so frames never mix.
            code = (t << 42) | _part1by2(w) | (_part1by2(h) << 1)
        else:
            code = _part1by2(w) | (_part1by2(h) << 1) | (_part1by2(t) << 2)

        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        _PERM_CACHE[key] = hit
    perm, inverse = hit
    return perm.to(device), inverse.to(device)


def aligned_morton_perm(grid, device, curve: str, start: int, block: int):
    """Morton permutation for a span beginning at absolute row ``start``.

    Attention blocks are cut from absolute position 0, so a video span that does
    not begin on a block boundary would split every Z-order cell across two
    blocks -- joining opposite ends of the volume and defeating the point.
    Rotating the permutation by the misalignment realigns the cells.  The ragged
    group this displaces lands in the block shared with the conditioning prefix,
    which prefix protection already keeps exact.
    """
    pad = (-int(start)) % int(block)
    key = (tuple(int(g) for g in grid), curve, str(device), pad)
    hit = _PERM_CACHE.get(key)
    if hit is None:
        perm, inverse = morton_perm(grid, device, curve)
        if pad:
            perm = torch.roll(perm, pad)
            inverse = torch.argsort(perm)
        hit = (perm, inverse)
        _PERM_CACHE[key] = hit
    return hit


__all__ = ["morton_perm", "aligned_morton_perm"]
