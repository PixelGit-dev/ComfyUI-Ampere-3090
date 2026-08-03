# SPDX-License-Identifier: Apache-2.0
# See NOTICE for attribution.

"""Sol-Attn block-sparse attention for MiniMax H3 on Ampere and newer.

No compute-capability gate: the backend is pure torch + flex_attention, which
has supported sm_80 since PyTorch 2.5.  Verified on RTX 3090 (sm_86).
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

try:
    from patch import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] < 8:
                logger.warning(
                    f"[Sol-Attn] {torch.cuda.get_device_name(0)} is sm_"
                    f"{cap[0]}{cap[1]}; flex_attention needs sm_80+. The node "
                    f"will load but always fall back."
                )
            else:
                from sol_attn_flex import warmup
                warmup()
    except Exception as exc:
        logger.warning(f"[Sol-Attn] warmup skipped: {type(exc).__name__}: {exc}")

except Exception as exc:
    logger.error(f"[Sol-Attn] failed to load: {exc}", exc_info=True)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
