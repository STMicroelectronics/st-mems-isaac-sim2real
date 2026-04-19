# ******************************************************************************
# File Name          : detect.py
# Description        : Isaac Sim adapter selection helpers.
# ******************************************************************************
# @attention
#
# Copyright (c) 2026 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.
#
# ******************************************************************************

import os

from .isaac_5_1 import IsaacSim51Adapter
from .isaac_6_0 import IsaacSim60Adapter

ISAAC_VERSION_ENV_VAR = "SIM2REAL_ISAAC_VERSION"


def get_isaac_adapter(preferred_version: str | None = None):
    """
    Return the first available adapter for the active Isaac Sim runtime.

    Release packages can set SIM2REAL_ISAAC_VERSION to force one adapter.
    Without an override, the newest supported runtime is tried first.
    """
    requested_version = (preferred_version or os.environ.get(ISAAC_VERSION_ENV_VAR, "")).strip()
    if requested_version.startswith("5.1"):
        candidates = (IsaacSim51Adapter(),)
    elif requested_version.startswith("6.0"):
        candidates = (IsaacSim60Adapter(),)
    else:
        candidates = (IsaacSim60Adapter(), IsaacSim51Adapter())

    import_errors = []
    for adapter in candidates:
        try:
            if adapter.is_available():
                print(
                    f"[Sim2Real IMU] Using Isaac Sim {adapter.isaac_version} "
                    f"adapter ({adapter.extension_dependency})."
                )
                return adapter
        except ImportError as error:
            import_errors.append(str(error))

    raise ImportError(
        "No supported Isaac Sim IMU adapter is available. "
        f"Set {ISAAC_VERSION_ENV_VAR}=5.1.0 or 6.0.0 if auto-detection is ambiguous. "
        f"Details: {'; '.join(import_errors)}"
    )

