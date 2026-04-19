# ******************************************************************************
# File Name          : isaac_6_0.py
# Description        : Isaac Sim 6.0 IMU sensor adapter.
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

from .base import IsaacImuAdapter


class IsaacSim60Adapter(IsaacImuAdapter):
    isaac_version = "6.0.0"
    extension_dependency = "isaacsim.sensors.physics"
    imu_module_candidates = (
        "isaacsim.sensors.physics",
        "isaacsim.sensors.experimental.physics",
    )

