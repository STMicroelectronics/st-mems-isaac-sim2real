# ******************************************************************************
# File Name          : isaac_5_1.py
# Description        : Isaac Sim 5.1 IMU sensor adapter.
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


class IsaacSim51Adapter(IsaacImuAdapter):
    isaac_version = "5.1.0"
    extension_dependency = "omni.isaac.sensor"
    imu_module_candidates = ("omni.isaac.sensor",)

