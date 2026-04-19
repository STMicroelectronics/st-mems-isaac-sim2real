# ******************************************************************************
# File Name          : base.py
# Description        : Shared Isaac Sim adapter contract and IMU frame
#                      normalization helpers.
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

from importlib import import_module

import numpy as np


class IsaacImuAdapter:
    """
    Adapter boundary for Isaac Sim version-specific sensor APIs.

    The rest of the Sim2Real runtime consumes only the normalized internal
    frame contract: {"lin_acc": np.ndarray, "ang_vel": np.ndarray}.
    """

    isaac_version = "unknown"
    extension_dependency = ""
    imu_module_candidates: tuple[str, ...] = ()

    def __init__(self):
        self._imu_module = None
        self._imu_sensor_cls = None

    def is_available(self) -> bool:
        try:
            self._resolve_imu_sensor_class()
            return True
        except ImportError:
            return False

    def create_imu_sensor(self, prim_path: str, name: str | None = None):
        imu_sensor_cls = self._resolve_imu_sensor_class()
        kwargs = {"prim_path": prim_path}
        if name:
            kwargs["name"] = name
        return imu_sensor_cls(**kwargs)

    def read_imu_frame(self, sensor, read_gravity: bool = True) -> dict | None:
        raw_frame = sensor.get_current_frame(read_gravity=read_gravity)
        if raw_frame is None:
            return None
        return self.normalize_imu_frame(raw_frame)

    def normalize_imu_frame(self, raw_frame) -> dict:
        return {
            "lin_acc": self._extract_vector(
                raw_frame,
                dict_keys=("lin_acc", "linear_acceleration"),
                attr_keys=(
                    ("lin_acc_x", "lin_acc_y", "lin_acc_z"),
                    ("linearAccelerationX", "linearAccelerationY", "linearAccelerationZ"),
                    ("linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"),
                ),
                label="linear acceleration",
            ),
            "ang_vel": self._extract_vector(
                raw_frame,
                dict_keys=("ang_vel", "angular_velocity"),
                attr_keys=(
                    ("ang_vel_x", "ang_vel_y", "ang_vel_z"),
                    ("angularVelocityX", "angularVelocityY", "angularVelocityZ"),
                    ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
                ),
                label="angular velocity",
            ),
        }

    def _resolve_imu_sensor_class(self):
        if self._imu_sensor_cls is not None:
            return self._imu_sensor_cls

        import_errors = []
        for module_name in self.imu_module_candidates:
            try:
                module = import_module(module_name)
                self._imu_module = module
                self._imu_sensor_cls = module.IMUSensor
                return self._imu_sensor_cls
            except (ImportError, AttributeError) as error:
                import_errors.append(f"{module_name}: {error}")

        raise ImportError(
            f"Could not import IMUSensor for Isaac Sim {self.isaac_version}. "
            f"Tried: {'; '.join(import_errors)}"
        )

    def _extract_vector(self, raw_frame, dict_keys, attr_keys, label: str):
        if isinstance(raw_frame, dict):
            for key in dict_keys:
                value = raw_frame.get(key)
                if value is not None:
                    return np.array(value, dtype=float)
            available = ", ".join(sorted(str(key) for key in raw_frame.keys()))
            raise KeyError(
                f"Missing {label} in Isaac Sim {self.isaac_version} IMU frame. "
                f"Expected one of {dict_keys}; available keys: {available}"
            )

        for keys in attr_keys:
            if all(hasattr(raw_frame, key) for key in keys):
                return np.array([getattr(raw_frame, key) for key in keys], dtype=float)

        raise KeyError(
            f"Missing {label} in Isaac Sim {self.isaac_version} IMU frame object. "
            f"Expected one of {attr_keys}; type={type(raw_frame).__name__}"
        )

