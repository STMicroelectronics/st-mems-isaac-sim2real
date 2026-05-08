# ******************************************************************************
# File Name          : test_runtime_truth_sensor_resolution.py
# Description        : Unit tests for Sim2Real runtime truth IMU resolution.
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

import importlib
from decimal import Decimal
import sys
import types
import unittest


_CURRENT_STAGE = None


class _FakeInvalidPrim:
    def __init__(self, path):
        self._path = path

    def IsValid(self):
        return False

    def GetPath(self):
        return self._path

    def GetCustomData(self):
        return {}

    def SetCustomDataByKey(self, _key, _value):
        pass

    def GetCustomDataByKey(self, _key):
        return None


class _FakePrim:
    def __init__(self, path, custom_data=None):
        self._path = path
        self._custom_data = dict(custom_data or {})

    def IsValid(self):
        return True

    def GetPath(self):
        return self._path

    def GetCustomData(self):
        return dict(self._custom_data)

    def SetCustomDataByKey(self, key, value):
        if isinstance(value, list):
            invalid_items = [item for item in value if type(item) not in (float, int, str, bool)]
            if invalid_items:
                raise ValueError(
                    f"Invalid value type for dictionary key-path '{key}': '{value}'."
                )
        self._custom_data[key] = value

    def GetCustomDataByKey(self, key):
        return self._custom_data.get(key)


class _FakeStage:
    def __init__(self, prim_paths=None):
        self._prims = {path: _FakePrim(path) for path in (prim_paths or [])}

    def add_prim(self, path):
        prim = _FakePrim(path)
        self._prims[path] = prim
        return prim

    def GetPrimAtPath(self, path):
        return self._prims.get(path, _FakeInvalidPrim(path))

    def Traverse(self):
        return list(self._prims.values())


class _FakeSensor:
    def __init__(self, prim_path):
        self.prim_path = prim_path
        self.initialized = False

    def initialize(self):
        self.initialized = True


class _FakeAdapter:
    def __init__(self, stage):
        self._stage = stage
        self.created_paths = []

    def create_imu_sensor(self, prim_path: str, name: str | None = None):
        self.created_paths.append((prim_path, name))
        if not self._stage.GetPrimAtPath(prim_path).IsValid():
            self._stage.add_prim(prim_path)
        return _FakeSensor(prim_path)


class _FakePhysxInterface:
    def subscribe_physics_step_events(self, _callback):
        return object()


class _FakeTimeline:
    def is_playing(self):
        return False

    def get_current_time(self):
        return 0.0


class _FakeUsdContext:
    def get_stage(self):
        return _CURRENT_STAGE


def _install_omni_stubs():
    omni_module = types.ModuleType("omni")
    physx_module = types.ModuleType("omni.physx")
    timeline_module = types.ModuleType("omni.timeline")
    usd_module = types.ModuleType("omni.usd")

    physx_module.get_physx_interface = lambda: _FakePhysxInterface()
    timeline_module.get_timeline_interface = lambda: _FakeTimeline()
    usd_module.get_context = lambda: _FakeUsdContext()

    omni_module.physx = physx_module
    omni_module.timeline = timeline_module
    omni_module.usd = usd_module

    sys.modules["omni"] = omni_module
    sys.modules["omni.physx"] = physx_module
    sys.modules["omni.timeline"] = timeline_module
    sys.modules["omni.usd"] = usd_module


_install_omni_stubs()
runtime_module = importlib.import_module("sim2real.imu.sensor.runtime")


class RuntimeTruthSensorResolutionTests(unittest.TestCase):
    def setUp(self):
        global _CURRENT_STAGE
        _CURRENT_STAGE = _FakeStage(
            [
                "/World/robot",
                "/World/robot/link",
                "/World/robot/link/ASM330LHH",
            ]
        )
        self.stage = _CURRENT_STAGE
        self.adapter = _FakeAdapter(self.stage)
        self.runtime = runtime_module.ImuSensorRuntime(
            noise_backend=types.SimpleNamespace(),
            isaac_adapter=self.adapter,
        )

    def test_resolve_uses_configured_truth_path_when_valid(self):
        self.stage.add_prim("/World/robot/link/custom_truth_imu")
        path, source = self.runtime._resolve_truth_sensor_path(
            self.stage,
            "/World/robot/link",
            configured_truth_sensor_path="/World/robot/link/custom_truth_imu",
        )
        self.assertEqual(path, "/World/robot/link/custom_truth_imu")
        self.assertEqual(source, "configured truthImuPrimPath")

    def test_resolve_discovers_imu_like_descendant(self):
        self.stage.add_prim("/World/robot/link/sensors/body_imu")
        path, source = self.runtime._resolve_truth_sensor_path(
            self.stage,
            "/World/robot/link",
        )
        self.assertEqual(path, "/World/robot/link/sensors/body_imu")
        self.assertEqual(source, "discovered IMU-like descendant")

    def test_initialize_auto_creates_default_truth_sensor_and_persists_metadata(self):
        sensor_prim_path = "/World/robot/link/ASM330LHH"
        truth_sensor_path = "/World/robot/link/Imu_Sensor"
        self.runtime._initialize_truth_sensor(sensor_prim_path, "/World/robot/link")

        self.assertIn(sensor_prim_path, self.runtime._truth_sensor_cache)
        self.assertTrue(self.stage.GetPrimAtPath(truth_sensor_path).IsValid())
        self.assertEqual(
            self.stage.GetPrimAtPath(sensor_prim_path).GetCustomDataByKey(
                self.runtime.TRUTH_SENSOR_PATH_KEY
            ),
            truth_sensor_path,
        )
        self.assertEqual(
            self.adapter.created_paths[-1],
            (truth_sensor_path, self.runtime.TRUTH_SENSOR_PRIM_NAME),
        )

    def test_tick_sensor_coerces_numeric_scalars_before_persisting_custom_data(self):
        sensor_prim_path = "/World/robot/link/ASM330LHH"
        self.runtime._backend = types.SimpleNamespace(
            step_sensor=lambda *_args, **_kwargs: {
                "lin_acc": [
                    Decimal("-8.724470849609375"),
                    Decimal("0.1292868896484375"),
                    Decimal("-4.762067102050781"),
                ],
                "ang_vel": [
                    Decimal("0.1"),
                    Decimal("0.2"),
                    Decimal("0.3"),
                ],
            }
        )
        self.runtime._read_truth_kinematics = lambda _sensor_prim_path: {
            "lin_acc": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0],
        }

        self.runtime._tick_sensor(self.stage, sensor_prim_path, 1.25)

        prim = self.stage.GetPrimAtPath(sensor_prim_path)
        self.assertEqual(
            prim.GetCustomDataByKey(self.runtime.LAST_LIN_ACC_KEY),
            [-8.724470849609375, 0.1292868896484375, -4.762067102050781],
        )
        self.assertEqual(
            prim.GetCustomDataByKey(self.runtime.LAST_ANG_VEL_KEY),
            [0.1, 0.2, 0.3],
        )


if __name__ == "__main__":
    unittest.main()
