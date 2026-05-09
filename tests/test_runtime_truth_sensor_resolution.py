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
import sys
import types
import unittest
from unittest import mock


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

    def get_current_frame(self, read_gravity=True):
        _ = read_gravity
        return {
            "lin_acc": [1.0, 2.0, 3.0],
            "ang_vel": [0.1, 0.2, 0.3],
        }


class _FakeAdapter:
    def __init__(self, stage):
        self._stage = stage
        self.created_paths = []
        self.failures_remaining = 0

    def create_imu_sensor(self, prim_path: str, name: str | None = None):
        self.created_paths.append((prim_path, name))
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError(
                "cannot access local variable 'current_physics_prim' where it is not associated with a value"
            )
        if not self._stage.GetPrimAtPath(prim_path).IsValid():
            self._stage.add_prim(prim_path)
        return _FakeSensor(prim_path)

    def normalize_imu_frame(self, raw):
        return raw


class _FakePhysxInterface:
    def subscribe_physics_step_events(self, _callback):
        return object()


class _FakeTimeline:
    def __init__(self):
        self.playing = False
        self.current_time = 0.0

    def is_playing(self):
        return self.playing

    def get_current_time(self):
        return self.current_time


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
        self.timeline = _FakeTimeline()
        self.runtime = runtime_module.ImuSensorRuntime(
            noise_backend=types.SimpleNamespace(),
            isaac_adapter=self.adapter,
        )
        self.runtime._timeline = self.timeline

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

    def test_runtime_retries_truth_sensor_binding_after_initial_physics_scene_failure(self):
        sensor_prim_path = "/World/robot/link/ASM330LHH"
        step_calls = []
        self.runtime._backend = types.SimpleNamespace(
            register_sensor=lambda *_args, **_kwargs: None,
            step_sensor=lambda *_args, **_kwargs: step_calls.append("step") or {
                "lin_acc": [4.0, 5.0, 6.0],
                "ang_vel": [0.4, 0.5, 0.6],
            },
        )
        self.adapter.failures_remaining = 1
        self.runtime.register_sensor(
            sensor_prim_path,
            {"attachPrimPath": "/World/robot/link", "odr_hz": 100.0},
        )

        self.assertNotIn(sensor_prim_path, self.runtime._truth_sensor_cache)
        self.assertIn(sensor_prim_path, self.runtime._truth_sensor_pending)

        self.timeline.playing = True
        self.timeline.current_time = 1.0
        self.runtime._on_physics_step(0.01)

        self.assertIn(sensor_prim_path, self.runtime._truth_sensor_cache)
        self.assertNotIn(sensor_prim_path, self.runtime._truth_sensor_pending)
        self.assertEqual(step_calls, ["step"])
        self.assertEqual(self.runtime._last_tick_sim_time_s[sensor_prim_path], 1.0)

    def test_runtime_skips_sensor_ticks_until_truth_imu_is_available(self):
        sensor_prim_path = "/World/robot/link/ASM330LHH"
        step_calls = []
        self.runtime._backend = types.SimpleNamespace(
            register_sensor=lambda *_args, **_kwargs: None,
            step_sensor=lambda *_args, **_kwargs: step_calls.append("step") or {
                "lin_acc": [4.0, 5.0, 6.0],
                "ang_vel": [0.4, 0.5, 0.6],
            },
        )
        self.adapter.failures_remaining = 99
        self.runtime.register_sensor(
            sensor_prim_path,
            {"attachPrimPath": "/World/robot/link", "odr_hz": 100.0},
        )

        self.timeline.playing = True
        self.timeline.current_time = 2.0
        self.runtime._on_physics_step(0.02)

        self.assertNotIn(sensor_prim_path, self.runtime._truth_sensor_cache)
        self.assertEqual(step_calls, [])
        self.assertNotIn(sensor_prim_path, self.runtime._last_tick_sim_time_s)
        prim = self.stage.GetPrimAtPath(sensor_prim_path)
        self.assertIsNone(prim.GetCustomDataByKey(self.runtime.LAST_LIN_ACC_KEY))
        self.assertIsNone(prim.GetCustomDataByKey(self.runtime.LAST_ANG_VEL_KEY))

    def test_runtime_warns_once_when_physics_rate_is_below_repo_baseline(self):
        self.runtime._sensor_registry["/World/robot/link/ASM330LHH"] = {"odr_hz": 104.0}
        with mock.patch("builtins.print") as print_mock:
            self.runtime._maybe_warn_on_physics_rate(1.0 / 60.0)
            self.runtime._maybe_warn_on_physics_rate(1.0 / 60.0)

        warning_calls = [
            call for call in print_mock.call_args_list if "Current physics step is approximately" in str(call)
        ]
        self.assertEqual(len(warning_calls), 1)

    def test_runtime_does_not_warn_when_physics_rate_meets_repo_baseline(self):
        self.runtime._sensor_registry["/World/robot/link/ASM330LHH"] = {"odr_hz": 104.0}
        with mock.patch("builtins.print") as print_mock:
            self.runtime._maybe_warn_on_physics_rate(1.0 / 208.0)

        warning_calls = [
            call for call in print_mock.call_args_list if "Current physics step is approximately" in str(call)
        ]
        self.assertEqual(warning_calls, [])


if __name__ == "__main__":
    unittest.main()
