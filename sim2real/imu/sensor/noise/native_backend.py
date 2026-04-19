# ******************************************************************************
# File Name          : native_backend.py
# Description        : Python wrapper around the native C++ noise backend used
#                      for ST IMU sim2real signal generation.
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
import importlib.machinery
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np

NATIVE_MODULE_NAME = "sim2real_native_v0_1"
NATIVE_MODULE_GLOB = f"{NATIVE_MODULE_NAME}*.so"
NATIVE_PATH_ENV_VAR = "SIM2REAL_NATIVE_PATH"
NATIVE_MANIFEST_NAME = "manifest.json"


def _runtime_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _runtime_architecture():
    machine = platform.machine().lower()
    return "x86_64" if machine in ("x86_64", "amd64") else machine


def _iter_candidate_native_paths():
    # User override takes precedence and can include multiple paths.
    env_value = os.environ.get(NATIVE_PATH_ENV_VAR, "").strip()
    if env_value:
        for raw_path in env_value.split(os.pathsep):
            path = Path(raw_path).expanduser()
            if path.is_dir():
                yield path

    module_dir = Path(__file__).resolve().parent
    extension_root = module_dir.parents[3] if len(module_dir.parents) > 3 else module_dir
    for path in (
        module_dir,
        extension_root,
        extension_root / "sim_binary",
        extension_root / "lib",
    ):
        if path.is_dir():
            yield path


def _iter_candidate_native_files():
    """Yield native module files in Python ABI preference order."""
    seen = set()
    for directory in _iter_candidate_native_paths():
        for candidate in _iter_manifest_native_files(directory):
            candidate_key = str(candidate.resolve())
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            yield candidate

        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = directory / f"{NATIVE_MODULE_NAME}{suffix}"
            if not candidate.is_file():
                continue

            candidate_key = str(candidate.resolve())
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            yield candidate


def _iter_manifest_native_files(directory: Path):
    manifest_path = directory / NATIVE_MANIFEST_NAME
    if not manifest_path.is_file():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[Sim2Real IMU] WARNING: Could not read native manifest {manifest_path}: {error}")
        return

    if manifest.get("module") != NATIVE_MODULE_NAME:
        return

    runtime_python = _runtime_python_version()
    runtime_arch = _runtime_architecture()
    for entry in manifest.get("binaries", []):
        if str(entry.get("python", "")).strip() != runtime_python:
            continue
        if str(entry.get("arch", "")).strip().lower() not in ("", runtime_arch):
            continue

        native_file = directory / str(entry.get("file", ""))
        if native_file.is_file():
            yield native_file


def _load_native_file(native_file: Path):
    spec = importlib.util.spec_from_file_location(NATIVE_MODULE_NAME, native_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {native_file}")

    native_module = importlib.util.module_from_spec(spec)
    sys.modules[NATIVE_MODULE_NAME] = native_module
    spec.loader.exec_module(native_module)
    return native_module


def _load_native_module():
    attempt_errors = []
    for native_file in _iter_candidate_native_files():
        try:
            return _load_native_file(native_file), None
        except (ImportError, OSError) as load_error:
            sys.modules.pop(NATIVE_MODULE_NAME, None)
            attempt_errors.append(f"{native_file}: {load_error}")

    try:
        native_module = importlib.import_module(NATIVE_MODULE_NAME)
        return native_module, None
    except ImportError as import_error:
        if attempt_errors:
            error_details = "; ".join(attempt_errors)
            return None, ImportError(
                f"Could not load any candidate native backend. Attempts: {error_details}. "
                f"Fallback import error: {import_error}"
            )
        return None, import_error


sim2real_native, _NATIVE_IMPORT_ERROR = _load_native_module()
_NATIVE_AVAILABLE = sim2real_native is not None
if _NATIVE_AVAILABLE:
    print("[Sim2Real IMU] Native C++ backend loaded successfully.")
else:
    print(f"[Sim2Real IMU] WARNING: Could not load C++ backend: {_NATIVE_IMPORT_ERROR}")
    print(
        f"[Sim2Real IMU] Set {NATIVE_PATH_ENV_VAR} to the directory containing "
        f"{NATIVE_MODULE_GLOB} if needed."
    )
    print("[Sim2Real IMU] IMU prims will be registered but noise will not be applied.")


class NativeNoiseBackend:
    """
    Wraps the C++ sim2real_native pybind module.
    One Sim2RealCore instance is created per IMU prim path,
    so each sensor has independent bias/filter state.
    """

    def __init__(self):
        # Maps prim_path -> Sim2RealCore instance
        self._engines = {}

    def register_sensor(self, sensor_prim_path: str, sensor_config: dict, seed: int = 123):
        """Create a C++ engine instance for a given sensor prim path."""
        if not _NATIVE_AVAILABLE:
            return

        engine = sim2real_native.Sim2RealCore(seed)
        if hasattr(engine, "update_configuration"):
            engine.update_configuration(sensor_config)
        self._engines[sensor_prim_path] = engine
        print(
            f"[Sim2Real IMU] Registered C++ engine for {sensor_prim_path} | "
            f"config={sensor_config}"
        )

    def unregister_sensor(self, sensor_prim_path: str):
        """Remove the engine for a prim (e.g. if it's deleted from stage)."""
        self._engines.pop(sensor_prim_path, None)

    def step_sensor(self, sensor_prim_path: str, sim_time: float, truth_kinematics: dict | None):
        """
        Run one noise step for this prim.
        truth_kinematics: dict with keys 'lin_acc' and 'ang_vel' (arrays, shape [3])
                          If None (truth not yet available), returns None.
        Returns dict with 'lin_acc' and 'ang_vel', or None.
        """
        if not _NATIVE_AVAILABLE:
            return truth_kinematics

        engine = self._engines.get(sensor_prim_path)
        if engine is None:
            return truth_kinematics

        if truth_kinematics is None:
            return None

        lin_acc = truth_kinematics.get("lin_acc", [0.0, 0.0, 0.0])
        ang_vel = truth_kinematics.get("ang_vel", [0.0, 0.0, 0.0])

        return engine.process(
            np.array(lin_acc, dtype=float),
            np.array(ang_vel, dtype=float),
            sim_time,
        )

    def has_sensor(self, sensor_prim_path: str) -> bool:
        return sensor_prim_path in self._engines

    # Backward-compatible API aliases.
    def register(self, prim_path: str, config: dict, seed: int = 123):
        self.register_sensor(prim_path, config, seed=seed)

    def unregister(self, prim_path: str):
        self.unregister_sensor(prim_path)

    def step(self, prim_path: str, sim_time: float, truth: dict | None):
        return self.step_sensor(prim_path, sim_time, truth)

    def is_registered(self, prim_path: str) -> bool:
        return self.has_sensor(prim_path)


# Backward-compatible class name for existing imports.
NativeBackend = NativeNoiseBackend
