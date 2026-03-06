import importlib
import os
from pathlib import Path
import sys

import numpy as np

NATIVE_MODULE_NAME = "sim2real_native_v0_1"
NATIVE_PATH_ENV_VAR = "SIM2REAL_NATIVE_PATH"


def _iter_candidate_native_paths():
    # User override takes precedence and can include multiple paths.
    env_value = os.environ.get(NATIVE_PATH_ENV_VAR, "").strip()
    if env_value:
        for raw_path in env_value.split(os.pathsep):
            path = Path(raw_path).expanduser()
            if path.is_dir():
                yield path

    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parents[4] if len(module_dir.parents) > 4 else module_dir
    for path in (
        module_dir,
        project_root,
        Path.cwd(),
        Path.home() / "Downloads",
    ):
        if path.is_dir():
            yield path


def _load_native_module():
    seen = set()
    for path in _iter_candidate_native_paths():
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        if path_str not in sys.path:
            sys.path.append(path_str)

    try:
        native_module = importlib.import_module(NATIVE_MODULE_NAME)
        return native_module, None
    except ImportError as import_error:
        return None, import_error


sim2real_native, _NATIVE_IMPORT_ERROR = _load_native_module()
_NATIVE_AVAILABLE = sim2real_native is not None
if _NATIVE_AVAILABLE:
    print("[Sim2Real IMU] Native C++ backend loaded successfully.")
else:
    print(f"[Sim2Real IMU] WARNING: Could not load C++ backend: {_NATIVE_IMPORT_ERROR}")
    print(
        f"[Sim2Real IMU] Set {NATIVE_PATH_ENV_VAR} to the directory containing "
        f"{NATIVE_MODULE_NAME}.so if needed."
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
