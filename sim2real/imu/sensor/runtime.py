# ******************************************************************************
# File Name          : runtime.py
# Description        : Physics-step runtime scheduler for ST IMU sensor ticks
#                      and realistic sample publishing.
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

import omni.physx
import omni.timeline
import omni.usd

from .adapters import get_isaac_adapter


class ImuSensorRuntime:
    """
    Subscribes to Isaac Sim physics step events and ticks each
    IMU prim tagged with sim2real:enabled=True at its configured ODR.

    Data flow per tick:
        Native Isaac IMUSensor (clean physics truth)
                    ->
          C++ sensor-realism backend (native wrapper)
                    ->
          Custom sim2real prim (stores realistic result as custom data)
    """

    DEFAULT_ODR_HZ = 100.0
    STAGE_DISCOVERY_INTERVAL_S = 1.0
    RECOMMENDED_PHYSICS_STEPS_PER_SECOND = 208.0
    TRUTH_SENSOR_PRIM_NAME = "Imu_Sensor"
    SENSOR_METADATA_PREFIX = "sim2real:"
    SENSOR_ENABLED_KEY = f"{SENSOR_METADATA_PREFIX}enabled"
    SENSOR_MODEL_KEY = f"{SENSOR_METADATA_PREFIX}model"
    TRUTH_SENSOR_PATH_KEY = f"{SENSOR_METADATA_PREFIX}truthImuPrimPath"
    LAST_LIN_ACC_KEY = f"{SENSOR_METADATA_PREFIX}last_lin_acc"
    LAST_ANG_VEL_KEY = f"{SENSOR_METADATA_PREFIX}last_ang_vel"

    def __init__(
        self,
        noise_backend,
        stage_discovery_interval_s: float = STAGE_DISCOVERY_INTERVAL_S,
        isaac_adapter=None,
    ):
        self._backend = noise_backend
        self._isaac_adapter = isaac_adapter
        self._physx_sub = None
        self._timeline = omni.timeline.get_timeline_interface()
        self._stage_discovery_interval_s = max(float(stage_discovery_interval_s), 0.1)
        self._time_since_stage_scan_s = self._stage_discovery_interval_s

        # prim_path -> accumulated dt waiting to fire next sensor tick
        self._sample_accumulators_s = {}

        # prim_path -> sim_time of last tick
        self._last_tick_sim_time_s = {}

        # prim_path -> config dict
        self._sensor_registry = {}

        # prim_path -> cached Isaac IMUSensor instance (initialized once at registration)
        self._truth_sensor_cache = {}

        # prim_paths waiting for a truth IMU binding retry once physics is ready.
        self._truth_sensor_pending = set()

        # prim_path -> last initialization error string to suppress repeated log spam.
        self._truth_sensor_init_errors = {}

        # Warn once if the stage physics rate is below the current repo baseline.
        self._physics_rate_warning_emitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._physx_sub is not None:
            return
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)
        print("[Sim2Real Runtime] Physics step subscription active.")

    def stop(self):
        self._physx_sub = None
        self._sample_accumulators_s.clear()
        self._last_tick_sim_time_s.clear()
        self._sensor_registry.clear()
        self._truth_sensor_cache.clear()
        self._truth_sensor_pending.clear()
        self._truth_sensor_init_errors.clear()
        self._physics_rate_warning_emitted = False
        print("[Sim2Real Runtime] Stopped.")

    def register_sensor(self, sensor_prim_path: str, sensor_config: dict, seed: int = 123):
        """
        Explicitly register a sensor prim so the runtime tracks it.
        Called by extension right after creating the prim.
        """
        normalized_config = dict(sensor_config or {})
        normalized_config["odr_hz"] = self._normalize_odr_hz(
            normalized_config.get("odr_hz"), sensor_prim_path
        )

        if "attachPrimPath" not in normalized_config:
            attach_prim_path = self._read_attach_path_from_prim(sensor_prim_path)
            if attach_prim_path:
                normalized_config["attachPrimPath"] = attach_prim_path

        self._sensor_registry[sensor_prim_path] = normalized_config
        self._sample_accumulators_s[sensor_prim_path] = 0.0

        if hasattr(self._backend, "register_sensor"):
            self._backend.register_sensor(sensor_prim_path, normalized_config, seed=seed)
        else:
            self._backend.register(sensor_prim_path, normalized_config, seed=seed)

        attach_prim_path = normalized_config.get("attachPrimPath", "")
        truth_sensor_prim_path = normalized_config.get("truthImuPrimPath", "")
        if attach_prim_path:
            self._ensure_truth_sensor_initialized(
                sensor_prim_path,
            )
        else:
            print(
                f"[Sim2Real Runtime] WARNING: No attachPrimPath for {sensor_prim_path}. "
                f"Truth kinematics will be unavailable."
            )

        print(f"[Sim2Real Runtime] Registered IMU: {sensor_prim_path}")

    def unregister_sensor(self, sensor_prim_path: str):
        self._sensor_registry.pop(sensor_prim_path, None)
        self._sample_accumulators_s.pop(sensor_prim_path, None)
        self._last_tick_sim_time_s.pop(sensor_prim_path, None)
        self._truth_sensor_cache.pop(sensor_prim_path, None)
        self._truth_sensor_pending.discard(sensor_prim_path)
        self._truth_sensor_init_errors.pop(sensor_prim_path, None)

        if hasattr(self._backend, "unregister_sensor"):
            self._backend.unregister_sensor(sensor_prim_path)
        else:
            self._backend.unregister(sensor_prim_path)

        print(f"[Sim2Real Runtime] Unregistered IMU: {sensor_prim_path}")

    # Backward-compatible method names.
    def register_imu(self, prim_path: str, config: dict, seed: int = 123):
        self.register_sensor(prim_path, config, seed=seed)

    def unregister_imu(self, prim_path: str):
        self.unregister_sensor(prim_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initialize_truth_sensor(
        self,
        sensor_prim_path: str,
        attach_prim_path: str,
        configured_truth_sensor_path: str = "",
        stage=None,
    ) -> bool:
        """
        Create and initialize a native Isaac IMUSensor for the given attach link.
        Resolution order:
          1. Explicit sim2real:truthImuPrimPath if present and valid
          2. Default child <attach_prim_path>/Imu_Sensor
          3. First IMU-like descendant under attachPrimPath
          4. Auto-create <attach_prim_path>/Imu_Sensor as a final fallback
        """
        if stage is None:
            stage = omni.usd.get_context().get_stage()
        if not stage:
            print(
                f"[Sim2Real Runtime] WARNING: No stage while initializing truth IMU for "
                f"{sensor_prim_path}."
            )
            return False

        try:
            truth_sensor_path, resolution_source = self._resolve_truth_sensor_path(
                stage,
                attach_prim_path,
                configured_truth_sensor_path=configured_truth_sensor_path,
            )
            if truth_sensor_path is None:
                truth_sensor_path = f"{attach_prim_path}/{self.TRUTH_SENSOR_PRIM_NAME}"
                sensor = self._create_truth_sensor(truth_sensor_path)
                resolution_source = "auto-created"
                print(
                    f"[Sim2Real Runtime] Auto-created native Isaac IMU at "
                    f"{truth_sensor_path} for {sensor_prim_path}."
                )
            else:
                sensor = self._get_isaac_adapter().create_imu_sensor(prim_path=truth_sensor_path)
                sensor.initialize()

            self._truth_sensor_cache[sensor_prim_path] = sensor
            self._truth_sensor_pending.discard(sensor_prim_path)
            self._truth_sensor_init_errors.pop(sensor_prim_path, None)
            sensor_config = self._sensor_registry.get(sensor_prim_path)
            if sensor_config is not None:
                sensor_config["truthImuPrimPath"] = truth_sensor_path
            self._persist_truth_sensor_path(sensor_prim_path, truth_sensor_path)
            print(
                f"[Sim2Real Runtime] Bound native Isaac IMU for {sensor_prim_path}: "
                f"{truth_sensor_path} ({resolution_source})"
            )
            return True

        except Exception as error:
            self._truth_sensor_pending.add(sensor_prim_path)
            self._record_truth_sensor_init_failure(
                sensor_prim_path,
                attach_prim_path,
                error,
            )
            return False

    def _ensure_truth_sensor_initialized(self, sensor_prim_path: str, stage=None) -> bool:
        if sensor_prim_path in self._truth_sensor_cache:
            return True

        sensor_config = self._sensor_registry.get(sensor_prim_path, {})
        attach_prim_path = sensor_config.get("attachPrimPath", "")
        if not attach_prim_path:
            attach_prim_path = self._read_attach_path_from_prim(sensor_prim_path)
            if attach_prim_path:
                sensor_config["attachPrimPath"] = attach_prim_path

        if not attach_prim_path:
            return False

        configured_truth_sensor_path = sensor_config.get("truthImuPrimPath", "")
        return self._initialize_truth_sensor(
            sensor_prim_path,
            attach_prim_path,
            configured_truth_sensor_path=configured_truth_sensor_path,
            stage=stage,
        )

    def _record_truth_sensor_init_failure(self, sensor_prim_path: str, attach_prim_path: str, error):
        error_text = str(error)
        if self._truth_sensor_init_errors.get(sensor_prim_path) == error_text:
            return

        self._truth_sensor_init_errors[sensor_prim_path] = error_text
        print(
            f"[Sim2Real Runtime] ERROR: Could not initialize native Isaac sensor "
            f"for {sensor_prim_path}: {error_text}"
        )

        if "current_physics_prim" in error_text:
            print(
                "[Sim2Real Runtime] Truth IMU initialization is waiting for a valid "
                "physics scene. The runtime will retry automatically."
            )
            return

        print(
            "[Sim2Real Runtime] Hint: set sim2real:truthImuPrimPath explicitly, "
            f"or create a child IMU under {attach_prim_path}, or allow auto-creation "
            f"at {attach_prim_path}/{self.TRUTH_SENSOR_PRIM_NAME}."
        )

    def _resolve_truth_sensor_path(
        self,
        stage,
        attach_prim_path: str,
        configured_truth_sensor_path: str = "",
    ) -> tuple[str | None, str]:
        if configured_truth_sensor_path:
            prim = stage.GetPrimAtPath(configured_truth_sensor_path)
            if prim.IsValid():
                return configured_truth_sensor_path, "configured truthImuPrimPath"
            print(
                f"[Sim2Real Runtime] WARNING: Configured truthImuPrimPath "
                f"{configured_truth_sensor_path} was not found. Falling back to auto-discovery."
            )

        default_truth_sensor_path = f"{attach_prim_path}/{self.TRUTH_SENSOR_PRIM_NAME}"
        prim = stage.GetPrimAtPath(default_truth_sensor_path)
        if prim.IsValid():
            return default_truth_sensor_path, "default child path"

        discovered_path = self._discover_truth_sensor_path(stage, attach_prim_path)
        if discovered_path:
            return discovered_path, "discovered IMU-like descendant"

        print(
            f"[Sim2Real Runtime] WARNING: No native Isaac IMU prim found under "
            f"{attach_prim_path}. Attempting auto-creation at {default_truth_sensor_path}."
        )
        print("[Sim2Real Runtime] Hint: Run this in Script Editor to locate IMU prims:")
        print("    for p in stage.Traverse():")
        print("        if 'imu' in str(p.GetPath()).lower(): print(p.GetPath())")
        return None, "auto-create fallback"

    def _discover_truth_sensor_path(self, stage, attach_prim_path: str) -> str | None:
        attach_prefix = f"{attach_prim_path.rstrip('/')}/"
        candidates = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(attach_prefix):
                continue
            leaf_name = prim_path.rsplit("/", 1)[-1].lower()
            if "imu" not in leaf_name:
                continue
            exact_default = leaf_name == self.TRUTH_SENSOR_PRIM_NAME.lower()
            generic_name = leaf_name in {"imu", "imu_sensor"}
            score = (
                0 if exact_default else
                1 if generic_name else
                2
            )
            depth = prim_path.count("/")
            candidates.append((score, depth, prim_path))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _create_truth_sensor(self, truth_sensor_path: str):
        sensor = self._get_isaac_adapter().create_imu_sensor(
            prim_path=truth_sensor_path,
            name=self.TRUTH_SENSOR_PRIM_NAME,
        )
        sensor.initialize()
        return sensor

    def _persist_truth_sensor_path(self, sensor_prim_path: str, truth_sensor_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        prim = stage.GetPrimAtPath(sensor_prim_path)
        if prim.IsValid():
            prim.SetCustomDataByKey(self.TRUTH_SENSOR_PATH_KEY, truth_sensor_path)

    def _on_physics_step(self, dt: float):
        if not self._timeline.is_playing():
            return

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        # Scan stage periodically for pre-existing IMU prims loaded from USD files.
        self._time_since_stage_scan_s += dt
        if self._time_since_stage_scan_s >= self._stage_discovery_interval_s:
            self._time_since_stage_scan_s = 0.0
            self._discover_sensors_from_stage(stage)

        self._maybe_warn_on_physics_rate(dt)

        sim_time = float(self._timeline.get_current_time())

        for sensor_prim_path, sensor_config in list(self._sensor_registry.items()):
            odr_hz = self._normalize_odr_hz(sensor_config.get("odr_hz"), sensor_prim_path)
            sensor_config["odr_hz"] = odr_hz
            sensor_period_s = 1.0 / odr_hz
            accumulated_dt = self._sample_accumulators_s.get(sensor_prim_path, 0.0) + dt

            if sensor_config.get("attachPrimPath") and sensor_prim_path not in self._truth_sensor_cache:
                self._ensure_truth_sensor_initialized(sensor_prim_path, stage=stage)

            if sensor_config.get("attachPrimPath") and sensor_prim_path not in self._truth_sensor_cache:
                self._sample_accumulators_s[sensor_prim_path] = min(accumulated_dt, sensor_period_s)
                continue

            while accumulated_dt >= sensor_period_s:
                accumulated_dt -= sensor_period_s
                self._tick_sensor(stage, sensor_prim_path, sim_time)

            self._sample_accumulators_s[sensor_prim_path] = accumulated_dt

    def _tick_sensor(self, stage, sensor_prim_path: str, sim_time: float):
        prim = stage.GetPrimAtPath(sensor_prim_path)
        if not prim.IsValid():
            self.unregister_sensor(sensor_prim_path)
            return

        truth_kinematics = self._read_truth_kinematics(sensor_prim_path)
        if truth_kinematics is None:
            return

        if hasattr(self._backend, "step_sensor"):
            realistic_kinematics = self._backend.step_sensor(
                sensor_prim_path,
                sim_time,
                truth_kinematics,
            )
        else:
            realistic_kinematics = self._backend.step(
                sensor_prim_path,
                sim_time,
                truth_kinematics,
            )

        self._last_tick_sim_time_s[sensor_prim_path] = sim_time

    def _read_truth_kinematics(self, sensor_prim_path: str) -> dict | None:
        """
        Read one frame from the cached native Isaac IMUSensor for this prim.
        Returns dict with 'lin_acc' and 'ang_vel' (gravity-inclusive, body frame),
        or None if the sensor is unavailable.
        """
        sensor = self._truth_sensor_cache.get(sensor_prim_path)
        if sensor is None:
            return None

        try:
            # read_gravity=True: lin_acc includes gravitational specific force,
            # which is exactly what a real IMU measures and what the C++ engine expects.
            raw = sensor.get_current_frame(read_gravity=True)
            return self._get_isaac_adapter().normalize_imu_frame(raw) if raw is not None else None
        except Exception as error:
            print(
                f"[Sim2Real Runtime] _read_truth_kinematics error for "
                f"{sensor_prim_path}: {error}"
            )
            return None

    def _discover_sensors_from_stage(self, stage):
        """
        Scan the stage for any sim2real:enabled prims not yet in the registry.
        Catches prims loaded from saved USD files.
        """
        for prim in stage.Traverse():
            custom_data = prim.GetCustomData()
            if not custom_data or not custom_data.get(self.SENSOR_ENABLED_KEY, False):
                continue

            sensor_prim_path = str(prim.GetPath())
            if sensor_prim_path in self._sensor_registry:
                continue

            rebuilt_config = {
                key.replace(self.SENSOR_METADATA_PREFIX, ""): value
                for key, value in custom_data.items()
                if key.startswith(self.SENSOR_METADATA_PREFIX)
                and key
                not in (
                    self.SENSOR_ENABLED_KEY,
                    self.SENSOR_MODEL_KEY,
                    self.LAST_LIN_ACC_KEY,
                    self.LAST_ANG_VEL_KEY,
                )
            }

            print(f"[Sim2Real Runtime] Auto-discovered IMU from stage: {sensor_prim_path}")
            self.register_sensor(sensor_prim_path, rebuilt_config)

    def _read_attach_path_from_prim(self, sensor_prim_path: str) -> str:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return ""

        prim = stage.GetPrimAtPath(sensor_prim_path)
        if not prim.IsValid():
            return ""

        return str(prim.GetCustomDataByKey(f"{self.SENSOR_METADATA_PREFIX}attachPrimPath") or "")

    def _get_isaac_adapter(self):
        if self._isaac_adapter is None:
            self._isaac_adapter = get_isaac_adapter()
        return self._isaac_adapter

    def _normalize_odr_hz(self, raw_odr_hz, sensor_prim_path: str) -> float:
        try:
            odr_hz = float(raw_odr_hz)
        except (TypeError, ValueError):
            print(
                f"[Sim2Real Runtime] WARNING: Invalid odr_hz={raw_odr_hz} for "
                f"{sensor_prim_path}. Using default {self.DEFAULT_ODR_HZ}Hz."
            )
            return self.DEFAULT_ODR_HZ

        if odr_hz <= 0.0:
            print(
                f"[Sim2Real Runtime] WARNING: Non-positive odr_hz={odr_hz} for "
                f"{sensor_prim_path}. Using default {self.DEFAULT_ODR_HZ}Hz."
            )
            return self.DEFAULT_ODR_HZ

        return odr_hz

    def _maybe_warn_on_physics_rate(self, dt: float):
        if self._physics_rate_warning_emitted:
            return
        if dt <= 0.0 or not self._sensor_registry:
            return

        physics_steps_per_second = 1.0 / float(dt)
        if physics_steps_per_second + 1e-6 >= self.RECOMMENDED_PHYSICS_STEPS_PER_SECOND:
            return

        print(
            "[Sim2Real Runtime] WARNING: Current physics step is approximately "
            f"{physics_steps_per_second:.3f} Hz based on dt={dt:.6f}s. "
            f"This repo currently recommends {self.RECOMMENDED_PHYSICS_STEPS_PER_SECOND:.0f} Hz "
            "physics for the public Sim2Real IMU baseline to reduce timing quantization."
        )
        self._physics_rate_warning_emitted = True



# Backward-compatible class name for existing imports.
Sim2RealRuntime = ImuSensorRuntime
