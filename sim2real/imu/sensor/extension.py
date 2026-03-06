import zlib

import omni.ext
import omni.kit.menu.utils as menu_utils
import omni.usd
from omni.kit.menu.utils import MenuItemDescription

from .config import load_sensor_model_config, resolve_extension_root
from .noise import NativeNoiseBackend
from .runtime import ImuSensorRuntime

SUPPORTED_SENSOR_MODELS = ("ASM330LHH", "LSM6DSV")
SENSOR_MENU_ROOT = "STMicroelectronics IMU"
SENSOR_METADATA_PREFIX = "sim2real:"
SENSOR_MARKER_PRIM_NAME = "visual"


class StImuSensorExtension(omni.ext.IExt):
    def on_startup(self, extension_id):
        print("[Sim2Real IMU] Extension starting up")
        self._extension_id = extension_id
        self._extension_root = resolve_extension_root(extension_id)
        print(f"[Sim2Real IMU] Extension root: {self._extension_root}")

        self._noise_backend = NativeNoiseBackend()
        self._sensor_runtime = ImuSensorRuntime(self._noise_backend)
        self._sensor_runtime.start()

        self._menu_items = self._build_create_menu_items()
        menu_utils.add_menu_items(self._menu_items, "Create")
        menu_utils.rebuild_menus()

    def on_shutdown(self):
        if hasattr(self, "_sensor_runtime"):
            self._sensor_runtime.stop()

        if hasattr(self, "_menu_items"):
            menu_utils.remove_menu_items(self._menu_items, "Create")
            menu_utils.rebuild_menus()

        print("[Sim2Real IMU] Extension shut down")

    def _build_create_menu_items(self):
        sensor_model_entries = [
            MenuItemDescription(
                name=model_name,
                onclick_fn=lambda selected_model=model_name: self._spawn_sensor(selected_model),
            )
            for model_name in SUPPORTED_SENSOR_MODELS
        ]
        return [
            MenuItemDescription(
                name="Sensors",
                sub_menu=[
                    MenuItemDescription(
                        name=SENSOR_MENU_ROOT,
                        sub_menu=sensor_model_entries,
                    )
                ],
            )
        ]

    def _spawn_sensor(self, model_name: str = "ASM330LHH"):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            print("[Sim2Real IMU] ERROR: No stage open.")
            return

        try:
            model_config = load_sensor_model_config(self._extension_root, model_name)
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return

        selection = omni.usd.get_context().get_selection().get_selected_prim_paths()
        parent_prim_path = selection[0] if selection else "/World"
        sensor_prim_path = self._build_unique_sensor_path(stage, parent_prim_path, model_name)

        from pxr import Gf, UsdGeom, Vt

        UsdGeom.Xform.Define(stage, sensor_prim_path)
        sensor_prim = stage.GetPrimAtPath(sensor_prim_path)

        runtime_config = dict(model_config)
        runtime_config["attachPrimPath"] = parent_prim_path
        self._write_sensor_metadata(sensor_prim, model_name, runtime_config)

        marker_path = f"{sensor_prim_path}/{SENSOR_MARKER_PRIM_NAME}"
        marker_cube = UsdGeom.Cube.Define(stage, marker_path)
        marker_cube.GetSizeAttr().Set(0.05)
        marker_cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.03, 0.07, 0.18)]))
        marker_cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.03))

        visual_prim = stage.GetPrimAtPath(marker_path)
        visual_prim.SetCustomDataByKey("physics:collisionEnabled", False)

        seed = self._stable_seed(sensor_prim_path)
        self._sensor_runtime.register_sensor(sensor_prim_path, runtime_config, seed=seed)

        print(f"[Sim2Real IMU] Spawned {model_name} at {sensor_prim_path}")
        print(f"[Sim2Real IMU] Attached to: {parent_prim_path}")
        print(f"[Sim2Real IMU] Config: {runtime_config}")

    # Backward-compatible method name for existing README snippets.
    def _spawn_imu(self, model: str = "ASM330LHH"):
        self._spawn_sensor(model)

    def _build_unique_sensor_path(self, stage, parent_prim_path: str, model_name: str) -> str:
        base_path = f"{parent_prim_path}/{model_name}"
        sensor_prim_path = base_path
        suffix = 0
        while stage.GetPrimAtPath(sensor_prim_path).IsValid():
            suffix += 1
            sensor_prim_path = f"{base_path}_{suffix}"
        return sensor_prim_path

    def _write_sensor_metadata(self, sensor_prim, model_name: str, runtime_config: dict):
        sensor_prim.SetCustomDataByKey(f"{SENSOR_METADATA_PREFIX}enabled", True)
        sensor_prim.SetCustomDataByKey(f"{SENSOR_METADATA_PREFIX}model", model_name)
        for key, value in runtime_config.items():
            sensor_prim.SetCustomDataByKey(f"{SENSOR_METADATA_PREFIX}{key}", value)

    def _stable_seed(self, sensor_prim_path: str) -> int:
        # Derive deterministic per-sensor seed from path for independent engine state.
        return zlib.crc32(sensor_prim_path.encode("utf-8")) & 0x7FFFFFFF


# Backward-compatible class name for extension entry points/imports.
Sim2RealIMUSensorExtension = StImuSensorExtension
