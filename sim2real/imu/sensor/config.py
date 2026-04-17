# ******************************************************************************
# File Name          : config.py
# Description        : Utilities to resolve extension root and load per-model
#                      IMU JSON configuration.
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

import json
from pathlib import Path
import omni.kit.app


MODEL_CONFIG_DIR = Path("data") / "models"


def resolve_extension_root(extension_id: str) -> Path:
    """Return the root folder of this extension, regardless of where it is installed."""
    mgr = omni.kit.app.get_app().get_extension_manager()
    return Path(mgr.get_extension_path(extension_id))


def load_sensor_model_config(extension_root: Path, model_name: str) -> dict:
    """
    Load a model's config from data/models/<model>.json.
    Raises FileNotFoundError if the JSON does not exist.
    """
    config_path = extension_root / MODEL_CONFIG_DIR / f"{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"[Sim2Real IMU] No config file found for model '{model_name}' at {config_path}\n"
            f"Create {config_path.name} in the data/models/ folder to add this model."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            f"[Sim2Real IMU] Expected JSON object in {config_path}, got {type(config).__name__}."
        )

    odr_hz = float(config.get("odr_hz", 100.0))
    if odr_hz <= 0.0:
        raise ValueError(
            f"[Sim2Real IMU] Invalid odr_hz={odr_hz} in {config_path}. Value must be > 0."
        )
    config["odr_hz"] = odr_hz

    return config


# Backward-compatible aliases for existing imports.
get_ext_root = resolve_extension_root
load_model_config = load_sensor_model_config
