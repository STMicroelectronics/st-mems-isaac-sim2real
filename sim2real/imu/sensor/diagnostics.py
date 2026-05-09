# ******************************************************************************
# File Name          : diagnostics.py
# Description        : Environment validation CLI for the Sim2Real IMU Isaac Sim
#                      extension.
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

import argparse
import json
import os
import sys

# Keep diagnostic CLI output as one structured report. Normal extension startup
# still emits native backend load warnings.
os.environ.setdefault("SIM2REAL_SUPPRESS_NATIVE_BACKEND_STARTUP_LOGS", "1")

from .adapters import get_isaac_adapter
from .noise.native_backend import (
    get_native_backend_diagnostics,
    format_native_backend_diagnostics,
)


def _python_environment_record():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    in_virtualenv = bool(virtual_env) or sys.prefix != base_prefix
    in_conda = bool(conda_prefix)
    executable = sys.executable or ""
    looks_like_isaac = "isaac" in executable.lower() or "isaac" in sys.prefix.lower()
    return {
        "prefix": sys.prefix,
        "base_prefix": base_prefix,
        "executable": executable,
        "virtual_env": virtual_env,
        "conda_prefix": conda_prefix,
        "isolated": in_virtualenv or in_conda or looks_like_isaac,
        "environment_type": "conda" if in_conda else "venv" if in_virtualenv else "isaac" if looks_like_isaac else "system",
    }


def _check_isaac_adapter():
    try:
        adapter = get_isaac_adapter()
        return {
            "available": True,
            "isaac_version": getattr(adapter, "isaac_version", "unknown"),
            "adapter": adapter.__class__.__name__,
        }
    except Exception as error:  # noqa: BLE001 - diagnostics must capture Isaac import failures.
        return {
            "available": False,
            "error": str(error),
        }


def _check_physics_scene():
    try:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics
    except Exception as error:  # noqa: BLE001 - diagnostics must capture Isaac import failures.
        return {
            "available": False,
            "error": str(error),
        }

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {
            "available": False,
            "warning": "No stage is currently open.",
        }

    physics_scene_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physics_scene_prim = prim
            break

    if physics_scene_prim is None:
        return {
            "available": False,
            "warning": "No PhysicsScene found. Isaac will use the default 60 steps/sec until a scene is created.",
            "recommended_steps_per_second": 208.0,
        }

    physx_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    steps_per_second = physx_api.GetTimeStepsPerSecondAttr().Get()
    if steps_per_second in (None, 0):
        steps_per_second = 60.0

    recommended_steps_per_second = 208.0
    return {
        "available": True,
        "path": str(physics_scene_prim.GetPath()),
        "steps_per_second": float(steps_per_second),
        "dt_seconds": 1.0 / float(steps_per_second),
        "recommended_steps_per_second": recommended_steps_per_second,
        "meets_recommendation": float(steps_per_second) + 1e-6 >= recommended_steps_per_second,
    }


def run_diagnostics():
    native = get_native_backend_diagnostics(include_import_check=True, include_smoke_test=True)
    adapter = _check_isaac_adapter()
    physics_scene = _check_physics_scene()
    environment = _python_environment_record()
    passed = bool(
        native.get("import_check", {}).get("available")
        and native.get("smoke_test", {}).get("passed")
        and adapter.get("available")
        and environment.get("isolated")
    )
    return {
        "passed": passed,
        "environment": environment,
        "native_backend": native,
        "isaac_adapter": adapter,
        "physics_scene": physics_scene,
    }


def _format_environment(record):
    status = "PASS" if record.get("isolated") else "WARN"
    lines = [
        f"[{status}] Python environment isolation",
        f"  Type: {record.get('environment_type')}",
        f"  Executable: {record.get('executable')}",
        f"  Prefix: {record.get('prefix')}",
    ]
    if record.get("conda_prefix"):
        lines.append(f"  CONDA_PREFIX: {record['conda_prefix']}")
    if record.get("virtual_env"):
        lines.append(f"  VIRTUAL_ENV: {record['virtual_env']}")
    if not record.get("isolated"):
        lines.append("  Recommendation: run diagnostics from Isaac's Python, Conda, or a virtualenv.")
    return "\n".join(lines)


def format_diagnostics(report, verbose=False):
    lines = [
        "Sim2Real IMU Environment Diagnostics",
        "====================================",
        f"Overall: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        _format_environment(report["environment"]),
        "",
        format_native_backend_diagnostics(report["native_backend"], verbose=verbose),
        "",
    ]

    adapter = report["isaac_adapter"]
    if adapter.get("available"):
        lines.extend(
            [
                "[PASS] Isaac IMU adapter",
                f"  Adapter: {adapter.get('adapter')}",
                f"  Isaac Sim version: {adapter.get('isaac_version')}",
            ]
        )
    else:
        lines.extend(
            [
                "[FAIL] Isaac IMU adapter",
                f"  Error: {adapter.get('error')}",
                "  Recommendation: run this command from inside the target Isaac Sim Python environment.",
            ]
        )

    lines.append("")
    physics_scene = report["physics_scene"]
    if physics_scene.get("available"):
        status = "PASS" if physics_scene.get("meets_recommendation") else "WARN"
        lines.extend(
            [
                f"[{status}] Physics step configuration",
                f"  PhysicsScene: {physics_scene.get('path')}",
                f"  Steps/sec: {physics_scene.get('steps_per_second')}",
                f"  dt: {physics_scene.get('dt_seconds'):.6f}s",
                "  Repo baseline recommendation: 208 Hz",
            ]
        )
        if not physics_scene.get("meets_recommendation"):
            lines.append(
                "  Recommendation: increase Physics Scene steps/sec to 208 for the current public Sim2Real baseline."
            )
    else:
        lines.extend(
            [
                "[WARN] Physics step configuration",
                f"  Status: {physics_scene.get('warning') or physics_scene.get('error')}",
                "  Repo baseline recommendation: 208 Hz",
                "  Recommendation: create or configure a PhysicsScene before validation.",
            ]
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate the Sim2Real IMU Isaac Sim environment.")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed candidate and search-path data.")
    args = parser.parse_args(argv)

    report = run_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_diagnostics(report, verbose=args.verbose))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
