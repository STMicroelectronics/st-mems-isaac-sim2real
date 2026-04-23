# ******************************************************************************
# File Name          : native_backend.py
# Description        : Python wrapper around the native C++ backend used for
#                      ST IMU sim2real signal generation.
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

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import sys

import numpy as np

NATIVE_MODULE_NAME = "sim2real_native_v0_1"
LEGACY_NATIVE_MODULE_NAME = "sim2real_native"
NATIVE_MODULE_GLOB = "sim2real_native*.so"
NATIVE_PATH_ENV_VAR = "SIM2REAL_NATIVE_PATH"
NATIVE_SUPPRESS_STARTUP_LOGS_ENV_VAR = "SIM2REAL_SUPPRESS_NATIVE_BACKEND_STARTUP_LOGS"
NATIVE_MANIFEST_NAME = "manifest.json"


def _runtime_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _runtime_architecture():
    machine = platform.machine().lower()
    return "x86_64" if machine in ("x86_64", "amd64") else machine


def _runtime_os():
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def _runtime_glibc_version():
    if _runtime_os() != "linux":
        return None
    libc_name, libc_version = platform.libc_ver()
    if libc_name and libc_name.lower() != "glibc":
        return None
    return libc_version or None


def _version_tuple(value):
    if value is None:
        return None
    parts = re.findall(r"\d+", str(value))
    return tuple(int(part) for part in parts[:3]) if parts else None


def _version_gte(actual, required):
    actual_tuple = _version_tuple(actual)
    required_tuple = _version_tuple(required)
    if actual_tuple is None or required_tuple is None:
        return True
    width = max(len(actual_tuple), len(required_tuple))
    return actual_tuple + (0,) * (width - len(actual_tuple)) >= required_tuple + (0,) * (
        width - len(required_tuple)
    )


def _extension_root():
    module_dir = Path(__file__).resolve().parent
    return module_dir.parents[3] if len(module_dir.parents) > 3 else module_dir


def _iter_search_locations():
    """Return native search locations, including direct file overrides."""
    seen = set()
    env_value = os.environ.get(NATIVE_PATH_ENV_VAR, "").strip()
    if env_value:
        for raw_path in env_value.split(os.pathsep):
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "source": NATIVE_PATH_ENV_VAR,
                "path": path,
                "kind": "file" if path.is_file() else "directory" if path.is_dir() else "missing",
                "exists": path.exists(),
            }

    module_dir = Path(__file__).resolve().parent
    root = _extension_root()
    for path in (module_dir, root, root / "sim_binary", root / "lib"):
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        yield {
            "source": "default",
            "path": path,
            "kind": "directory" if path.is_dir() else "missing",
            "exists": path.exists(),
        }


def _infer_python_from_filename(path):
    match = re.search(r"cpython-(\d)(\d+)", path.name)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def _infer_arch_from_filename(path):
    name = path.name.lower()
    if "x86_64" in name or "amd64" in name:
        return "x86_64"
    if "aarch64" in name or "arm64" in name:
        return "aarch64"
    return None


def _infer_module_from_filename(path):
    if path.name.startswith(f"{LEGACY_NATIVE_MODULE_NAME}."):
        return LEGACY_NATIVE_MODULE_NAME
    return NATIVE_MODULE_NAME


def _sha1(path):
    digest = hashlib.sha1()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_pyinit_symbol(path, module_name):
    try:
        return f"PyInit_{module_name}".encode("ascii") in path.read_bytes()
    except OSError:
        return False


def _entry_glibc_floor(entry):
    if "min_glibc" in entry:
        return str(entry.get("min_glibc"))
    if "glibc_symbol_floor" in entry:
        return str(entry.get("glibc_symbol_floor"))
    if "max_glibc" in entry:
        return str(entry.get("max_glibc"))
    return None


def _manifest_deprecation_warnings(entry):
    warnings = []
    if "max_glibc" in entry:
        warnings.append(
            "manifest field 'max_glibc' is deprecated; use 'min_glibc' for the "
            "minimum required runtime glibc version"
        )
    return warnings


def _read_manifest(directory):
    manifest_path = directory / NATIVE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None, None
    try:
        return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return manifest_path, {"_error": str(error)}


def _candidate_from_manifest(entry, manifest_module, manifest_path):
    native_file = manifest_path.parent / str(entry.get("file", ""))
    module_name = str(entry.get("module") or manifest_module or NATIVE_MODULE_NAME)
    return {
        "path": native_file,
        "module": module_name,
        "source": f"manifest:{manifest_path}",
        "manifest_path": manifest_path,
        "entry": dict(entry),
        "python": str(entry.get("python") or _infer_python_from_filename(native_file) or ""),
        "isaac_sim": str(entry.get("isaac_sim") or ""),
        "os": str(entry.get("os") or ""),
        "arch": str(entry.get("arch") or _infer_arch_from_filename(native_file) or ""),
        "min_glibc": _entry_glibc_floor(entry),
        "sha1": str(entry.get("sha1") or ""),
        "warnings": _manifest_deprecation_warnings(entry),
    }


def _candidate_from_file(path, source):
    return {
        "path": path,
        "module": _infer_module_from_filename(path),
        "source": source,
        "manifest_path": None,
        "entry": None,
        "python": _infer_python_from_filename(path) or "",
        "isaac_sim": "",
        "os": "linux" if path.suffix == ".so" else "",
        "arch": _infer_arch_from_filename(path) or "",
        "min_glibc": None,
        "sha1": "",
        "warnings": [],
    }


def _evaluate_candidate(candidate, runtime):
    reasons = []
    warnings = list(candidate.get("warnings", []))
    path = candidate["path"]

    if not path.is_file():
        reasons.append("file does not exist")
    else:
        candidate["actual_sha1"] = _sha1(path)

    candidate_os = str(candidate.get("os") or "").lower()
    if candidate_os and candidate_os != runtime["os"]:
        reasons.append(f"OS mismatch: runtime {runtime['os']}, binary {candidate_os}")

    candidate_python = str(candidate.get("python") or "").strip()
    if candidate_python and candidate_python != runtime["python"]:
        reasons.append(f"Python ABI mismatch: runtime {runtime['python']}, binary {candidate_python}")
    elif not candidate_python:
        warnings.append("binary has no Python ABI metadata; import attempt may still fail")

    candidate_arch = str(candidate.get("arch") or "").lower()
    if candidate_arch and candidate_arch != runtime["arch"]:
        reasons.append(f"architecture mismatch: runtime {runtime['arch']}, binary {candidate_arch}")

    min_glibc = candidate.get("min_glibc")
    if runtime.get("glibc") and min_glibc and not _version_gte(runtime["glibc"], min_glibc):
        reasons.append(f"glibc too old: runtime {runtime['glibc']}, required >= {min_glibc}")

    expected_sha1 = str(candidate.get("sha1") or "")
    if expected_sha1 and candidate.get("actual_sha1") and candidate["actual_sha1"] != expected_sha1:
        reasons.append(
            f"sha1 mismatch: manifest {expected_sha1}, actual {candidate['actual_sha1']}"
        )

    if path.is_file() and not _has_pyinit_symbol(path, candidate["module"]):
        reasons.append(f"missing PyInit_{candidate['module']} symbol")

    candidate["status"] = "accepted" if not reasons else "skipped"
    candidate["reasons"] = reasons
    candidate["warnings"] = warnings
    candidate["path"] = str(path)
    candidate["manifest_path"] = str(candidate["manifest_path"]) if candidate.get("manifest_path") else ""
    return candidate


def _discover_candidates(runtime):
    candidates = []
    manifests = []
    seen = set()

    for location in _iter_search_locations():
        path = location["path"]
        if location["kind"] == "file":
            key = (str(path.resolve()), _infer_module_from_filename(path))
            if key not in seen:
                seen.add(key)
                candidates.append(_candidate_from_file(path, f"{location['source']}:file"))
            continue

        if location["kind"] != "directory":
            continue

        manifest_path, manifest = _read_manifest(path)
        if manifest_path:
            manifest_record = {"path": str(manifest_path), "status": "loaded"}
            if manifest and manifest.get("_error"):
                manifest_record["status"] = "error"
                manifest_record["error"] = manifest["_error"]
            manifests.append(manifest_record)

            if manifest and not manifest.get("_error"):
                manifest_module = str(manifest.get("module") or NATIVE_MODULE_NAME)
                if manifest_module != NATIVE_MODULE_NAME:
                    manifests[-1]["warning"] = (
                        f"manifest module is {manifest_module}; expected {NATIVE_MODULE_NAME}"
                    )
                for entry in manifest.get("binaries", []):
                    candidate = _candidate_from_manifest(entry, manifest_module, manifest_path)
                    key = (str(Path(candidate["path"]).resolve()), candidate["module"])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)

        # Report matching files even when they are not in a manifest.
        for native_file in sorted(path.glob(NATIVE_MODULE_GLOB)):
            key = (str(native_file.resolve()), _infer_module_from_filename(native_file))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_candidate_from_file(native_file, f"{location['source']}:scan"))

    return [_evaluate_candidate(candidate, runtime) for candidate in candidates], manifests


def _runtime_record():
    return {
        "python": _runtime_python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "os": _runtime_os(),
        "machine": platform.machine(),
        "arch": _runtime_architecture(),
        "glibc": _runtime_glibc_version(),
        "extension_suffixes": list(importlib.machinery.EXTENSION_SUFFIXES),
    }


def _expected_native_filenames(runtime):
    return [f"{NATIVE_MODULE_NAME}{suffix}" for suffix in runtime["extension_suffixes"]]


def _public_search_locations():
    records = []
    for location in _iter_search_locations():
        records.append(
            {
                "source": location["source"],
                "path": str(location["path"]),
                "kind": location["kind"],
                "exists": location["exists"],
            }
        )
    return records


def _load_native_file(native_file: Path, module_name: str = NATIVE_MODULE_NAME):
    spec = importlib.util.spec_from_file_location(module_name, native_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {native_file}")

    native_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = native_module
    spec.loader.exec_module(native_module)
    if module_name != NATIVE_MODULE_NAME:
        sys.modules[NATIVE_MODULE_NAME] = native_module
    return native_module


def _candidate_module_from_cache(candidate):
    for module_name in (candidate["module"], NATIVE_MODULE_NAME):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "Sim2RealCore"):
            return module
    return None


def _run_smoke_test(native_module):
    engine = native_module.Sim2RealCore(123)
    if hasattr(engine, "update_configuration"):
        engine.update_configuration({"accel_fs_g": 8.0, "gyro_fs_dps": 2000.0, "odr_hz": 104.0})
    result = engine.process(np.zeros(3, dtype=float), np.zeros(3, dtype=float), 0.0)
    if not isinstance(result, dict):
        raise RuntimeError(f"expected dict from Sim2RealCore.process, got {type(result).__name__}")
    if "lin_acc" not in result or "ang_vel" not in result:
        raise RuntimeError("Sim2RealCore.process result missing lin_acc or ang_vel")
    return "Sim2RealCore smoke test passed"


def get_native_backend_diagnostics(include_import_check=False, include_smoke_test=False):
    """Return JSON-serializable native backend environment diagnostics."""
    runtime = _runtime_record()
    candidates, manifests = _discover_candidates(runtime)
    accepted = [candidate for candidate in candidates if candidate["status"] == "accepted"]
    diagnostics = {
        "runtime": runtime,
        "env": {NATIVE_PATH_ENV_VAR: os.environ.get(NATIVE_PATH_ENV_VAR, "")},
        "extension_root": str(_extension_root()),
        "expected_filenames": _expected_native_filenames(runtime),
        "search_locations": _public_search_locations(),
        "manifests": manifests,
        "candidates": candidates,
        "accepted_candidates": accepted,
        "compatible": bool(accepted),
        "import_check": {"requested": include_import_check, "available": False},
        "smoke_test": {"requested": include_smoke_test, "passed": False},
    }

    if not accepted:
        available_abis = sorted({candidate.get("python") for candidate in candidates if candidate.get("python")})
        diagnostics["summary"] = (
            f"No compatible native backend found for Python {runtime['python']} "
            f"on {runtime['os']}/{runtime['arch']}."
        )
        diagnostics["available_python_abis"] = available_abis
        return diagnostics

    diagnostics["summary"] = "Compatible native backend candidate found."

    if include_import_check:
        candidate = accepted[0]
        try:
            native_module = _candidate_module_from_cache(candidate)
            if native_module is None:
                native_module = _load_native_file(Path(candidate["path"]), candidate["module"])
            diagnostics["import_check"] = {
                "requested": True,
                "available": True,
                "module": getattr(native_module, "__name__", candidate["module"]),
                "file": getattr(native_module, "__file__", candidate["path"]),
            }
            if include_smoke_test:
                diagnostics["smoke_test"] = {
                    "requested": True,
                    "passed": True,
                    "message": _run_smoke_test(native_module),
                }
        except Exception as error:  # noqa: BLE001 - diagnostics must capture all loader failures.
            diagnostics["import_check"] = {
                "requested": True,
                "available": False,
                "candidate": candidate["path"],
                "module": candidate["module"],
                "error": str(error),
            }
            if include_smoke_test:
                diagnostics["smoke_test"] = {
                    "requested": True,
                    "passed": False,
                    "error": "native import failed before smoke test",
                }
    return diagnostics


def format_native_backend_diagnostics(diagnostics, verbose=False):
    runtime = diagnostics["runtime"]
    lines = [
        "[Sim2Real IMU] Native backend diagnostics:",
        f"  Runtime Python: {runtime['python']} ({runtime['executable']})",
        f"  Platform: {runtime['platform']}",
        f"  Architecture: {runtime['arch']}",
        f"  glibc: {runtime.get('glibc') or 'not detected / not Linux'}",
        f"  {NATIVE_PATH_ENV_VAR}: {diagnostics['env'].get(NATIVE_PATH_ENV_VAR) or '<unset>'}",
        f"  Extension root: {diagnostics['extension_root']}",
        f"  Expected filenames: {', '.join(diagnostics['expected_filenames'])}",
    ]

    if diagnostics.get("accepted_candidates"):
        lines.append("  Compatible native backend candidates:")
        for candidate in diagnostics["accepted_candidates"]:
            lines.append(f"    PASS {candidate['path']} (module={candidate['module']})")
    else:
        lines.append("  Compatible native backend candidates: none")
        available = diagnostics.get("available_python_abis") or []
        if available:
            lines.append(f"  Available Python ABIs in discovered binaries: {', '.join(available)}")

    if verbose or not diagnostics.get("accepted_candidates"):
        lines.append("  Search locations:")
        for location in diagnostics["search_locations"]:
            lines.append(
                f"    {location['source']} {location['kind']} {location['path']} "
                f"exists={location['exists']}"
            )
        lines.append("  Discovered native binaries:")
        if not diagnostics["candidates"]:
            lines.append("    none")
        for candidate in diagnostics["candidates"]:
            reasons = "; ".join(candidate.get("reasons") or []) or "compatible"
            lines.append(
                f"    {candidate['status'].upper()} {candidate['path']} "
                f"module={candidate['module']} python={candidate.get('python') or '?'}: {reasons}"
            )
            for warning in candidate.get("warnings") or []:
                lines.append(f"      warning: {warning}")

    import_check = diagnostics.get("import_check") or {}
    if import_check.get("requested"):
        if import_check.get("available"):
            lines.append(
                f"  Import check: PASS {import_check.get('module')} from {import_check.get('file')}"
            )
        else:
            lines.append(f"  Import check: FAIL {import_check.get('error', 'not available')}")

    smoke_test = diagnostics.get("smoke_test") or {}
    if smoke_test.get("requested"):
        if smoke_test.get("passed"):
            lines.append(f"  Smoke test: PASS {smoke_test.get('message')}")
        else:
            lines.append(f"  Smoke test: FAIL {smoke_test.get('error', 'not run')}")

    lines.append(f"  Summary: {diagnostics.get('summary', '')}")
    return "\n".join(lines)


def _iter_candidate_native_load_specs():
    diagnostics = get_native_backend_diagnostics(include_import_check=False)
    for candidate in diagnostics["accepted_candidates"]:
        yield Path(candidate["path"]), candidate["module"]


def _iter_candidate_native_files():
    """Backward-compatible helper yielding loadable native module file paths."""
    for native_file, _module_name in _iter_candidate_native_load_specs():
        yield native_file


def _iter_manifest_native_files(directory: Path):
    """Backward-compatible helper yielding manifest-selected native files for a directory."""
    runtime = _runtime_record()
    manifest_path, manifest = _read_manifest(directory)
    if not manifest_path or not manifest or manifest.get("_error"):
        return
    manifest_module = str(manifest.get("module") or NATIVE_MODULE_NAME)
    for entry in manifest.get("binaries", []):
        candidate = _candidate_from_manifest(entry, manifest_module, manifest_path)
        candidate = _evaluate_candidate(candidate, runtime)
        if candidate["status"] == "accepted":
            yield Path(candidate["path"])


def _load_native_module():
    attempt_errors = []
    for native_file, module_name in _iter_candidate_native_load_specs():
        try:
            return _load_native_file(native_file, module_name), None
        except (ImportError, OSError) as load_error:
            sys.modules.pop(module_name, None)
            sys.modules.pop(NATIVE_MODULE_NAME, None)
            attempt_errors.append(f"{native_file} (module={module_name}): {load_error}")

    for module_name in (NATIVE_MODULE_NAME, LEGACY_NATIVE_MODULE_NAME):
        try:
            native_module = importlib.import_module(module_name)
            return native_module, None
        except ImportError as import_error:
            attempt_errors.append(f"import {module_name}: {import_error}")

    diagnostics = get_native_backend_diagnostics(include_import_check=False)
    error_details = "; ".join(attempt_errors)
    return None, ImportError(
        f"Could not load a compatible native backend. {diagnostics.get('summary', '')} "
        f"Attempts: {error_details}"
    )


sim2real_native, _NATIVE_IMPORT_ERROR = _load_native_module()
_NATIVE_AVAILABLE = sim2real_native is not None
_SUPPRESS_STARTUP_LOGS = os.environ.get(NATIVE_SUPPRESS_STARTUP_LOGS_ENV_VAR) == "1"
if not _SUPPRESS_STARTUP_LOGS:
    if _NATIVE_AVAILABLE:
        print("[Sim2Real IMU] Native C++ backend loaded successfully.")
    else:
        _NATIVE_DIAGNOSTICS = get_native_backend_diagnostics(include_import_check=False)
        print(f"[Sim2Real IMU] WARNING: Could not load C++ backend: {_NATIVE_IMPORT_ERROR}")
        print(format_native_backend_diagnostics(_NATIVE_DIAGNOSTICS, verbose=False))
        print(
            "[Sim2Real IMU] IMU prims will be registered but sensor-realism effects "
            "will not be applied."
        )


class NativeNoiseBackend:
    """
    Wraps the C++ sim2real native pybind module.
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
        Run one sensor-realism step for this prim.
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
