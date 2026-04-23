#!/usr/bin/env python3
# ******************************************************************************
# File Name          : validate_manifest.py
# Description        : Release gate for Sim2Real native backend manifest files.
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
import hashlib
import json
import os
from pathlib import Path
import sys

REQUIRED_RUNTIMES = {
    "5.1.0": "3.11",
    "6.0.0": "3.12",
}


def _is_isolated_python():
    return bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")) or (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    )


def _sha1(path):
    digest = hashlib.sha1()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_pyinit(path, module_name):
    return f"PyInit_{module_name}".encode("ascii") in path.read_bytes()


def validate_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    if manifest.get("schema_version") not in (1, 2):
        errors.append("schema_version must be 1 or 2")

    default_module = manifest.get("module")
    if not default_module:
        errors.append("manifest missing top-level module")

    entries = manifest.get("binaries", [])
    if not isinstance(entries, list) or not entries:
        errors.append("manifest binaries must be a non-empty list")

    by_runtime = {}
    for index, entry in enumerate(entries):
        label = f"binaries[{index}]"
        filename = entry.get("file")
        module_name = entry.get("module") or default_module
        python_version = str(entry.get("python") or "")
        isaac_version = str(entry.get("isaac_sim") or "")
        path = root / str(filename or "")

        if not filename:
            errors.append(f"{label}: missing file")
            continue
        if not module_name:
            errors.append(f"{label}: missing module")
        if not python_version:
            errors.append(f"{label}: missing python")
        if not isaac_version:
            errors.append(f"{label}: missing isaac_sim")
        if "max_glibc" in entry:
            errors.append(f"{label}: max_glibc is deprecated; use min_glibc")
        if not entry.get("min_glibc"):
            errors.append(f"{label}: missing min_glibc")
        if not path.is_file():
            errors.append(f"{label}: file not found: {path}")
            continue
        if entry.get("sha1") and _sha1(path) != entry["sha1"]:
            errors.append(f"{label}: sha1 mismatch for {path.name}")
        if module_name and not _has_pyinit(path, module_name):
            errors.append(f"{label}: {path.name} missing PyInit_{module_name}")

        by_runtime[(isaac_version, python_version)] = entry

    for isaac_version, python_version in REQUIRED_RUNTIMES.items():
        if (isaac_version, python_version) not in by_runtime:
            errors.append(
                f"required runtime missing: Isaac Sim {isaac_version} / Python {python_version}"
            )

    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate sim_binary/manifest.json for release.")
    parser.add_argument("manifest", nargs="?", default="sim_binary/manifest.json")
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="Allow running outside Conda/venv for CI or emergency validation.",
    )
    args = parser.parse_args(argv)

    if not args.allow_system_python and not _is_isolated_python():
        print(
            "FAIL: run release validation from Conda/venv/Isaac Python, or pass --allow-system-python.",
            file=sys.stderr,
        )
        return 2

    errors = validate_manifest(args.manifest)
    if errors:
        print("Manifest validation: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Manifest validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
