#!/usr/bin/env python3
# ******************************************************************************
# File Name          : validate_release_zip.py
# Description        : Release ZIP validation for Sim2Real IMU packages.
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
from pathlib import PurePosixPath
import sys
import zipfile

EXPECTED_DEPENDENCIES = {
    "5.1.0": '"omni.isaac.sensor" = {}',
    "6.0.0": '"isaacsim.sensors.physics" = {}',
}

EXPECTED_PYTHON = {
    "5.1.0": "3.11",
    "6.0.0": "3.12",
}

DISALLOWED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
DISALLOWED_SUFFIXES = {".pyc", ".pyo"}
DISALLOWED_NAMES = {".DS_Store", "Thumbs.db"}


def _is_isolated_python():
    return bool(os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")) or (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    )


def _find_member(names, suffix):
    matches = [name for name in names if name.endswith(suffix)]
    return matches[0] if matches else None


def _has_disallowed_artifact(name):
    path = PurePosixPath(name)
    if path.name in DISALLOWED_NAMES:
        return True
    if path.suffix in DISALLOWED_SUFFIXES:
        return True
    return any(part in DISALLOWED_PARTS for part in path.parts)


def _has_pyinit(payload, module_name):
    return f"PyInit_{module_name}".encode("ascii") in payload


def validate_release_zip(zip_path, isaac_version):
    errors = []
    zip_path = str(zip_path)
    expected_dependency = EXPECTED_DEPENDENCIES[isaac_version]
    expected_python = EXPECTED_PYTHON[isaac_version]

    with zipfile.ZipFile(zip_path) as package:
        names = package.namelist()
        bad_artifacts = [name for name in names if _has_disallowed_artifact(name)]
        if bad_artifacts:
            errors.append(f"disallowed local artifacts present: {bad_artifacts[:10]}")

        config_name = _find_member(names, "/config/extension.toml")
        if not config_name:
            errors.append("missing config/extension.toml")
        else:
            config_text = package.read(config_name).decode("utf-8")
            if expected_dependency not in config_text:
                errors.append(
                    f"config/extension.toml missing dependency for Isaac {isaac_version}: "
                    f"{expected_dependency}"
                )

        manifest_name = _find_member(names, "/sim_binary/manifest.json")
        if not manifest_name:
            errors.append("missing sim_binary/manifest.json")
        else:
            manifest = json.loads(package.read(manifest_name).decode("utf-8"))
            entries = manifest.get("binaries", [])
            matching = [
                entry
                for entry in entries
                if str(entry.get("isaac_sim")) == isaac_version
                and str(entry.get("python")) == expected_python
            ]
            if not matching:
                errors.append(
                    f"manifest missing Isaac {isaac_version} / Python {expected_python} binary entry"
                )
            for entry in matching:
                member_suffix = f"/sim_binary/{entry.get('file')}"
                binary_member = _find_member(names, member_suffix)
                if not binary_member:
                    errors.append(f"manifest binary file missing from ZIP: {entry.get('file')}")
                    continue
                payload = package.read(binary_member)
                expected_sha1 = entry.get("sha1")
                if expected_sha1 and hashlib.sha1(payload).hexdigest() != expected_sha1:
                    errors.append(f"manifest sha1 mismatch for ZIP binary: {entry.get('file')}")
                module_name = entry.get("module") or manifest.get("module")
                if module_name and not _has_pyinit(payload, module_name):
                    errors.append(
                        f"ZIP binary {entry.get('file')} missing PyInit_{module_name}"
                    )

    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate a Sim2Real IMU release ZIP.")
    parser.add_argument("--isaac", choices=sorted(EXPECTED_DEPENDENCIES), required=True)
    parser.add_argument("zip_path")
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

    errors = validate_release_zip(args.zip_path, args.isaac)
    if errors:
        print("Release ZIP validation: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Release ZIP validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
