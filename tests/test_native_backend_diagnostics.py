# ******************************************************************************
# File Name          : test_native_backend_diagnostics.py
# Description        : Unit tests for Sim2Real native backend diagnostics.
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
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("SIM2REAL_SUPPRESS_NATIVE_BACKEND_STARTUP_LOGS", "1")

from sim2real.imu.sensor.noise import native_backend


class NativeBackendDiagnosticsTests(unittest.TestCase):
    def _write_manifest_fixture(self, root: Path, *, use_deprecated_glibc=False):
        binary = root / "sim2real_native.cpython-311-x86_64-linux-gnu.so"
        binary.write_bytes(b"ELF\0PyInit_sim2real_native\0Sim2RealCore\0")
        glibc_field = "max_glibc" if use_deprecated_glibc else "min_glibc"
        manifest = {
            "schema_version": 2,
            "module": "sim2real_native_v0_1",
            "binaries": [
                {
                    "file": binary.name,
                    "module": "sim2real_native",
                    "python": "3.11",
                    "isaac_sim": "5.1.0",
                    "os": "linux",
                    "arch": "x86_64",
                    glibc_field: "2.34",
                    "sha1": hashlib.sha1(binary.read_bytes()).hexdigest(),
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return binary

    def _patch_runtime(self, root: Path, python_version="3.11"):
        return mock.patch.multiple(
            native_backend,
            _runtime_python_version=mock.Mock(return_value=python_version),
            _runtime_architecture=mock.Mock(return_value="x86_64"),
            _runtime_os=mock.Mock(return_value="linux"),
            _runtime_glibc_version=mock.Mock(return_value="2.35"),
            _iter_search_locations=mock.Mock(
                return_value=iter(
                    [
                        {
                            "source": "test",
                            "path": root,
                            "kind": "directory",
                            "exists": True,
                        }
                    ]
                )
            ),
        )

    def test_manifest_selects_python_311_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_manifest_fixture(root)
            with self._patch_runtime(root, "3.11"):
                diagnostics = native_backend.get_native_backend_diagnostics()
        self.assertTrue(diagnostics["compatible"])
        self.assertEqual(len(diagnostics["accepted_candidates"]), 1)
        self.assertEqual(diagnostics["accepted_candidates"][0]["module"], "sim2real_native")

    def test_missing_abi_reports_available_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_manifest_fixture(root)
            with self._patch_runtime(root, "3.12"):
                diagnostics = native_backend.get_native_backend_diagnostics()
        self.assertFalse(diagnostics["compatible"])
        self.assertIn("3.11", diagnostics["available_python_abis"])
        self.assertIn("Python ABI mismatch", diagnostics["candidates"][0]["reasons"][0])

    def test_direct_file_path_override_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = self._write_manifest_fixture(root)
            with mock.patch.multiple(
                native_backend,
                _runtime_python_version=mock.Mock(return_value="3.11"),
                _runtime_architecture=mock.Mock(return_value="x86_64"),
                _runtime_os=mock.Mock(return_value="linux"),
                _runtime_glibc_version=mock.Mock(return_value="2.35"),
                _iter_search_locations=mock.Mock(
                    return_value=iter(
                        [
                            {
                                "source": "SIM2REAL_NATIVE_PATH",
                                "path": binary,
                                "kind": "file",
                                "exists": True,
                            }
                        ]
                    )
                ),
            ):
                diagnostics = native_backend.get_native_backend_diagnostics()
        self.assertTrue(diagnostics["compatible"])
        self.assertEqual(diagnostics["accepted_candidates"][0]["path"], str(binary))

    def test_deprecated_max_glibc_warns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_manifest_fixture(root, use_deprecated_glibc=True)
            with self._patch_runtime(root, "3.11"):
                diagnostics = native_backend.get_native_backend_diagnostics()
        self.assertTrue(diagnostics["compatible"])
        self.assertIn("max_glibc", diagnostics["accepted_candidates"][0]["warnings"][0])


if __name__ == "__main__":
    unittest.main()
