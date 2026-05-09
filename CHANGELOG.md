# Changelog

All notable changes to this project are documented in this file.

## [v2.1.1] - Unreleased
- Corrected Isaac Sim 5.1.0 support to Python 3.11 and added the canonical `sim2real_native_v0_1` native backend.
- Added release-grade native backend diagnostics with Python ABI, glibc, search path, manifest, and import/smoke-test reporting.
- Added release validation scripts for native manifest and Isaac-versioned ZIP packages.
- Removed the hard dependency on a pre-named `Imu_Sensor` child by resolving `sim2real:truthImuPrimPath`, discovering IMU-like descendants, and auto-creating a default Isaac IMU fallback when needed.
- Deferred truth IMU binding until physics is ready and added automatic retry so sensors added before the physics scene no longer require delete/re-add workarounds.
- Removed per-tick IMU debug-vector writes from USD custom data so Isaac 6 no longer throws non-fatal metadata type errors during simulation.
- Replaced deprecated `max_glibc` manifest metadata with `min_glibc`.
- Documented compatibility-focused branch naming and isolated Python environment requirements.
- Clarified that the current `main` support matrix differs from the historical `v2.1.0` Python 3.10 note for Isaac Sim 5.1.0.
- Repositioned `main` as the Isaac Sim 5.1.0 stable production baseline and isolated Isaac Sim 6.0.0 Early Developer Release work into a dedicated experimental branch policy.
- Documented the verification plotting dependency installation path for clean Isaac pip/conda environments.

## [v2.1.0] - 2026-04-21
- Added Isaac Sim adapter layer so Isaac API changes stay isolated from the Sim2Real runtime.
- Added Isaac Sim 5.1.0 and 6.0.0 extension manifest templates under `packaging/`.
- Added native backend manifest metadata under `sim_binary/`.
- Historical tag asset state for Isaac Sim 5.1.0 used the Ubuntu 22.04-compatible Python 3.10 native backend.
- Updated verification script to use the shared Isaac adapter boundary.
- Updated public-facing terminology from noisy IMU to realistic Sim2Real IMU.
- Published dual tagged assets at that point in time: Isaac Sim 6.0.0 / Python 3.12 and Isaac Sim 5.1.0 / Python 3.10.
- Validated end-to-end in Isaac Sim 5.1.0 and Isaac Sim 6.0.0.
- This historical tag has been superseded by the current stable support line, which uses Isaac Sim 5.1.0 / Python 3.11.

## [v2.0.0] - 2026-04-16
- Breaking release: standardized production support on Isaac Sim 6.0.0 with Python 3.12.
- Dropped Isaac Sim 5.1.0 / Python 3.10 compatibility from the main production branch.
- Kept only the bundled Python 3.12 native backend artifact in `sim_binary/`.
- Removed GLIBC-incompatible Python 3.10-style native backend artifacts from the production package.
- Hardened native backend loading so incompatible override binaries do not abort extension startup.

## [v1.0.2] - 2026-03-06
- Added `requirements.txt` for verification/plotting dependencies.
- Added `LICENSE.md` to repository root.
- Added ST copyright headers to Python source files.
- Updated extension package version in `config/extension.toml` to `1.0.2`.

## [v1.0.1] - 2026-03-05
- Updated Isaac Sim version references to `5.1.0` in documentation.

## [v1.0.0] - 2026-03-05
- Initial public release of the ST IMU Isaac Sim extension.
- Added compliance files (`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`).
