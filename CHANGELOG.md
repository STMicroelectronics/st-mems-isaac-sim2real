# Changelog

All notable changes to this project are documented in this file.

## [v2.1.0] - Unreleased
- Added Isaac Sim adapter layer so Isaac API changes stay isolated from the Sim2Real runtime.
- Added Isaac Sim 5.1.0 and 6.0.0 extension manifest templates under `packaging/`.
- Added native backend manifest metadata under `sim_binary/`.
- Added Ubuntu 22.04-compatible Python 3.10 native backend for Isaac Sim 5.1.0.
- Updated verification script to use the shared Isaac adapter boundary.
- Restored dual-version support policy: Isaac Sim 6.0.0 active support, Isaac Sim 5.1.0 maintenance/customer support.

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
