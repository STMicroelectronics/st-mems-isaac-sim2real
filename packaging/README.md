# Release Packaging

This repository keeps one shared Python implementation and version-specific
extension manifests. Build and validate release assets from an isolated Python
environment: Conda, `venv`, or Isaac's bundled Python. Do not use system Python
for release validation unless CI explicitly passes `--allow-system-python`.

## Branch Strategy

Use branch names that encode compatibility-sensitive changes:

```text
hotfix/isaac-5.1-python311-diagnostics
compat/isaac-6.0-python312-ubuntu22-glibc234
release/v2.1.1
```

Rules:

- `main` is the production branch and should only receive reviewed, validated release or hotfix commits.
- `release/vX.Y.Z` is used for packaging, changelog date updates, and final release validation.
- `hotfix/...` is used for urgent customer-facing fixes.
- `compat/...` is used for Isaac/Python/Ubuntu/glibc/native-backend compatibility work.
- If a change touches Isaac version, Python ABI, Ubuntu, glibc, or native binaries, include that axis in the branch name.

## Manifest Selection

Before creating a release package, copy the matching manifest into
`config/extension.toml`:

```bash
cp packaging/isaac-5.1.0/extension.toml config/extension.toml
```

or:

```bash
cp packaging/isaac-6.0.0/extension.toml config/extension.toml
```

Release assets should be named by Isaac Sim version, for example:

```text
sim2real-imu-isaac-5.1.0-v2.1.1.zip
sim2real-imu-isaac-6.0.0-v2.1.1.zip
```

## Required Release Gates

Run these before creating a tag:

```bash
python packaging/validate_manifest.py
python packaging/validate_release_zip.py --isaac 5.1.0 path/to/sim2real-imu-isaac-5.1.0-v2.1.1.zip
python packaging/validate_release_zip.py --isaac 6.0.0 path/to/sim2real-imu-isaac-6.0.0-v2.1.1.zip
```

Then run the environment diagnostic inside each supported Isaac Sim runtime and
capture the output in the release record:

```bash
./python.sh -m sim2real.imu.sensor.diagnostics --verbose
```

A release must not be tagged unless the diagnostic reports `Overall: PASS` in:

- Isaac Sim 5.1.0 / Python 3.11 on Ubuntu 22.04.
- Isaac Sim 6.0.0 / Python 3.12 on Ubuntu 22.04.
