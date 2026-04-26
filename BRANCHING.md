# Compatibility Branching Strategy

This repository ships Isaac Sim extension code plus Python ABI-specific native
backends. Compatibility changes must be isolated in branches whose names make the
compatibility axis explicit.

## Branch Types

```text
main
support/isaac-sim-5.1.0-python-3.11-stable
experimental/isaac-sim-6.0.0-early-developer-release-python-3.12
release/vX.Y.Z
hotfix/<isaac-sim-and-risk-area>
feature/<short-feature-name>
```

Examples:

```text
support/isaac-sim-5.1.0-python-3.11-stable
experimental/isaac-sim-6.0.0-early-developer-release-python-3.12
hotfix/isaac-sim-5.1.0-python-3.11-native-backend
release/v2.1.1
feature/add-sensor-model-lsm6xyz
```

## Branch Meanings

- `main`: customer-facing production branch. It must follow the most stable
  NVIDIA Isaac Sim release that ST has validated end-to-end for customers.
  Today that baseline is Isaac Sim `5.1.0` with Python `3.11`.
- `support/isaac-sim-5.1.0-python-3.11-stable`: long-lived stable support line
  for the current production baseline. Keep this branch fast-forwardable from
  `main` while `main` remains on the same baseline.
- `experimental/isaac-sim-6.0.0-early-developer-release-python-3.12`: isolated
  branch for Isaac Sim `6.0.0` Early Developer Release work. This branch exists
  because upstream marks 6.0.0 as an Early Developer Release rather than a
  normal stable release.

## Rules

- Do not mix Isaac Sim `5.1.0` stable support work with Isaac Sim `6.0.0` Early
  Developer Release exploration.
- Branch names for native backend changes must include the Python ABI when
  relevant, for example `python-3.11` or `python-3.12`.
- Branch names for Linux native backend changes must include Ubuntu/glibc when
  relevant, for example `ubuntu-22.04-glibc-2.34`.
- Release branches are for final changelog, package validation, release ZIP creation, and tag preparation only.
- Customer hotfix branches must include the risk area and must be merged only after diagnostic and package validation pass.
- `main` must not be retargeted to Isaac Sim `6.0.0` until NVIDIA ships a
  normal stable/GA release and the full validation matrix passes.

## Environment Policy

Use an isolated Python environment for development and release tooling:

- Isaac's bundled Python when validating the extension in Isaac Sim.
- Conda or `venv` when running packaging, plotting, or manifest checks outside Isaac.
- Do not use system Python for release validation unless CI explicitly opts in with `--allow-system-python`.

Before tagging a release, capture diagnostic output from every supported Isaac
runtime using:

```bash
./python.sh -m sim2real.imu.sensor.diagnostics --verbose
```
