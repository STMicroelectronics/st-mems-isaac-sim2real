# Compatibility Branching Strategy

This repository ships Isaac Sim extension code plus Python ABI-specific native
backends. Compatibility changes must be isolated in branches whose names make the
compatibility axis explicit.

## Branch Types

```text
main
release/vX.Y.Z
hotfix/<customer-or-risk-area>
compat/<isaac-python-os-glibc-axis>
feature/<short-feature-name>
```

Examples:

```text
hotfix/isaac-5.1-python311-diagnostics
compat/isaac-6.0-python312-ubuntu22-glibc234
release/v2.1.1
feature/add-sensor-model-lsm6xyz
```

## Rules

- Do not mix Isaac 5.1 and Isaac 6.0 compatibility changes with unrelated feature work.
- Branch names for native backend changes must include Python ABI when relevant, for example `python311` or `python312`.
- Branch names for Linux native backend changes must include Ubuntu/glibc when relevant, for example `ubuntu22-glibc234`.
- Release branches are for final changelog, package validation, release ZIP creation, and tag preparation only.
- Customer hotfix branches must include the risk area and must be merged only after diagnostic and package validation pass.

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
