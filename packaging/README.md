# Release Packaging

This repository keeps one shared Python implementation and version-specific
extension manifests.

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
sim2real-imu-isaac-5.1.0-v2.1.0.zip
sim2real-imu-isaac-6.0.0-v2.1.0.zip
```

