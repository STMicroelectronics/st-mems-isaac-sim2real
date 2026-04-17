# Changelog
All notable changes to this project are documented in this file.

## [v1.1.0] - 2026-04-16

### Isaac Sim 6.0.0 Compatibility

- **`config/extension.toml`**: replaced dependency `omni.isaac.sensor` with `isaacsim.sensors.experimental.physics`
  - In Isaac Sim 6.0.0, the entire `omni.isaac.*` namespace was rebranded to `isaacsim.*`
  - `isaacsim.sensors.physics` is deprecated in 6.0.0 in favor of `isaacsim.sensors.experimental.physics`

- **`sim2real/imu/sensor/runtime.py`**: updated import and frame keys for Isaac Sim 6.0.0
  - `from omni.isaac.sensor import IMUSensor` → `from isaacsim.sensors.experimental.physics import IMUSensor`
  - `raw["lin_acc"]` → `raw["linear_acceleration"]`
  - `raw["ang_vel"]` → `raw["angular_velocity"]`
  - Isaac Sim 6.0.0 renamed the keys returned by `IMUSensor.get_current_frame()`

- **`sim2real/imu/sensor/verification_script.py`**: updated imports and frame keys for Isaac Sim 6.0.0
  - `from omni.isaac.core.utils.types import ArticulationAction` → `from isaacsim.core.utils.types import ArticulationAction`
  - `from omni.isaac.franka import Franka` → `from isaacsim.robot.manipulators.examples.franka import Franka`
  - `from omni.isaac.sensor import IMUSensor` → `from isaacsim.sensors.experimental.physics import IMUSensor`
  - Frame keys updated in `NoisyImuSensor.get_current_frame()` and `log_sample()`
  - Note: internal keys (`lin_acc`, `ang_vel`) passed through the C++ noise backend remain unchanged

### Namespace Migration Reference

| Old (5.1.0) | New (6.0.0) |
|---|---|
| `omni.isaac.sensor` | `isaacsim.sensors.experimental.physics` |
| `omni.isaac.core.utils.types` | `isaacsim.core.utils.types` |
| `omni.isaac.franka` | `isaacsim.robot.manipulators.examples.franka` |

---

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
