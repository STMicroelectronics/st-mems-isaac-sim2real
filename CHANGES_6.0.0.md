# Isaac Sim 6.0.0 Migration Changes

## Overview
This documents the changes made to port the `sim2real.imu.sensor` extension from Isaac Sim 5.1.0 to 6.0.0.
The original repository remains unchanged and continues to target Isaac Sim 5.1.0.

---

## `extension.toml`
- **Replaced** dependency `omni.isaac.sensor` with `isaacsim.sensors.experimental.physics`
  - In Isaac Sim 6.0.0, the entire `omni.isaac.*` namespace was rebranded to `isaacsim.*`
  - `isaacsim.sensors.physics` is deprecated in 6.0.0 in favor of `isaacsim.sensors.experimental.physics`

---

## `sim2real/imu/sensor/runtime.py`
- **Updated import** in `_initialize_truth_sensor()`:
  - `from omni.isaac.sensor import IMUSensor`
  - → `from isaacsim.sensors.experimental.physics import IMUSensor`
- **Updated frame keys** in `_read_truth_kinematics()`:
  - `raw["lin_acc"]` → `raw["linear_acceleration"]`
  - `raw["ang_vel"]` → `raw["angular_velocity"]`
  - Isaac Sim 6.0.0 renamed the keys returned by `IMUSensor.get_current_frame()`

---

## `sim2real/imu/sensor/verification_script.py`
- **Updated imports**:
  - `from omni.isaac.core.utils.types import ArticulationAction`
  - → `from isaacsim.core.utils.types import ArticulationAction`
  - `from omni.isaac.franka import Franka`
  - → `from isaacsim.robot.manipulators.examples.franka import Franka`
  - `from omni.isaac.sensor import IMUSensor`
  - → `from isaacsim.sensors.experimental.physics import IMUSensor`
- **Updated frame keys** in `NoisyImuSensor.get_current_frame()` and `log_sample()`:
  - `raw["lin_acc"]` → `raw["linear_acceleration"]`
  - `raw["ang_vel"]` → `raw["angular_velocity"]`
  - Note: internal keys (`lin_acc`, `ang_vel`) passed through the C++ noise backend remain unchanged

---

## Namespace Migration Reference
| Old (5.1.0) | New (6.0.0) |
|---|---|
| `omni.isaac.sensor` | `isaacsim.sensors.experimental.physics` |
| `omni.isaac.core.utils.types` | `isaacsim.core.utils.types` |
| `omni.isaac.franka` | `isaacsim.robot.manipulators.examples.franka` |
