import csv
import math
import os
import random

import numpy as np
import omni.timeline
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.sensor import IMUSensor
from omni.kit.app import get_app

from sim2real.imu.sensor.noise.native_backend import NativeNoiseBackend


class NoisyImuSensor:
    def __init__(self, prim_path, name="imu", seed=123, config=None):
        self._sensor = IMUSensor(prim_path=prim_path, name=name)
        self._timeline = omni.timeline.get_timeline_interface()
        self._prim_path = prim_path
        self._noise_backend = NativeNoiseBackend()

        if config:
            self._noise_backend.register_sensor(prim_path, config, seed=seed)
            print(f"C++ Config Applied via NativeNoiseBackend: {config}")
        else:
            self._noise_backend.register_sensor(prim_path, {}, seed=seed)

    def initialize(self, physics_sim_view=None):
        self._sensor.initialize(physics_sim_view)

    def get_current_frame(self, read_gravity=True):
        raw = self._sensor.get_current_frame(read_gravity=read_gravity)
        current_time = self._timeline.get_current_time()

        result = self._noise_backend.step_sensor(
            self._prim_path,
            sim_time=current_time,
            truth_kinematics={
                "lin_acc": raw["lin_acc"],
                "ang_vel": raw["ang_vel"],
            },
        )

        if result is not None:
            return {"lin_acc": result["lin_acc"], "ang_vel": result["ang_vel"]}
        return {"lin_acc": raw["lin_acc"], "ang_vel": raw["ang_vel"]}

    def __getattr__(self, name):
        return getattr(self._sensor, name)


# Backward-compatible wrapper class name used in older docs/scripts.
Sim2RealIMUSensor = NoisyImuSensor


ROBOT_PATH = "/World/franka"

# Both sensors read from the same native Isaac physics IMU prim.
CLEAN_IMU_PATH = "/World/franka/panda_hand/Imu_Sensor"
NOISY_IMU_PATH = "/World/franka/panda_hand/Imu_Sensor"

OUTPUT_DIR = os.path.expanduser("~/Documents/trajectories_verification")

# Simulation Params
STARTING_TRAJECTORY_INDEX = 0
NUM_TRAJECTORIES = 5
TRAJECTORY_DURATION_S = 4.0
SETTLE_DURATION_S = 1.0
RESET_DURATION_S = 2.0
STARTUP_FRAMES = 60

HOME = np.array(
    [0.000167, -0.786, 4.01e-5, -2.35502, 9.29e-5, 1.571, 0.786, 0.04, 0.04]
)

# Noise config for ASM330LHH profile
NOISE_CONFIG = {
    "accel_fs_g": 8.0,
    "gyro_fs_dps": 2000.0,
    "odr_hz": 104.0,
    "vibration": True,
}


# Runtime state
robot = None
imu_clean = None
imu_noisy = None
timeline = omni.timeline.get_timeline_interface()

frame = 0
phase = "startup"
trajectory_index = STARTING_TRAJECTORY_INDEX
time_in_phase = 0.0
targets = []

# Log handles
file_clean = None
writer_clean = None
file_noisy = None
writer_noisy = None

CSV_HEADER = [
    "time",
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "j7",
    "gripper1",
    "gripper2",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
]


def smooth_interp(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def gen_target():
    target = HOME.copy()
    max_offset = math.pi / 5
    for joint_index in range(7):
        target[joint_index] = HOME[joint_index] + random.uniform(-max_offset, max_offset)
    return target


def start_log():
    global file_clean, writer_clean, file_noisy, writer_noisy

    trajectory_folder = os.path.join(OUTPUT_DIR, f"traj_{trajectory_index}")
    os.makedirs(trajectory_folder, exist_ok=True)

    clean_csv_path = os.path.join(trajectory_folder, f"clean_imu_{trajectory_index}.csv")
    file_clean = open(clean_csv_path, "w", newline="")
    writer_clean = csv.writer(file_clean)
    writer_clean.writerow(CSV_HEADER)

    noisy_csv_path = os.path.join(trajectory_folder, f"noisy_imu_{trajectory_index}.csv")
    file_noisy = open(noisy_csv_path, "w", newline="")
    writer_noisy = csv.writer(file_noisy)
    writer_noisy.writerow(CSV_HEADER)


def log_sample(joint_positions):
    global writer_clean, writer_noisy
    if not (imu_clean and imu_noisy and writer_clean and writer_noisy):
        return

    current_time = timeline.get_current_time()

    clean_frame = imu_clean.get_current_frame(read_gravity=True)
    clean_acc = clean_frame["lin_acc"]
    clean_vel = clean_frame["ang_vel"]

    noisy_frame = imu_noisy.get_current_frame(read_gravity=True)
    noisy_acc = noisy_frame["lin_acc"]
    noisy_vel = noisy_frame["ang_vel"]

    writer_clean.writerow(
        [current_time]
        + list(joint_positions[:9])
        + [clean_acc[2], -1 * clean_acc[1], clean_acc[0], clean_vel[2], -1 * clean_vel[1], clean_vel[0]]
    )

    writer_noisy.writerow(
        [current_time]
        + list(joint_positions[:9])
        + [noisy_acc[2], -1 * noisy_acc[1], noisy_acc[0], noisy_vel[2], -1 * noisy_vel[1], noisy_vel[0]]
    )

    file_clean.flush()
    file_noisy.flush()


def close_log():
    global file_clean, writer_clean, file_noisy, writer_noisy
    if file_clean:
        file_clean.close()
    if file_noisy:
        file_noisy.close()
    file_clean = None
    writer_clean = None
    file_noisy = None
    writer_noisy = None


def update(event):
    global robot, imu_clean, imu_noisy, frame, phase, trajectory_index, time_in_phase, targets

    if not timeline.is_playing():
        return

    frame += 1
    dt = event.payload["dt"]

    if robot is None:
        robot = Franka(prim_path=ROBOT_PATH, name="franka")
        robot.initialize()
        robot.set_joints_default_state(positions=HOME, velocities=np.zeros(9))
        robot.post_reset()

        imu_clean = IMUSensor(prim_path=CLEAN_IMU_PATH, name="imu_clean")
        imu_clean.initialize()

        imu_noisy = NoisyImuSensor(
            prim_path=NOISY_IMU_PATH,
            name="imu_noisy",
            seed=123,
            config=NOISE_CONFIG,
        )
        imu_noisy.initialize()

        targets = [gen_target() for _ in range(NUM_TRAJECTORIES)]
        print(f"VERIFICATION MODE: Logging to {OUTPUT_DIR}")
        print(f"  Clean sensor: {CLEAN_IMU_PATH}")
        print(f"  Noisy sensor: {NOISY_IMU_PATH} (+ C++ noise)")
        return

    if phase == "startup":
        if frame > STARTUP_FRAMES:
            phase = "moving"
            time_in_phase = 0.0
            start_log()
            print(f"Trajectory {trajectory_index}")
        return

    time_in_phase += dt
    current_positions = robot.get_joint_positions()

    if phase == "moving":
        progress = min(time_in_phase / TRAJECTORY_DURATION_S, 1.0)
        blend = smooth_interp(progress)
        target_pos = HOME + blend * (targets[trajectory_index - STARTING_TRAJECTORY_INDEX] - HOME)
        robot.apply_action(ArticulationAction(joint_positions=target_pos))
        log_sample(current_positions)
        if progress >= 1.0:
            phase = "settling"
            time_in_phase = 0.0

    elif phase == "settling":
        log_sample(current_positions)
        if time_in_phase >= SETTLE_DURATION_S:
            close_log()
            trajectory_index += 1
            if trajectory_index >= STARTING_TRAJECTORY_INDEX + NUM_TRAJECTORIES:
                print("Done: Verification Data Collected")
                timeline.stop()
                return
            targets.append(gen_target())
            phase = "resetting"
            time_in_phase = 0.0

    elif phase == "resetting":
        progress = min(time_in_phase / RESET_DURATION_S, 1.0)
        blend = smooth_interp(progress)
        reset_pos = current_positions + blend * (HOME - current_positions)
        robot.apply_action(ArticulationAction(joint_positions=reset_pos))
        if progress >= 1.0:
            robot.set_joint_positions(HOME)
            robot.set_joint_velocities(np.zeros(9))
            phase = "moving"
            time_in_phase = 0.0
            start_log()
            print(f"Trajectory {trajectory_index}")


stream = get_app().get_update_event_stream()
sub = stream.create_subscription_to_pop(update)
print(f"Collecting {NUM_TRAJECTORIES} trajectories -> {OUTPUT_DIR}")
print("Press PLAY to begin.")
