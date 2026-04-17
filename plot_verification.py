# ******************************************************************************
# File Name          : plot_verification.py
# Description        : Plotting utility for comparing clean and noisy IMU
#                      verification trajectories.
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

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIG ---
DATA_DIR = os.path.expanduser("~/Documents/trajectories_verification")

def plot_verification(traj_path):
    # Prefer the new naming convention and fall back to legacy names.
    clean_files = glob.glob(os.path.join(traj_path, "clean_imu_*.csv"))
    noisy_files = glob.glob(os.path.join(traj_path, "noisy_imu_*.csv"))
    if not noisy_files:
        noisy_files = glob.glob(os.path.join(traj_path, "gui_imu_*.csv"))

    if not clean_files or not noisy_files:
        print(f"Skipping {traj_path}: Missing CSV files.")
        print("  Expected: clean_imu_*.csv and noisy_imu_*.csv (or legacy gui_imu_*.csv)")
        return

    # Load Data
    df_clean = pd.read_csv(clean_files[0])
    df_noisy = pd.read_csv(noisy_files[0])

    # Normalize Time (start at 0)
    t_clean = df_clean['time'] - df_clean['time'].iloc[0]
    t_noisy = df_noisy['time'] - df_noisy['time'].iloc[0]

    # Setup Figure (3x2 Grid)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Verification: {os.path.basename(traj_path)}", fontsize=16)

    plots = [
        ("Accel X", "imu_ax", "m/s^2", (0, 0)),
        ("Accel Y", "imu_ay", "m/s^2", (1, 0)),
        ("Accel Z", "imu_az", "m/s^2", (2, 0)),
        ("Gyro X",  "imu_gx", "rad/s", (0, 1)),
        ("Gyro Y",  "imu_gy", "rad/s", (1, 1)),
        ("Gyro Z",  "imu_gz", "rad/s", (2, 1)),
    ]

    for title, col, unit, (row, col_idx) in plots:
        ax = axes[row, col_idx]

        # Plot noisy trace first so clean trace stays visible on top.
        ax.plot(
            t_noisy,
            df_noisy[col],
            "r-",
            alpha=0.5,
            linewidth=1.0,
            label="Noisy Sensor (ASM330LHH)",
        )

        # Plot Clean on top
        ax.plot(t_clean, df_clean[col], 'k--', alpha=0.8, linewidth=1.5, label='Clean (Isaac)')

        ax.set_ylabel(f"{title} [{unit}]")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc='upper right', frameon=True)

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(traj_path, "verification_plot.png")
    plt.savefig(out_path)
    print(f"Generated plot: {out_path}")
    plt.close()

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory not found: {DATA_DIR}")
        return

    traj_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "traj_*")))

    if not traj_dirs:
        print("No trajectory folders found!")
        return

    print(f"Found {len(traj_dirs)} trajectories. Generating plots...")
    for traj in traj_dirs:
        plot_verification(traj)

if __name__ == "__main__":
    main()
