"""
02_spatial_filtering.py
-----------------------
Converts time-series data into the spatial domain based on kinematic scan velocity,
applies spatial band-pass filtering (0.006 - 0.450 cycles/mm), and dynamically trims
transients (sensor touchdown and probe liftoff).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Linear scan velocity for all 5 surface classes [m/s]
SCAN_VELOCITIES = {
    "asphalt": 0.20768,  # Stone / Asphalt class
    "carpet": 0.24919,   # Carpet class
    "marble": 0.48013,   # Smooth / Marble / Glass class
    "parquet": 0.32362,  # Parquet / Wood class
    "tile": 0.39200,     # Ceramic Tile class
}

# Dynamic windowing & spatial filtering parameters
ACTIVATION_THRESHOLD_G = 0.07
START_OFFSET_MM = 33.0
END_OFFSET_MM = 33.0
MIN_VALID_LENGTH_MM = 150.0

LOWCUT_SPATIAL_CYCLES_MM = 0.006   # Removes macro-geometric drift (lambda ~ 166.7 mm)
HIGHCUT_SPATIAL_CYCLES_MM = 0.450  # Isolates textures down to lambda ~ 2.22 mm (probe tip limit)


def filter_and_trim_spatial(input_dir: Path, output_dir: Path, scale_acc: bool = False) -> None:
    """Processes surface runs: temporal-to-spatial conversion, band-pass filtering,
    and dynamic transient trimming.
    """
    subdirectories = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirectories:
        print(f"[WARNING] No subdirectories found in: {input_dir}")
        return

    for sub_dir in subdirectories:
        material_name = sub_dir.name.lower()
        if material_name not in SCAN_VELOCITIES:
            print(f"[WARNING] Velocity not defined for '{material_name}'. Skipping.")
            continue

        velocity_m_s = SCAN_VELOCITIES[material_name]
        target_dir = output_dir / material_name
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(list(sub_dir.glob("*.csv")))
        if not csv_files:
            continue

        print(f"Processing: {material_name.upper():<10} | Velocity: {velocity_m_s:.5f} m/s | Files: {len(csv_files)}")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, sep=",")
                time_s = df.iloc[:, 0].values.astype(float)
                acc_x_raw = df.iloc[:, 1].values.astype(float)
                acc_y_raw = df.iloc[:, 2].values.astype(float)
                acc_z_raw = df.iloc[:, 3].values.astype(float)

                # Kinematic conversion: time (s) -> space (mm)
                space_mm = time_s * velocity_m_s * 1000.0

                # Spatial sampling parameters
                dx = np.mean(np.diff(space_mm))
                if dx <= 0:
                    print(f"  [ERROR] Invalid spatial step (dx <= 0) in {file_path.name}.")
                    continue
                fs_spatial = 1.0 / dx

                # Baseline zero-mean removal
                if scale_acc:
                    norm_factor = (velocity_m_s / 0.3) ** 2
                    acc_x = (acc_x_raw / norm_factor) - np.mean(acc_x_raw / norm_factor)
                    acc_y = (acc_y_raw / norm_factor) - np.mean(acc_y_raw / norm_factor)
                    acc_z = (acc_z_raw / norm_factor) - np.mean(acc_z_raw / norm_factor)
                else:
                    acc_x = acc_x_raw - np.mean(acc_x_raw)
                    acc_y = acc_y_raw - np.mean(acc_y_raw)
                    acc_z = acc_z_raw - np.mean(acc_z_raw)

                # Spatial 2nd-order Butterworth band-pass filter
                nyq_spatial = fs_spatial / 2.0
                highcut_corr = min(HIGHCUT_SPATIAL_CYCLES_MM, nyq_spatial - 0.001)
                lowcut_corr = max(0.001, LOWCUT_SPATIAL_CYCLES_MM)

                b, a = butter(
                    2,
                    [lowcut_corr / nyq_spatial, highcut_corr / nyq_spatial],
                    btype="bandpass",
                    analog=False,
                )
                filtered_x = filtfilt(b, a, acc_x)
                filtered_y = filtfilt(b, a, acc_y)
                filtered_z = filtfilt(b, a, acc_z)

                # Dynamic transient trimming
                offset_start_idx = int(np.round(START_OFFSET_MM / dx))
                offset_end_idx = int(np.round(END_OFFSET_MM / dx))

                idx_start = 0
                idx_end = len(filtered_z) - 1
                triggered = False

                for i in range(len(filtered_z)):
                    if np.abs(filtered_z[i]) > ACTIVATION_THRESHOLD_G:
                        idx_start = i + offset_start_idx
                        triggered = True
                        break

                if triggered:
                    for i in range(len(filtered_z) - 1, idx_start, -1):
                        if np.abs(filtered_z[i]) > ACTIVATION_THRESHOLD_G:
                            idx_end = i + offset_end_idx
                            break

                idx_start = max(0, idx_start)
                idx_end = min(len(space_mm) - 1, idx_end)

                if idx_start >= idx_end:
                    continue

                effective_length_mm = space_mm[idx_end] - space_mm[idx_start]
                if not triggered or effective_length_mm < MIN_VALID_LENGTH_MM:
                    continue

                trimmed_space = space_mm[idx_start:idx_end] - space_mm[idx_start]
                trimmed_x = filtered_x[idx_start:idx_end]
                trimmed_y = filtered_y[idx_start:idx_end]
                trimmed_z = filtered_z[idx_start:idx_end]

                df_clean = pd.DataFrame(
                    {
                        "space_mm": np.round(trimmed_space, 4),
                        "ax": np.round(trimmed_x, 5),
                        "ay": np.round(trimmed_y, 5),
                        "az": np.round(trimmed_z, 5),
                    }
                )

                output_file = target_dir / file_path.name
                df_clean.to_csv(output_file, index=False)

            except Exception as e:
                print(f"  [ERROR] Processing failed on {file_path.name}: {e}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_standardized_dir = project_root / "data" / "interim" / "01_time_standardized"
    output_filtered_dir = project_root / "data" / "interim" / "02_spatial_filtered"

    print("=== STEP 02: SPATIAL TRANSFORMATION & FILTERING ===")
    filter_and_trim_spatial(input_standardized_dir, output_filtered_dir, scale_acc=False)
    print("=== COMPLETED ===\n")