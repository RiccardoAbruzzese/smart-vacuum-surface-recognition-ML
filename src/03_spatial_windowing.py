"""
03_spatial_windowing.py
-----------------------
Partitions filtered continuous spatial signals into constant geometric windows 
of 250 mm. The space axis is re-zeroed at the start of each segment.
"""

from pathlib import Path
import numpy as np
import pandas as pd

WINDOW_LENGTH_MM = 250.0


def split_spatial_windows(input_dir: Path, output_dir: Path) -> None:
    """Segments spatial-domain signals into fixed 250 mm observation windows."""
    subdirectories = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirectories:
        print(f"[WARNING] No subdirectories found in: {input_dir}")
        return

    for sub_dir in subdirectories:
        csv_files = sorted(list(sub_dir.glob("*.csv")))
        if not csv_files:
            continue

        target_dir = output_dir / sub_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        total_windows_created = 0
        print(f"Windowing: {sub_dir.name.upper():<10} | Files: {len(csv_files)}")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, sep=",")
                space_values = df["space_mm"].values.astype(float)

                dx = np.mean(np.diff(space_values))
                if dx <= 0:
                    continue

                samples_per_window = int(np.round(WINDOW_LENGTH_MM / dx))
                num_windows = len(df) // samples_per_window

                for i in range(num_windows):
                    idx_start = i * samples_per_window
                    idx_end = idx_start + samples_per_window

                    df_window = df.iloc[idx_start:idx_end].copy()

                    # Reset spatial axis to start from 0.0 mm
                    df_window["space_mm"] = (
                        df_window["space_mm"] - df_window["space_mm"].iloc[0]
                    ).round(4)

                    output_name = target_dir / f"{file_path.stem}_win_{i+1:02d}.csv"
                    df_window.to_csv(output_name, index=False)
                    total_windows_created += 1

            except Exception as e:
                print(f"  [ERROR] Windowing failed for {file_path.name}: {e}")

        print(f"  -> Generated {total_windows_created} windows of {WINDOW_LENGTH_MM} mm.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_filtered_dir = project_root / "data" / "interim" / "02_spatial_filtered"
    output_windowed_dir = project_root / "data" / "interim" / "03_spatial_windows_250mm"

    print("=== STEP 03: CONSTANT GEOMETRIC WINDOWING (250 mm) ===")
    split_spatial_windows(input_filtered_dir, output_windowed_dir)
    print("=== COMPLETED ===\n")