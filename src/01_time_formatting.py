"""
01_time_formatting.py
---------------------
Pre-processing step: converts raw millisecond timestamps from ESP8266 
serial logging into seconds and standardizes column headers across all surface runs.
"""

from pathlib import Path
import pandas as pd


def process_raw_time(input_dir: Path, output_dir: Path) -> None:
    """Reads raw sensor CSV files, scales the time column to seconds,

    and saves standardized copies preserving the original directory structure.
    """
    subdirectories = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirectories:
        print(f"[WARNING] No subdirectories found in: {input_dir}")
        return

    print(
        f"Found {len(subdirectories)} surface classes in '{input_dir.name}'.\n"
    )

    for sub_dir in subdirectories:
        material_name = sub_dir.name
        csv_files = sorted(list(sub_dir.glob("*.csv")))

        if not csv_files:
            continue

        target_dir = output_dir / material_name
        target_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, sep=",")

                # The first column contains milliseconds; convert to seconds
                first_col = df.columns[0]
                df[first_col] = df[first_col].astype(float) / 1000.0
                df = df.rename(columns={first_col: "time"})

                # Standardize accelerometer axis names if needed
                if len(df.columns) >= 4:
                    df.columns = ["time", "ax", "ay", "az"]

                output_path = target_dir / file_path.name
                df.to_csv(output_path, index=False)
                success_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to process {file_path.name}: {e}")

        print(
            f"Processed: {material_name:<15} | Converted {success_count}/{len(csv_files)} files."
        )


if __name__ == "__main__":
    # Base directory set relative to project root
    project_root = Path(__file__).resolve().parent.parent

    raw_input_directory = project_root / "data" / "raw"
    formatted_output_directory = (
        project_root / "data" / "interim" / "01_time_standardized"
    )

    print("=== STEP 01: TIME STANDARDIZATION ===")
    process_raw_time(raw_input_directory, formatted_output_directory)
    print("=== COMPLETED ===\n")