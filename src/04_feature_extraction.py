"""
04_feature_extraction.py
------------------------
Extracts 14 statistical, morphological, spatial-spectral, and dynamic damping features 
from 250 mm spatial acceleration windows.

Safety feature: by default, saves extracted features to a sample file 
(data/processed/features_dataset_sample.csv) to avoid accidentally overwriting 
the verified full-benchmark dataset (features_dataset.csv).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as sc
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, butter, filtfilt, find_peaks, peak_widths
from scipy.signal.windows import hann

# Linear scan velocity for quadratic normalization
SCAN_VELOCITIES = {
    "asphalt": 0.20768,
    "carpet": 0.24919,
    "marble": 0.48013,
    "parquet": 0.32362,
    "tile": 0.39200
}

HIGH_FREQ_SPATIAL_THRESHOLD_CYCLES_MM = 0.30


def compute_equivalent_damping(signal: np.ndarray, fs_spatial: float) -> float:
    """
    Estimates average equivalent damping ratio (zeta) across the 3 dominant peaks
    using the half-power bandwidth method (-3 dB).
    """
    freqs, pxx = welch(signal, fs=fs_spatial, nperseg=len(signal)//2, window='hann')
    peaks, _ = find_peaks(pxx, prominence=np.max(pxx) * 0.05)
    
    if len(peaks) == 0:
        return np.nan

    peaks_sorted = peaks[np.argsort(pxx[peaks])][::-1]
    top_peaks = peaks_sorted[:3]
    zetas = []

    for idx_peak in top_peaks:
        f_center = freqs[idx_peak]
        bandwidth = max(0.01, f_center * 0.20)
        
        lowcut = max(0.001, f_center - bandwidth / 2.0)
        highcut = min(fs_spatial / 2.0 - 0.001, f_center + bandwidth / 2.0)
        nyq = fs_spatial / 2.0

        b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
        filtered_sig = filtfilt(b, a, signal)

        f_filt, pxx_filt = welch(filtered_sig, fs=fs_spatial, nperseg=len(signal)//2, window='hann')
        peaks_filt, _ = find_peaks(pxx_filt, prominence=np.max(pxx_filt) * 0.1)
        
        if len(peaks_filt) > 0:
            idx_max_filt = peaks_filt[np.argmax(pxx_filt[peaks_filt])]
            f_dom = f_filt[idx_max_filt]
            
            widths, _, _, _ = peak_widths(pxx_filt, [idx_max_filt], rel_height=0.5)
            delta_f = widths[0] * (f_filt[1] - f_filt[0])
            
            if f_dom > 0:
                zetas.append(delta_f / (2.0 * f_dom))

    return float(np.nanmean(zetas)) if len(zetas) > 0 else np.nan


def extract_spatial_features(input_dir: Path, output_dir: Path, overwrite_benchmark: bool = False) -> None:
    """
    Extracts spatial domain features and writes to CSV without destroying the benchmark file.
    """
    subdirectories = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not subdirectories:
        print(f"[WARNING] No subdirectories found in: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_extracted_records = []

    for sub_dir in subdirectories:
        material_name = sub_dir.name.lower()
        if material_name not in SCAN_VELOCITIES:
            continue

        velocity_m_s = SCAN_VELOCITIES[material_name]
        csv_files = sorted(list(sub_dir.glob("*.csv")))
        if not csv_files:
            continue

        print(f"Feature Extraction: {material_name.upper():<10} | Windows: {len(csv_files)}")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, sep=",")
                space = df["space_mm"].values.astype(float)
                acc_z_raw = df["az"].values.astype(float)

                # Quadratic acceleration scaling to isolate surface roughness
                scaling_ratio = velocity_m_s / 0.3
                signal = acc_z_raw / (scaling_ratio ** 2)

                dx = np.mean(np.diff(space))
                if dx <= 0:
                    continue
                fs_spatial = 1.0 / dx

                # 1. Amplitude & Shape metrics
                rms = np.sqrt(np.mean(signal ** 2))
                std_dev = np.std(signal)
                variance = np.var(signal)
                peak_max = np.max(np.abs(signal))
                peak_min = np.min(np.abs(signal))
                crest_factor = peak_max / rms if rms != 0 else 0.0
                skewness = float(sc.skew(signal))
                kurtosis = float(sc.kurtosis(signal))

                # 2. Fourier spatial spectral metrics
                n = len(signal)
                fft_vals = rfft(signal * hann(n))
                fft_freq_spatial = rfftfreq(n, d=1.0 / fs_spatial)
                spectrum_amp = (np.abs(fft_vals) * 2.0 / n) * 2.0

                idx_dominant = 1 + np.argmax(spectrum_amp[1:])
                dominant_spatial_freq = fft_freq_spatial[idx_dominant]

                total_spatial_energy = np.sum(spectrum_amp ** 2)
                high_freq_mask = fft_freq_spatial > HIGH_FREQ_SPATIAL_THRESHOLD_CYCLES_MM
                high_freq_energy = np.sum(spectrum_amp[high_freq_mask] ** 2)
                low_freq_energy = total_spatial_energy - high_freq_energy
                low_freq_energy_quote = low_freq_energy / total_spatial_energy if total_spatial_energy != 0 else 0.0

                # 3. Dynamic equivalent damping
                avg_damping = compute_equivalent_damping(signal, fs_spatial)

                record = {
                    "file_name": file_path.name,
                    "target_material": material_name,
                    "RMS": round(rms, 6),
                    "Standard Deviation": round(std_dev, 6),
                    "Variance": round(variance, 6),
                    "Maximum Peak": round(peak_max, 6),
                    "Minimum Peak": round(peak_min, 6),
                    "Crest Factor": round(crest_factor, 4),
                    "Skewness": round(skewness, 4),
                    "Kurtosis": round(kurtosis, 4),
                    "Dominant Spatial Freq": round(dominant_spatial_freq, 6),
                    "Total Spatial Energy": round(total_spatial_energy, 6),
                    "High-Freq Energy": round(high_freq_energy, 6),
                    "Low-Freq Energy": round(low_freq_energy, 6),
                    "Low-Freq Energy quote": round(low_freq_energy_quote, 6),
                    "Average Damping": round(avg_damping, 6) if pd.notna(avg_damping) else np.nan
                }
                all_extracted_records.append(record)

            except Exception as e:
                print(f"  [ERROR] Feature extraction failed on {file_path.name}: {e}")

    if all_extracted_records:
        df_all = pd.DataFrame(all_extracted_records)
        
        # Determine target file name to protect benchmark data
        if overwrite_benchmark:
            target_csv = output_dir / "features_dataset.csv"
            print("\n[CAUTION] Overwriting benchmark dataset: features_dataset.csv")
        else:
            target_csv = output_dir / "features_dataset_sample.csv"
            print(f"\n[INFO] Saving sample extraction output to: {target_csv.name}")
            print("  (The validated 'features_dataset.csv' was kept intact for model benchmarks)")

        df_all.to_csv(target_csv, index=False)
        print(f"[SUCCESS] Extracted {len(df_all)} feature records across {df_all['target_material'].nunique()} classes.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_windows_dir = project_root / "data" / "interim" / "03_spatial_windows_250mm"
    output_processed_dir = project_root / "data" / "processed"

    print("=== STEP 04: SPATIAL FEATURE EXTRACTION ===")
    # Set overwrite_benchmark=True only if you explicitly want to regenerate the main dataset
    extract_spatial_features(input_windows_dir, output_processed_dir, overwrite_benchmark=False)
    print("=== COMPLETED ===\n")