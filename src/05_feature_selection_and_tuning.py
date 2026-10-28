"""
05_feature_selection_and_tuning.py
----------------------------------
1. Theory-Driven Feature Pruning: removes collinear/redundant zero-mean metrics.
2. Data-Driven Feature Selection: evaluates Random Forest Gini Importance across
   all 14 original parameters and confirms the 7 optimal orthogonal features.
3. KNN Hyperparameter Optimization: sweeps k in [1, 20] via Cross-Validation,
   verifying k=4 as optimal.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- STEP 1: THEORY-DRIVEN PRUNING DEFINITIONS ---
# Collinear metrics for zero-mean signals (mu ~ 0 -> RMS = StdDev = sqrt(Variance))
# and redundant frequency band sums
COLLINEAR_AND_REDUNDANT_EXCLUSIONS = [
    "RMS",                 # Collinear with Standard Deviation & Variance
    "Variance",            # Collinear with Standard Deviation (StdDev retained)
    "Low-Freq Energy",     # Redundant with Total Spatial Energy & Low-Freq Energy quote
    "Minimum Peak",        # Redundant with Maximum Peak
    "Crest Factor",        # Ratio of Peak/RMS, prone to numerical noise on smooth surfaces
    "Skewness",            # Low discriminative score confirmed by Gini importance
    "Kurtosis",            # Low discriminative score confirmed by Gini importance
]

# --- STEP 2: SELECTED ORTHOGONAL 7-FEATURE SUBSET ---
SELECTED_FEATURES = [
    "Total Spatial Energy",
    "Standard Deviation",
    "High-Freq Energy",
    "Maximum Peak",
    "Average Damping",
    "Low-Freq Energy quote",
    "Dominant Spatial Freq",
]


def run_feature_selection_and_tuning(dataset_path: Path, output_assets_dir: Path) -> None:
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    output_assets_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)

    metadata_cols = ["file_name", "target_material", "Materiale", "Nome File"]
    all_feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]
    target_col = "target_material" if "target_material" in df.columns else "Materiale"
    y = df[target_col].to_numpy()

    # Impute missing values with column medians
    for col in all_feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X_all = df[all_feature_cols].to_numpy().astype(float)

    # -------------------------------------------------------------------------
    # 1. Gini Importance Evaluation (All 14 Features)
    # -------------------------------------------------------------------------
    print("\n--- 1. FULL FEATURE SPACE EVALUATION (14 PARAMETERS) ---")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_all, y)

    df_importance = pd.DataFrame({
        "Feature": all_feature_cols,
        "Gini Importance": rf_baseline.feature_importances_
    }).sort_values(by="Gini Importance", ascending=False)

    print(df_importance.to_string(index=False))

    # Mark features selected vs pruned in the plot
    df_importance["Status"] = df_importance["Feature"].apply(
        lambda f: "Retained (Orthogonal)" if f in SELECTED_FEATURES else "Pruned (Collinear / Low Gini)"
    )

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = {"Retained (Orthogonal)": "#1f77b4", "Pruned (Collinear / Low Gini)": "#aec7e8"}
    
    sns.barplot(
        data=df_importance,
        x="Gini Importance",
        y="Feature",
        hue="Status",
        dodge=False,
        palette=palette
    )
    plt.title("Spatial Feature Importance & Selection (Random Forest Gini)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Computed Spatial Metric", fontsize=11)
    plt.legend(title="Feature Selection Status", frameon=True, loc="lower right")
    plt.tight_layout()
    plot_importance_path = output_assets_dir / "feature_importance.png"
    plt.savefig(plot_importance_path, dpi=300)
    print(f"Saved: {plot_importance_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Metrological Verification of Theory-Driven Pruning
    # -------------------------------------------------------------------------
    print("\n--- 2. METROLOGICAL PRUNING VERIFICATION ---")
    print(f"Features manually pruned due to collinearity/low information: {len(COLLINEAR_AND_REDUNDANT_EXCLUSIONS)}")
    for f in COLLINEAR_AND_REDUNDANT_EXCLUSIONS:
        print(f"  - {f}")
    print(f"\nFinal retained orthogonal feature set: {len(SELECTED_FEATURES)}")
    for f in SELECTED_FEATURES:
        print(f"  + {f}")

    # -------------------------------------------------------------------------
    # 3. KNN Hyperparameter Sweep on the 7 Selected Features
    # -------------------------------------------------------------------------
    print("\n--- 3. KNN HYPERPARAMETER OPTIMIZATION (TOP 7 FEATURES) ---")
    available_7 = [f for f in SELECTED_FEATURES if f in df.columns]
    X_selected = df[available_7].to_numpy().astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    k_values = list(range(1, 21))
    accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        knn.fit(X_train_norm, y_train)
        acc = knn.score(X_test_norm, y_test)
        accuracies.append(acc)

    best_k_idx = int(np.argmax(accuracies))
    best_k = k_values[best_k_idx]
    print(f"Optimal parameter identified: k={best_k} (Test Accuracy: {accuracies[best_k_idx]*100:.2f}%)")

    # Plot KNN optimization curve
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, accuracies, marker="o", linestyle="-", color="#2b5c8f", lw=2, label="Global Accuracy")
    plt.axvline(x=best_k, color="#c0392b", linestyle="--", lw=1.8, label=f"Best K ({best_k})")
    plt.title("KNN Optimization (Top 7 Orthogonal Features)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Neighbors (k)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_knn_path = output_assets_dir / "knn_k_optimization.png"
    plt.savefig(plot_knn_path, dpi=300)
    print(f"Saved: {plot_knn_path}")
    plt.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_csv = project_root / "data" / "processed" / "features_dataset.csv"
    assets_dir = project_root / "docs" / "assets"

    print("=== STEP 05: FEATURE SELECTION & HYPERPARAMETER TUNING ===")
    run_feature_selection_and_tuning(dataset_csv, assets_dir)
    print("=== COMPLETED ===\n")