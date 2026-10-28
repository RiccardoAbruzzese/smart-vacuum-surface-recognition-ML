"""
06_model_benchmark.py
---------------------
Rigorous multi-seed benchmarking across SVM, KNN (k=4), and Random Forest 
using the 7 selected orthogonal spatial features. Computes Mean, Std Dev, 
Min-Max intervals, normalized percentage confusion matrices, and 2D t-SNE projections.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

# Top 7 orthogonal features verified in step 05 (retaining Standard Deviation over Variance)
SELECTED_FEATURES = [
    "Total Spatial Energy",
    "Standard Deviation",
    "High-Freq Energy",
    "Maximum Peak",
    "Average Damping",
    "Low-Freq Energy quote",
    "Dominant Spatial Freq",
]

NUM_SEEDS = 10
OPTIMAL_K = 4


def benchmark_models(dataset_path: Path, output_assets_dir: Path) -> None:
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return

    output_assets_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)

    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    target_col = "target_material" if "target_material" in df.columns else "Materiale"

    for col in available_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[available_features].to_numpy().astype(float)
    y = df[target_col].to_numpy()
    classes = np.unique(y)

    print(f"Loaded {X.shape[0]} samples across {len(classes)} classes: {list(classes)}")
    print(f"Active Features ({len(available_features)}): {available_features}\n")

    svm_accs, knn_accs, rf_accs = [], [], []
    last_y_test, last_rf_pred = None, None
    last_X_test_norm = None

    for seed in range(NUM_SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )

        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)

        # 1. Support Vector Machine
        clf_svm = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        clf_svm.fit(X_train_norm, y_train)
        svm_accs.append(clf_svm.score(X_test_norm, y_test))

        # 2. K-Nearest Neighbors (k=4)
        clf_knn = KNeighborsClassifier(n_neighbors=OPTIMAL_K, weights="uniform")
        clf_knn.fit(X_train_norm, y_train)
        knn_accs.append(clf_knn.score(X_test_norm, y_test))

        # 3. Random Forest (100 trees)
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight="balanced")
        clf_rf.fit(X_train_norm, y_train)
        rf_accs.append(clf_rf.score(X_test_norm, y_test))

        if seed == 0:
            last_y_test = y_test
            last_rf_pred = clf_rf.predict(X_test_norm)
            last_X_test_norm = X_test_norm

    print("=" * 65)
    print(f"MULTI-SEED BENCHMARK RESULTS ({NUM_SEEDS} INDEPENDENT ITERATIONS)")
    print("=" * 65)
    for model_name, acc_list in [
        ("SUPPORT VECTOR MACHINE (SVM)", svm_accs),
        ("K-NEAREST NEIGHBORS (KNN, k=4)", knn_accs),
        ("RANDOM FOREST (RF)", rf_accs),
    ]:
        mean_acc = np.mean(acc_list) * 100
        std_acc = np.std(acc_list) * 100
        min_acc = np.min(acc_list) * 100
        max_acc = np.max(acc_list) * 100
        print(f"{model_name:<32}")
        print(f"  Average Accuracy : {mean_acc:.2f}%")
        print(f"  Standard Deviation: {std_acc:.2f}%")
        print(f"  Interval          : [{min_acc:.2f}% - {max_acc:.2f}%]\n")
    print("=" * 65)

    print("\nDetailed Random Forest Classification Report (Seed 0):")
    print(classification_report(last_y_test, last_rf_pred))

    # Confusion Matrix Normalized (%)
    cm = confusion_matrix(last_y_test, last_rf_pred, labels=classes)
    cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

    plt.figure(figsize=(8, 6.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_perc,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in classes],
        yticklabels=[c.capitalize() for c in classes],
        cbar_kws={"label": "Percentage (%)"}
    )
    plt.title("Random Forest - Normalized Confusion Matrix (%)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = output_assets_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"Saved: {cm_path}")
    plt.close()

    # t-SNE 2D Projection
    perplexity_val = min(30, max(5, len(last_X_test_norm) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    X_tsne = tsne.fit_transform(last_X_test_norm)

    plt.figure(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", len(classes))

    for i, cls in enumerate(classes):
        mask_cls = (last_y_test == cls)
        mask_ok = mask_cls & (last_y_test == last_rf_pred)
        mask_err = mask_cls & (last_y_test != last_rf_pred)

        if np.any(mask_ok):
            plt.scatter(
                X_tsne[mask_ok, 0], X_tsne[mask_ok, 1],
                color=palette[i], label=f"{cls.capitalize()} (Correct)",
                edgecolor="black", linewidth=0.5, s=65, alpha=0.85
            )
        if np.any(mask_err):
            plt.scatter(
                X_tsne[mask_err, 0], X_tsne[mask_err, 1],
                color="red", label=f"{cls.capitalize()} (Error)",
                edgecolor="black", linewidth=0.8, s=80, marker="X"
            )

    plt.title("t-SNE Spatial Feature Space (Random Forest Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    tsne_path = output_assets_dir / "tsne_projection.png"
    plt.savefig(tsne_path, dpi=300)
    print(f"Saved: {tsne_path}")
    plt.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_csv = project_root / "data" / "processed" / "features_dataset.csv"
    assets_dir = project_root / "docs" / "assets"

    print("=== STEP 06: MULTI-SEED MODEL BENCHMARKING ===")
    benchmark_models(dataset_csv, assets_dir)
    print("=== COMPLETED ===\n")