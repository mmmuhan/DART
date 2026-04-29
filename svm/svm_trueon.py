import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import pandas as pd
import warnings
from tqdm import tqdm
import joblib

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(data):
    """extract statistical features from time distributions
       Input data shape: (n_samples, n_timepoints)
    """
    n_features = data.shape[1]

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    cv = std / (mean + 1e-10)  # avoid /0
    max_ = np.max(data, axis=1, keepdims=True)

    percentile25 = np.percentile(data, 25, axis=1, keepdims=True)
    percentile50 = np.percentile(data, 50, axis=1, keepdims=True)
    percentile75 = np.percentile(data, 75, axis=1, keepdims=True)

    prop_gt1 = np.sum(data > 1, axis=1, keepdims=True) / n_features
    prop_gt2 = np.sum(data > 2, axis=1, keepdims=True) / n_features
    prop_gt3 = np.sum(data > 3, axis=1, keepdims=True) / n_features

    features = np.hstack([
        mean, std, cv, max_, prop_gt1, prop_gt2, prop_gt3,
        percentile25, percentile50, percentile75
    ])
    return features


# ------------------------------
# Plot pairwise CM + ROC (optional)
# ------------------------------
def plot_pairwise_cm_roc(y_true_all, y_pred_proba_all,
                         class_a, class_b, class_names,
                         exp_dir, kernel_name, fs1=13, fs2=12,
                         vmin=0.0, vmax=None, cbar_ticks=None):
    mask = (y_true_all == class_a) | (y_true_all == class_b)
    y_true = y_true_all[mask]
    y_proba_pair = y_pred_proba_all[mask][:, [class_a, class_b]]
    y_pred_pair = np.where(y_proba_pair[:, 0] >= y_proba_pair[:, 1], class_a, class_b)

    cm = confusion_matrix(y_true, y_pred_pair, labels=[class_a, class_b]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_percent = cm / row_sums

    y_true_bin = (y_true == class_a).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, y_proba_pair[:, 0])
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(2, 1, figsize=(4, 6.5))
    wd = 1.3

    sns.heatmap(
        cm_percent, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=[class_names[class_a], class_names[class_b]],
        yticklabels=[class_names[class_a], class_names[class_b]],
        annot_kws={"size": 14}, ax=axes[0],
        vmin=vmin, vmax=vmax, cbar=False
    ) #cbar_kws={"ticks": cbar_ticks}
    
    # optional styling
    #cbar = axes[0].collections[0].colorbar
    #cbar.ax.tick_params(labelsize=fs2, width=1.3)
    
    axes[0].set_xlabel('Predicted model', fontsize=fs1)
    axes[0].set_ylabel('True model', fontsize=fs1)
    axes[0].tick_params(axis='both', labelsize=fs2, width=wd)

    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2.0, color="#d73027")
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=fs1)
    axes[1].set_ylabel('True Positive Rate', fontsize=fs1)
    legend1 = axes[1].legend(fontsize=fs2, loc='lower right')
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].tick_params(axis='both', labelsize=fs2, width=wd)
    for spine in axes[1].spines.values():
        spine.set_linewidth(wd)

    pair_label = f"{class_names[class_a]} vs {class_names[class_b]}"
    legend2 = axes[1].legend(
        [pair_label], loc='upper left', fontsize=11,
        frameon=True, handlelength=0, handletextpad=0
    )
    handles_attr = getattr(legend2, "legend_handles", None) or getattr(legend2, "legendHandles", None)
    if handles_attr:
        for item in handles_attr:
            try:
                item.set_visible(False)
            except Exception:
                pass
    axes[1].add_artist(legend1)

    plt.tight_layout()
    out_path = os.path.join(exp_dir, f"rev_{class_a+2}v{class_b+2}_cm_roc_{kernel_name}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Pairwise CM + ROC saved: {out_path}")


# ------------------------------
# Main
# ------------------------------
def main(args):
    exp_dir = args.save_dir
    assert os.path.isdir(exp_dir), f"The directory {exp_dir} does not exist."

    print("[INFO] Loading data...")
    t0 = time.time()
    data = np.load(args.train_data)
    xmat = data["xmat"]        # (n_timepoints, n_samples) or (n_samples, n_timepoints)
    label = data["label"]
    # Ensure samples are rows
    xmat = xmat.T              # -> (n_samples, n_timepoints)

    # Map labels to 0..3 -> ['2-state','3-state','4-state','5-state']
    # Assumes original labels are 1..4 in the .npz (consistent with previous code).
    label = (label - 1).astype(int)
    class_names = ['2-state', '3-state', '4-state', '5-state']

    print(f"[INFO] Extracting features... input shape: {xmat.shape}")
    xmat_f = extract_features(xmat)
    print(f"[INFO] Features shape: {xmat_f.shape}")

    # ========= split (keep global indices) =========
    all_idx = np.arange(xmat.shape[0])   # global row indices in xmat
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        xmat_f, label, all_idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=label
    )

    # Save test set (features only + features+raw)
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.savez(os.path.join(results_dir, f"test_set_features_{args.kernel}.npz"),
             x_test=x_test, y_test=y_test, idx_test=idx_test)
    print(f"[INFO] Saved test set (features) -> results/test_set_features_{args.kernel}.npz")

    np.savez(os.path.join(results_dir, f"test_set_both_{args.kernel}.npz"),
             x_test_features=x_test,
             y_test=y_test,
             idx_test=idx_test,             # global row ids
             x_test_raw=xmat[idx_test])
    print(f"[INFO] Saved test set (features + raw) -> results/test_set_both_{args.kernel}.npz")

    # ========= Pipeline & GridSearch =========
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=args.random_state, class_weight='balanced'))
    ])
    param_grid = {
        'svm__C': np.logspace(args.c_min, args.c_max, args.c_steps),
        'svm__kernel': [args.kernel]
    }
    print("[INFO] Starting training with GridSearchCV...")
    t1 = time.time()
    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid,
        cv=args.cv, scoring=args.scoring,   # works for multiclass (ovo/ovr)
        n_jobs=args.n_jobs, verbose=args.verbose
    )
    grid_search.fit(x_train, y_train)
    t2 = time.time()
    print(f"[INFO] Training finished, time: {t2 - t1:.2f} s")
    best_model = grid_search.best_estimator_
    best_kernel_name = best_model.named_steps['svm'].kernel
    print(f"[INFO] Best parameters: {grid_search.best_params_}")

    # Save model
    model_path = os.path.join(exp_dir, f"svm_model_{best_kernel_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    # ========= Predict on test =========
    print("[INFO] Predicting on test set...")
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)

    # ========= Evaluation =========
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    print(f"[INFO] Accuracy: {acc:.4f}")
    print("[INFO] Classification report:\n", classification_report(y_test, y_pred))
    print(f"[INFO] Macro F1: {f1m:.4f}")

    # ========= Labels CSV aligned with test order =========
    test_idx_local = np.arange(len(y_test))
    df_labels = pd.DataFrame({
        "test_idx": test_idx_local,
        "global_idx": idx_test,
        "y_true": y_test,
        "y_pred": y_pred
    })
    df_labels["y_true_name"] = [class_names[i] for i in df_labels["y_true"]]
    df_labels["y_pred_name"] = [class_names[i] for i in df_labels["y_pred"]]
    # probabilities in class order: 0->2, 1->3, 2->4, 3->5
    for i, name in enumerate(class_names):
        df_labels[f"proba_{name}"] = y_pred_proba[:, i]

    labels_csv = os.path.join(results_dir, f"test_labels_{best_kernel_name}.csv")
    df_labels.to_csv(labels_csv, index=False)
    print(f"[INFO] Saved labels CSV (aligned with test order) -> {labels_csv}")

    # ========= Pairwise (only 2v3, 2v4, 2v5) =========
    print("[INFO] Pairwise confusion matrix + ROC + cell lists ...")
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]   # 2v3, 2v4, 2v5
    global_max = 0.9
    tick_vals = [0.2, 0.4, 0.6, 0.8]

    for a, b in pairs:
        plot_pairwise_cm_roc(
            y_true_all=y_test,
            y_pred_proba_all=y_pred_proba,
            class_a=a, class_b=b,
            class_names=class_names,
            exp_dir=exp_dir,
            kernel_name=best_kernel_name,
            vmin=0.0, vmax=global_max, cbar_ticks=tick_vals
        )

    print(f"[INFO] Total runtime: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM classification')

    # Data
    parser.add_argument('--train-data', type=str, default='synthetic_data/svmon/true/true_revon2345.npz',
                        help='Path to training data (.npz) with 4 classes (labels 1..4)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of test set')

    # Model
    parser.add_argument('--kernel', type=str, default='linear', help='SVM kernel type')
    parser.add_argument('--c-min', type=int, default=-3, help='Minimum C exponent (log10)')
    parser.add_argument('--c-max', type=int, default=2, help='Maximum C exponent (log10)')
    parser.add_argument('--c-steps', type=int, default=100, help='Number of C values to search')

    # Training
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    parser.add_argument('--scoring', type=str, default='roc_auc_ovo', help='Scoring')
    parser.add_argument('--random-state', type=int, default=1, help='Random seed')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose')

    # Output
    parser.add_argument('--save_dir', type=str, default='synthetic_data/svmon/true', help='Output folder path')

    args, _ = parser.parse_known_args()
    results = main(args)
