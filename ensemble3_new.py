"""
ensemble.py  –  Soft-Voting Ensemble: CNN + Random Forest
==========================================================
Strategy
--------
  1. Load the saved CNN model  (cough_cnn_model.h5)
  2. Load the saved RF model   (cough_rf_model.pkl)
  3. For every sample in the *shared* validation set:
       • CNN gets the Mel-spectrogram  (128×128×1)
       • RF  gets the 416-feature vector
  4. Average the two probability arrays  → argmax → final label
  5. Evaluate: classification report, confusion matrix, ROC curve

Directory assumptions (edit paths at the bottom if needed)
-----------------------------------------------------------
  CNN manifest : cnn_data_manifest.json   (produced by cnn_extract.py)
  RF  features : features_raw.npz          (produced by rf_extract.py)
  CNN model    : cough_cnn_model.h5
  RF  model    : cough_rf_model.pkl
"""

import os, json, warnings

# ── Force CPU inference (avoids cuDNN not found error on GPU) ─────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths (edit if your files live elsewhere) ─────────────────────────────────
CNN_MANIFEST   = "cnn_data_manifest_new.json"
RF_NPZ         = "features_raw_new.npz"
CNN_MODEL_PATH = "cough_cnn_model_new.h5"
RF_MODEL_PATH  = "cough_rf_model_new.pkl"
XGB_MODEL_PATH = "cough_xgb_model_new.pkl"

CLASSES     = ["covid", "healthy", "symptomatic"]
IMAGE_SHAPE = (128, 128, 1)
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Load & prepare CNN data  (mirrors train_cnn.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_cnn_data():
    print("🔵 [CNN] loading manifest …")
    with open(CNN_MANIFEST) as f:
        manifest = json.load(f)

    X, y = [], []
    for entry in manifest:
        try:
            feat = np.load(entry["filepath"])
            cols = IMAGE_SHAPE[1]
            if feat.shape[1] > cols:
                feat = feat[:, :cols, :]
            elif feat.shape[1] < cols:
                feat = np.pad(feat, ((0, 0), (0, cols - feat.shape[1]), (0, 0)))
            X.append(feat)
            y.append(entry["label"])
        except FileNotFoundError:
            pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # same normalisation as training
    X_min, X_max = X.min(), X.max()
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # same oversampling as training (RandomOverSampler)
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    idx_res, y_res = ros.fit_resample(
        np.arange(len(X)).reshape(-1, 1), y
    )
    idx_res = idx_res.ravel()

    # same 80/20 split + stratify
    tr_idx, te_idx, _, y_te = train_test_split(
        idx_res, y_res,
        test_size=0.2, stratify=y_res, random_state=RANDOM_SEED,
    )
    X_te_cnn = X[te_idx]          # shape (N, 128, 128, 1)
    print(f"   → val samples: {len(y_te):,}")
    return X_te_cnn, y_te


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Load & prepare RF data  (mirrors train_rf.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def load_rf_data():
    print("🔵 [RF] loading features …")
    data = np.load(RF_NPZ)
    X, y = data["X"], data["y"]

    # same SMOTE as training
    print("   ⚖️  SMOTE (may take a moment) …")
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X, y)

    _, X_te_rf, _, y_te = train_test_split(
        X_res, y_res,
        test_size=0.2, stratify=y_res, random_state=RANDOM_SEED,
    )
    print(f"   → val samples: {len(y_te):,}")
    return X_te_rf, y_te


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Soft-Voting Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_predict(cnn_model, rf_model, xgb_model, X_cnn, X_rf):
    """
    Soft-voting 3 ทาง — เฉลี่ยเท่ากัน (CNN + RF + XGB)
    XGB ใช้ฟีเจอร์ชุดเดียวกับ RF (X_rf, 416-D)
    """
    p_cnn = cnn_model.predict(X_cnn, verbose=0)          # (N, 3)
    p_rf  = rf_model.predict_proba(X_rf)                 # (N, 3)
    p_xgb = xgb_model.predict_proba(X_rf)                # (N, 3)
    p_ens = (p_cnn + p_rf + p_xgb) / 3.0                 # (N, 3)
    return p_ens


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, out_path="ensemble3_confusion_matrix_new.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – Ensemble (CNN + RF)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved {out_path}")


def plot_roc_curve(y_true, y_proba, out_path="ensemble3_roc_curve_new.png"):
    n_classes = len(CLASSES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)) / n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    colors = ["#e377c2", "#17becf", "#bcbd22"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{CLASSES[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(all_fpr, mean_tpr, "darkorange", lw=2.5, linestyle="--",
             label=f"Macro-avg (AUC={macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Ensemble (CNN + RF)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── load models ───────────────────────────────────────────────────────────
    print(f"📂 Loading CNN model from  {CNN_MODEL_PATH}")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    print(f"📂 Loading RF  model from  {RF_MODEL_PATH}")
    rf_model = joblib.load(RF_MODEL_PATH)

    print(f"📂 Loading XGB model from  {XGB_MODEL_PATH}")
    xgb_model = joblib.load(XGB_MODEL_PATH)

    # ── load validation data ──────────────────────────────────────────────────
    X_te_cnn, y_te_cnn = load_cnn_data()
    X_te_rf,  y_te_rf  = load_rf_data()

    # Sanity check – both pipelines must produce the same label sequence
    assert len(y_te_cnn) == len(y_te_rf), (
        f"Val-set size mismatch: CNN={len(y_te_cnn)}, RF={len(y_te_rf)}. "
        "Ensure both scripts use the same RANDOM_SEED and test_size."
    )
    assert np.array_equal(y_te_cnn, y_te_rf), (
        "Label order differs between CNN and RF val sets. "
        "Check random_state / stratify settings."
    )
    y_true = y_te_cnn

    # ── ensemble predict ──────────────────────────────────────────────────────
    print("\n🤝 Running Soft-Voting Ensemble 3-way (CNN + RF + XGB, เฉลี่ยเท่ากัน) …")
    p_ens  = ensemble_predict(cnn_model, rf_model, xgb_model, X_te_cnn, X_te_rf)
    y_pred = np.argmax(p_ens, axis=1)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n--- Classification Report (Ensemble 3-way) ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, p_ens)

    print("\n🎉 Done!  Output files:")
    print("   ensemble3_confusion_matrix_new.png")
    print("   ensemble3_roc_curve_new.png")


if __name__ == "__main__":
    main()