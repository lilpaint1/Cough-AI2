import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import threading
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---  (ใช้ data ชุดเดียวกับ RF: features_raw_new.npz)
DATA_PATH = "features_raw_new.npz"
MODEL_OUTPUT_PATH = "cough_xgb_model_new.pkl"
CLASSES = ["covid", "healthy", "symptomatic"]

# ─── Hyperparameters (XGBoost) ───────────────────────────────────────────────
N_ESTIMATORS  = 600
MAX_DEPTH     = 8        # boosting ใช้ต้นไม้ตื้นกว่า RF (RF=25 ลึกเพราะไม่ boost)
LEARNING_RATE = 0.1
SUBSAMPLE     = 0.9
COLSAMPLE     = 0.9
N_JOBS        = -1
# ────────────────────────────────────────────────────────────────────────────


def smote_with_progress(X, y):
    """
    รัน SMOTE บน background thread พร้อม spinner + elapsed time
    เพราะ SMOTE ไม่รองรับ callback → ใช้ thread แสดง progress แทน
    (logic เดียวกับ train_rf_new.py)
    """
    result = {}
    exc_holder = {}

    def _run():
        try:
            sm = SMOTE(random_state=42)
            result['X'], result['y'] = sm.fit_resample(X, y)
        except Exception as e:
            exc_holder['err'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    idx = 0
    t0 = time.time()
    while t.is_alive():
        elapsed = time.time() - t0
        print(f"\r   {spinner[idx % len(spinner)]}  SMOTE กำลังทำงาน... "
              f"{elapsed:.1f}s  (สร้าง synthetic samples สำหรับ {len(X):,} rows × {X.shape[1]} features)",
              end="", flush=True)
        idx += 1
        time.sleep(0.15)

    print()  # newline หลัง spinner หยุด

    if 'err' in exc_holder:
        raise exc_holder['err']

    elapsed = time.time() - t0
    print(f"   ✅ SMOTE เสร็จใน {elapsed:.1f}s")
    return result['X'], result['y']


def train_xgb_model():
    print("🔄 กำลังโหลดข้อมูลสำหรับ XGBoost...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ ไม่พบไฟล์ {DATA_PATH} โปรดรัน rf_extract_new.py ก่อน")
        return

    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]
    print(f"   → X shape: {X.shape} | classes: {np.unique(y, return_counts=True)}")

    # balance ด้วย SMOTE (เหมือน RF) แล้วค่อยแบ่ง — logic เดิมเป๊ะ
    print("⚖️  ใช้ SMOTE เพื่อ balance class  [อาจใช้เวลา 1-3 นาที]...")
    X_resampled, y_resampled = smote_with_progress(X, y)
    print(f"   → หลัง SMOTE: {np.unique(y_resampled, return_counts=True)}")

    print("✂️  แบ่งข้อมูล train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2, stratify=y_resampled, random_state=42
    )

    # ─── Train XGBoost ────────────────────────────────────────────────────────
    print(f"🚀 เริ่มเทรน XGBoost ({N_ESTIMATORS} rounds, max_depth={MAX_DEPTH})...")
    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        objective="multi:softprob",
        num_class=len(CLASSES),
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"   ✅ เทรนเสร็จใน {time.time()-t0:.1f}s")

    # ─── Evaluation ──────────────────────────────────────────────────────────
    print("\n🧠 กำลังประเมินโมเดล...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy (XGB): {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    importances = clf.feature_importances_
    top5_idx = np.argsort(importances)[::-1][:5]
    print("🔍 Top-5 feature indices:", top5_idx,
          "| importances:", np.round(importances[top5_idx], 4))

    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print(f"💾 บันทึกโมเดล: {MODEL_OUTPUT_PATH}")

    # ─── Confusion Matrix ────────────────────────────────────────────────────
    print("\n📊 กำลังสร้าง Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – XGBoost")
    plt.xlabel("Predicted labels"); plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig("xgb_confusion_matrix_new.png", dpi=150)
    plt.close()
    print("✅ บันทึก xgb_confusion_matrix_new.png")

    # ─── ROC Curve ───────────────────────────────────────────────────────────
    print("📈 กำลังสร้าง ROC Curve...")
    n_classes = len(CLASSES)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = clf.predict_proba(X_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    colors = ['#e377c2', '#17becf', '#bcbd22']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{CLASSES[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2.5, linestyle='--',
             label=f"Macro-avg (AUC = {macro_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – XGBoost")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("xgb_roc_curve_new.png", dpi=150)
    plt.close()
    print("✅ บันทึก xgb_roc_curve_new.png")


if __name__ == "__main__":
    train_xgb_model()
