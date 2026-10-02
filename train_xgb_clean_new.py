"""
train_xgb_clean_new.py — XGBoost เวอร์ชัน "แก้ data leakage"
================================================================
ต่างจาก train_xgb_new.py ตรงลำดับเท่านั้น (โมเดล/ฟีเจอร์เหมือนเดิม):

  เดิม (leak) : SMOTE ทั้งก้อน → แบ่ง train/test   (test มีของปลอม)
  ใหม่ (clean): แบ่ง train/test ก่อน → SMOTE เฉพาะ train
                → test เป็นข้อมูลจริง ไม่บาลานซ์ (real-world distribution)

ผลที่คาด: ตัวเลขจะ "ลดลงและจริงขึ้น" — โดยเฉพาะ AUC จะไม่ใช่ 1.00 ปลอม ๆ
================================================================
"""
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_curve, auc, roc_auc_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
import time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "features_raw_new.npz"
MODEL_OUTPUT_PATH = "cough_xgb_model_clean.pkl"
CLASSES = ["covid", "healthy", "symptomatic"]

# Hyperparameters เดียวกับ train_xgb_new.py
N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE = 600, 8, 0.1
SUBSAMPLE, COLSAMPLE = 0.9, 0.9


def main():
    print("🔄 โหลดข้อมูล...")
    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]
    print(f"   → X: {X.shape} | classes: {np.unique(y, return_counts=True)}")

    # ✅ FIX 1: แบ่ง train/test "ก่อน" บนข้อมูลดิบ (test คงสัดส่วนจริง ไม่บาลานซ์)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"✂️  split ก่อน balance:")
    print(f"   train (ดิบ): {np.unique(y_train, return_counts=True)}")
    print(f"   test  (ดิบ-จริง): {np.unique(y_test, return_counts=True)}")

    # ✅ FIX 2: SMOTE เฉพาะ train เท่านั้น (test ไม่แตะ → ไม่มี leakage)
    print("⚖️  SMOTE เฉพาะ train fold...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"   train หลัง SMOTE: {np.unique(y_train_res, return_counts=True)}")

    # ─── Train ────────────────────────────────────────────────────────────────
    print(f"🚀 เทรน XGBoost ({N_ESTIMATORS} rounds)...")
    clf = XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE,
        objective="multi:softprob", num_class=len(CLASSES),
        tree_method="hist", eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train_res, y_train_res)
    print(f"   ✅ เทรนเสร็จใน {time.time()-t0:.1f}s")

    # ─── Evaluate บน test "จริง" ───────────────────────────────────────────────
    print("\n🧠 ประเมินบน test จริง (ไม่บาลานซ์)...")
    y_pred  = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)

    acc       = accuracy_score(y_test, y_pred)
    macro_f1  = f1_score(y_test, y_pred, average="macro")
    macro_pre = precision_score(y_test, y_pred, average="macro")
    macro_rec = recall_score(y_test, y_pred, average="macro")
    macro_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")

    print(f"\n================ ผลลัพธ์ (CLEAN / no leakage) ================")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro-AUC     : {macro_auc:.4f}   <-- ค่า 'จริง' (ไม่เว่อร์แล้ว)")
    print(f"  Macro-Precision: {macro_pre:.4f}")
    print(f"  Macro-Recall  : {macro_rec:.4f}")
    print(f"  Macro-F1      : {macro_f1:.4f}")
    print("==============================================================\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

    joblib.dump(clf, MODEL_OUTPUT_PATH)
    print(f"💾 บันทึกโมเดล: {MODEL_OUTPUT_PATH}")

    # ─── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – XGBoost (CLEAN, no leakage)")
    plt.xlabel("Predicted labels"); plt.ylabel("True labels")
    plt.tight_layout(); plt.savefig("xgb_clean_confusion_matrix.png", dpi=150); plt.close()
    print("✅ บันทึก xgb_clean_confusion_matrix.png")

    # ─── ROC Curve ─────────────────────────────────────────────────────────────
    n_classes = len(CLASSES)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)) / n_classes
    macro_auc_plot = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    colors = ['#e377c2', '#17becf', '#bcbd22']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{CLASSES[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2.5, linestyle='--',
             label=f"Macro-avg (AUC = {macro_auc_plot:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – XGBoost (CLEAN, no leakage)")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("xgb_clean_roc_curve.png", dpi=150); plt.close()
    print("✅ บันทึก xgb_clean_roc_curve.png")


if __name__ == "__main__":
    main()
