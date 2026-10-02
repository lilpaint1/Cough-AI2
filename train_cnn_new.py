import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import time, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_MANIFEST_PATH = "cnn_data_manifest_new.json"
MODEL_OUTPUT_PATH  = "cough_cnn_model_new.h5"
CLASSES     = ["covid", "healthy", "symptomatic"]
BATCH_SIZE  = 32
EPOCHS      = 100    # เพิ่ม ceiling ให้ EarlyStopping ตัดสินใจ
IMAGE_SHAPE = (128, 128, 1)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"🎮 GPU: {gpus[0].name}  |  memory_growth=True")
else:
    print("⚠️  CPU mode")


# ─── tqdm Callback (เหมือนเดิมทุกอย่าง) ────────────────────────────────────
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs    = self.params["epochs"]
        self.epoch_bar = tqdm(total=self.epochs, desc="📦 Epochs",
                              unit="ep", ncols=95, colour="cyan", position=0)
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_bar = tqdm(total=self.params.get("steps","?"),
                              desc=f"  Epoch {epoch+1:03d}/{self.epochs}",
                              unit="batch", ncols=95, leave=False,
                              colour="green", position=1)
        self._t0 = time.time()
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_bar.set_postfix(loss=f"{logs.get('loss',0):.4f}",
                                   acc=f"{logs.get('accuracy',0):.4f}")
        self.batch_bar.update(1)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.batch_bar.close()
        self.epoch_bar.set_postfix(
            loss=f"{logs.get('loss',0):.4f}",
            acc=f"{logs.get('accuracy',0):.4f}",
            val_loss=f"{logs.get('val_loss',0):.4f}",
            val_acc=f"{logs.get('val_accuracy',0):.4f}",
            t=f"{time.time()-self._t0:.0f}s")
        self.epoch_bar.update(1)
    def on_train_end(self, logs=None):
        self.epoch_bar.close()


# ─── Architecture เหมือนเดิมทุก block ──────────────────────────────────────
def build_model(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax'),
    ])
    return model


def augment(img, label):
    """
    Augment เฉพาะตอนเทรน (ไม่แตะ val)
    - SpecAugment แบบเบา: mask แนวเวลา + แนวความถี่
    - time shift เล็กน้อย
    logic เหมือนเดิม แค่เพิ่ม noise เพื่อให้ generalize ดีขึ้น
    """
    # Time masking: random block 0-20 cols
    t_start = tf.random.uniform((), 0, IMAGE_SHAPE[1]-20, dtype=tf.int32)
    t_mask  = tf.concat([
        tf.ones([IMAGE_SHAPE[0], t_start, IMAGE_SHAPE[2]]),
        tf.zeros([IMAGE_SHAPE[0], tf.random.uniform((), 0, 20, dtype=tf.int32), IMAGE_SHAPE[2]]),
        tf.ones([IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]),
    ], axis=1)[:, :IMAGE_SHAPE[1], :]
    img = img * tf.cast(t_mask, tf.float32)

    # Frequency masking: random block 0-16 rows
    f_start = tf.random.uniform((), 0, IMAGE_SHAPE[0]-16, dtype=tf.int32)
    f_mask  = tf.concat([
        tf.ones([f_start, IMAGE_SHAPE[1], IMAGE_SHAPE[2]]),
        tf.zeros([tf.random.uniform((), 0, 16, dtype=tf.int32), IMAGE_SHAPE[1], IMAGE_SHAPE[2]]),
        tf.ones([IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]]),
    ], axis=0)[:IMAGE_SHAPE[0], :, :]
    img = img * tf.cast(f_mask, tf.float32)

    return img, label


def train_cnn_model():
    print("🔵 โหลด manifest...")
    if not os.path.exists(DATA_MANIFEST_PATH):
        print(f"❌ ไม่พบ {DATA_MANIFEST_PATH}"); return

    with open(DATA_MANIFEST_PATH) as f:
        manifest = json.load(f)

    print("🔵 โหลดข้อมูล...")
    X_full, y_full = [], []
    for entry in tqdm(manifest, desc="📂 Loading", unit="file", ncols=80, colour="blue"):
        try:
            feat = np.load(entry['filepath'])
            cols = IMAGE_SHAPE[1]
            if feat.shape[1] > cols:
                feat = feat[:, :cols, :]
            elif feat.shape[1] < cols:
                feat = np.pad(feat, ((0,0),(0,cols-feat.shape[1]),(0,0)))
            X_full.append(feat)
            y_full.append(entry['label'])
        except FileNotFoundError:
            pass

    X_full = np.array(X_full, dtype=np.float32)
    y_full = np.array(y_full)
    print(f"   → {X_full.shape}  RAM: {X_full.nbytes/1e9:.2f} GB")

    # normalize เหมือนเดิม
    X_min, X_max = X_full.min(), X_full.max()
    X_full = (X_full - X_min) / (X_max - X_min + 1e-8)

    print("⚖️  RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    idx_res, y_res = ros.fit_resample(
        np.arange(len(X_full)).reshape(-1,1), y_full)
    idx_res = idx_res.ravel()
    print(f"   → {np.unique(y_res, return_counts=True)}")

    tr_idx, te_idx, y_tr, y_te = train_test_split(
        idx_res, y_res, test_size=0.2, stratify=y_res, random_state=42)
    print(f"   → train: {len(tr_idx):,}  val: {len(te_idx):,}")

    # ─── tf.data pipeline (logic เหมือนเดิม + augment บน train เท่านั้น) ────
    AUTO = tf.data.AUTOTUNE

    def make_ds(indices, labels, shuffle=False, augment_fn=None):
        ds = tf.data.Dataset.from_tensor_slices(
            (indices.astype(np.int32), labels.astype(np.int32)))
        if shuffle:
            ds = ds.shuffle(8192, seed=42)
        def load(i, lbl):
            img = tf.py_function(lambda x: X_full[int(x)], [i], tf.float32)
            img.set_shape(IMAGE_SHAPE)
            return img, lbl
        ds = ds.map(load, num_parallel_calls=AUTO)
        if augment_fn:
            ds = ds.map(augment_fn, num_parallel_calls=AUTO)
        return ds.batch(BATCH_SIZE).prefetch(AUTO)

    train_ds = make_ds(tr_idx, y_tr, shuffle=True)
    val_ds   = make_ds(te_idx, y_te)

    print("🏗️  สร้างโมเดล...")
    model = build_model(IMAGE_SHAPE, len(CLASSES))
    model.summary(print_fn=lambda x: print("  " + x))

    # class_weight: ช่วย healthy ที่ recall ต่ำ (67%) ให้ได้รับ penalty มากขึ้น
    n_total = len(y_tr)
    n_cls   = len(CLASSES)
    class_weight = {
        i: n_total / (n_cls * np.sum(y_tr == i))
        for i in range(n_cls)
    }
    print(f"   → class_weight: { {CLASSES[k]: round(v,3) for k,v in class_weight.items()} }")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        TQDMProgressBar(),
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),
    ]

    print(f"\n⚙️  เทรน  batch={BATCH_SIZE}  epochs≤{EPOCHS}...")
    history = model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks, verbose=0,
    )

    print("\n✅ เสร็จสิ้น")

    # ─── Evaluate ─────────────────────────────────────────────────────────────
    y_pred_proba = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_te, y_pred, target_names=CLASSES))

    # ─── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix – CNN")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix_new.png", dpi=150)
    plt.close()
    print("✅ บันทึก cnn_confusion_matrix_new.png")

    # ─── ROC Curve ────────────────────────────────────────────────────────────
    n_classes   = len(CLASSES)
    y_te_bin    = label_binarize(y_te, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_te_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)) / n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(7,6))
    colors = ['#e377c2', '#17becf', '#bcbd22']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f"{CLASSES[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot(all_fpr, mean_tpr, 'darkorange', lw=2.5, linestyle='--',
             label=f"Macro-avg (AUC={macro_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – CNN")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("cnn_roc_curve_new.png", dpi=150)
    plt.close()
    print("✅ บันทึก cnn_roc_curve_new.png")

    # ─── Training Curve ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig("cnn_training_curve_new.png", dpi=150)
    plt.close()
    print("✅ บันทึก cnn_training_curve_new.png")

    model.save(MODEL_OUTPUT_PATH)
    print(f"💾 {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_cnn_model()