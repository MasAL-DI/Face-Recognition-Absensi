"""
Training CNN anti-spoofing berbasis transfer learning MobileNetV2.

Struktur dataset:

dataset_spoof/
  train/
    real/
      real_001.jpg
    spoof/
      foto_hp_001.jpg
  val/
    real/
    spoof/

Catatan skripsi:
- real  = wajah asli langsung dari kamera
- spoof = serangan foto HP, foto cetak, atau video replay
- semakin beragam cahaya, jarak, dan perangkat, semakin baik generalisasi model
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

try:
    from config_lanjutan import MODEL_DIR, SPOOF_INPUT_SIZE, SPOOF_LABEL_FILE, SPOOF_MODEL_FILE
except Exception:  # pragma: no cover
    MODEL_DIR = Path("model")
    SPOOF_INPUT_SIZE = 224
    SPOOF_MODEL_FILE = MODEL_DIR / "spoof_cnn.keras"
    SPOOF_LABEL_FILE = MODEL_DIR / "spoof_labels.json"

DATASET_DIR = Path("dataset_spoof")
BATCH_SIZE = 16
EPOCHS = 12


def build_model() -> tf.keras.Model:
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        print("Menggunakan bobot ImageNet untuk MobileNetV2.")
    except Exception:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE, 3),
            include_top=False,
            weights=None,
        )
        print("Bobot ImageNet tidak tersedia. Training dari awal; butuh data lebih banyak.")

    base.trainable = False

    inputs = layers.Input(shape=(SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spoof_probability")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main() -> None:
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        print("Dataset belum ada.")
        print("Buat folder: dataset_spoof/train/real, dataset_spoof/train/spoof, dataset_spoof/val/real, dataset_spoof/val/spoof")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "spoof"],
        image_size=(SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["real", "spoof"],
        image_size=(SPOOF_INPUT_SIZE, SPOOF_INPUT_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(SPOOF_MODEL_FILE), monitor="val_auc", mode="max", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SPOOF_MODEL_FILE)

    metadata = {
        "classes": {"0": "real", "1": "spoof"},
        "output": "spoof_probability",
        "input_size": SPOOF_INPUT_SIZE,
        "epochs": len(history.history.get("loss", [])),
        "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
        "final_val_auc": float(history.history.get("val_auc", [0])[-1]),
    }
    SPOOF_LABEL_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model tersimpan: {SPOOF_MODEL_FILE}")
    print(f"Metadata tersimpan: {SPOOF_LABEL_FILE}")


if __name__ == "__main__":
    main()
