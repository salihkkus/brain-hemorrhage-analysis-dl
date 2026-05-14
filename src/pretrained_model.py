import os
import json
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
ISLENMIS_FOLDER = "data/islenmis_veri"
MODEL_KAYIT_YOLU = "modeller"
IMG_SIZE = (224, 224)
os.makedirs(MODEL_KAYIT_YOLU, exist_ok=True)
def seti_yukle_resnet(fold_no, set_adi):
    fold_yolu = os.path.join(ISLENMIS_FOLDER, f"fold_{fold_no}")
    klasor_yolu = os.path.join(fold_yolu, set_adi)
    csv_yolu = os.path.join(fold_yolu, f"{set_adi}_labels.csv")
    if not os.path.exists(csv_yolu):
        raise FileNotFoundError(f"Veri bulunamadı: {csv_yolu}. Önce veri_hazirlama.py çalıştırılmalı!")
    df = pd.read_csv(csv_yolu)
    X, y = [], []
    for index, row in df.iterrows():
        img_path = os.path.join(klasor_yolu, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(row['label'])
    X = np.array(X)
    X = preprocess_input(X)
    return X, np.array(y)
def resnet50_tasarla():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
def fine_tune_icin_coz(model, learning_rate=1e-5):
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
def capraz_dogrulama_resnet50_egitimi():
    print("\n--- 5-FOLD CROSS VALIDATION (RESNET-50 TRANSFER LEARNING) ---")
    katlama_skorlari = []
    tum_history = []
    en_iyi_val_acc = 0
    en_iyi_model_fold = 1
    for fold in range(1, 6):
        print(f"\n================ FOLD {fold} ================")
        X_train, y_train = seti_yukle_resnet(fold, "train")
        X_val, y_val = seti_yukle_resnet(fold, "val")
        print(f"Bölüm: Train={len(X_train)} (Artırılmış) | Val={len(X_val)} (Orijinal)")
        print(">> Aşama 1: Sınıflandırıcı Başlığı Eğitiliyor")
        model = resnet50_tasarla()
        early_stop_1 = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        hist_1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            callbacks=[early_stop_1],
            verbose=1
        )
        print(">> Aşama 2: Son 30 Katman Çözülüyor (Fine-Tuning)")
        model = fine_tune_icin_coz(model)
        early_stop_2 = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        hist_2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=32,
            callbacks=[early_stop_2],
            verbose=1
        )
        full_history = {}
        for metric in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
            full_history[metric] = hist_1.history[metric] + hist_2.history[metric]
        try:
            best_epoch_idx = early_stop_2.best_epoch if early_stop_2.best_epoch is not None else len(hist_2.history['val_loss']) - 1
            best_epoch_idx += len(hist_1.history['val_loss']) 
        except AttributeError:
            best_epoch_idx = len(full_history['val_loss']) - 1
        val_acc = full_history['val_accuracy'][best_epoch_idx]
        katlama_skorlari.append(val_acc)
        tum_history.append(full_history)
        print(f"✅ Fold {fold} ResNet50 Başarısı: %{val_acc*100:.2f}")
        if val_acc > en_iyi_val_acc:
            en_iyi_val_acc = val_acc
            en_iyi_model_fold = fold
            model.save(os.path.join(MODEL_KAYIT_YOLU, "pretrained_model.h5"))
    print("\n================ TRANSFER LEARNING (RESNET50) ÖZETİ ================")
    print(f"Ayrı Ayrı Katlama Başarıları : {[round(s, 3) for s in katlama_skorlari]}")
    print(f"🌟 ORTALAMA MODEL DOĞRULUĞU : %{np.mean(katlama_skorlari)*100:.2f}")
    print(f"💾 En başarılı ağırlık Fold {en_iyi_model_fold}'den diske kaydedildi.")
    avg_hist = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    min_epochs = min([len(h['accuracy']) for h in tum_history])
    for i in range(min_epochs):
        avg_hist['accuracy'].append( float(np.mean([h['accuracy'][i] for h in tum_history])) )
        avg_hist['val_accuracy'].append( float(np.mean([h['val_accuracy'][i] for h in tum_history])) )
        avg_hist['loss'].append( float(np.mean([h['loss'][i] for h in tum_history])) )
        avg_hist['val_loss'].append( float(np.mean([h['val_loss'][i] for h in tum_history])) )
    history_path = os.path.join(MODEL_KAYIT_YOLU, "pretrained_history.json")
    with open(history_path, 'w') as f:
        json.dump(avg_hist, f)
    print(f"📊 Ortalama eğitim grafiği kaydedildi: {history_path}")
if __name__ == "__main__":
    capraz_dogrulama_resnet50_egitimi()
