import os
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

ISLENMIS_FOLDER = "data/islenmis_veri"
MODEL_KAYIT_YOLU = "modeller"
IMG_SIZE = (224, 224)

os.makedirs(MODEL_KAYIT_YOLU, exist_ok=True)

def seti_yukle(fold_no, set_adi):
    """Belirli bir fold'un içindeki train veya val setini okur."""
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(row['label'])
            
    # CNN için [0, 1] arası scale işlemi burada yapılıyor
    return np.array(X) / 255.0, np.array(y)

def ozgun_cnn_tasarla(learning_rate=1e-3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def capraz_dogrulama_egitimi():
    print("\n--- 5-FOLD CROSS VALIDATION (ÖZGÜN CNN) BAŞLIYOR ---")
    katlama_skorlari = []
    tum_history = []
    en_dusuk_val_loss = float('inf')
    en_iyi_model_fold = 1
    
    for fold in range(1, 6):
        print(f"\n================ FOLD {fold} ================")
        X_train, y_train = seti_yukle(fold, "train")
        X_val, y_val = seti_yukle(fold, "val")
        
        print(f"Bölüm: Train={len(X_train)} (Artırılmış) | Val={len(X_val)} (Orijinal)")
        
        model = ozgun_cnn_tasarla()
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # En iyi epoch'un özelliklerini al
        try:
            best_epoch_idx = early_stop.best_epoch if early_stop.best_epoch is not None else len(history.history['val_loss']) - 1
        except AttributeError:
            best_epoch_idx = len(history.history['val_loss']) - 1
            
        val_acc = history.history['val_accuracy'][best_epoch_idx]
        val_loss = history.history['val_loss'][best_epoch_idx]
        katlama_skorlari.append(val_acc)
        tum_history.append(history.history)

        print(f"✅ Fold {fold} Başarısı: %{val_acc*100:.2f} | Val Loss: {val_loss:.4f}")

        # En düşük val_loss'a sahip Fold'un ağırlıklarını kaydet (daha genellenebilir)
        if val_loss < en_dusuk_val_loss:
            en_dusuk_val_loss = val_loss
            en_iyi_model_fold = fold
            model.save(os.path.join(MODEL_KAYIT_YOLU, "ozgun_cnn_modeli.h5"))
            
    # Sonuç Raporlama
    print("\n================ CROSS VALIDATION ÖZETİ ================")
    print(f"Ayrı Ayrı Katlama Başarıları : {[round(s, 3) for s in katlama_skorlari]}")
    print(f"🌟 ORTALAMA MODEL DOĞRULUĞU : %{np.mean(katlama_skorlari)*100:.2f}")
    print(f"💾 En düşük val_loss'lu ağırlık Fold {en_iyi_model_fold}'den diske kaydedildi (val_loss={en_dusuk_val_loss:.4f}).")
    
    # GUI Grafikleri için 5 Fold'un ortalama history'sini hesapla
    avg_hist = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    min_epochs = min([len(h['accuracy']) for h in tum_history])
    for i in range(min_epochs):
        avg_hist['accuracy'].append( float(np.mean([h['accuracy'][i] for h in tum_history])) )
        avg_hist['val_accuracy'].append( float(np.mean([h['val_accuracy'][i] for h in tum_history])) )
        avg_hist['loss'].append( float(np.mean([h['loss'][i] for h in tum_history])) )
        avg_hist['val_loss'].append( float(np.mean([h['val_loss'][i] for h in tum_history])) )
        
    history_path = os.path.join(MODEL_KAYIT_YOLU, "ozgun_history.json")
    with open(history_path, 'w') as f:
        json.dump(avg_hist, f)
    print(f"📊 Ortalama eğitim grafiği kaydedildi: {history_path}")

if __name__ == "__main__":
    capraz_dogrulama_egitimi()