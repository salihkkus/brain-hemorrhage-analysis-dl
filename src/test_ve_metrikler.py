import os
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
ISLENMIS_FOLDER = "data/islenmis_veri"
TEST_FOLDER = os.path.join(ISLENMIS_FOLDER, "test")
TEST_CSV_PATH = os.path.join(ISLENMIS_FOLDER, "test_labels.csv")
MODEL_KAYIT_YOLU = "modeller"
GORSEL_CIKTI_YOLU = "rapor_gorselleri"
IMG_SIZE = (224, 224)
os.makedirs(GORSEL_CIKTI_YOLU, exist_ok=True)
SINIF_ADLARI = ['Normal (0)', 'Kanamalı (1)']
def test_verisini_yukle():
    print("\n--- Test Verisi Yükleniyor ---")
    df = pd.read_csv(TEST_CSV_PATH)
    images_raw, y = [], []
    for index, row in df.iterrows():
        img_path = os.path.join(TEST_FOLDER, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_raw.append(img.astype(np.float32))
            y.append(row['label'])
    images_raw = np.array(images_raw)
    y_test = np.array(y)
    X_cnn_test = images_raw / 255.0
    X_res_test = resnet_preprocess(images_raw.copy())
    print(f"✅ Test Görüntü Sayısı: {len(y_test)}")
    return X_cnn_test, X_res_test, y_test
def confusion_matrix_ciz(y_true, y_pred, model_adi, dosya_adi):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SINIF_ADLARI,
                yticklabels=SINIF_ADLARI)
    plt.title(f'{model_adi} — Hata Matrisi (Confusion Matrix)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    yol = os.path.join(GORSEL_CIKTI_YOLU, dosya_adi)
    plt.savefig(yol, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix kaydedildi: {yol}")
def egitim_grafigi_ciz(history_dict, model_adi, dosya_adi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_dict['accuracy'], label='Eğitim', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Doğrulama', linewidth=2)
    ax1.set_title(f'{model_adi} — Doğruluk (Accuracy)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history_dict['loss'], label='Eğitim', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Doğrulama', linewidth=2)
    ax2.set_title(f'{model_adi} — Kayıp (Loss)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    yol = os.path.join(GORSEL_CIKTI_YOLU, dosya_adi)
    plt.savefig(yol, dpi=150)
    plt.close()
    print(f"✅ Eğitim grafiği kaydedildi: {yol}")
def model_degerlendir(model, X_test, y_test, model_adi):
    print(f"\n{'='*50}")
    print(f"📊 {model_adi} DEĞERLENDİRME SONUÇLARI")
    print(f"{'='*50}")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"   Accuracy  (Doğruluk)  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"   Precision (Kesinlik)  : {prec:.4f} ({prec*100:.1f}%)")
    print(f"   Recall    (Duyarlılık): {rec:.4f} ({rec*100:.1f}%)")
    print(f"   F1-Score              : {f1:.4f} ({f1*100:.1f}%)")
    print(f"\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=SINIF_ADLARI))
    return {
        'model': model_adi,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4)
    }, y_pred
def karsilastirma_tablosu_ciz(sonuclar):
    df = pd.DataFrame(sonuclar)
    fig, ax = plt.subplots(figsize=(10, 4))
    metrikler = ['accuracy', 'precision', 'recall', 'f1_score']
    metrik_labels = ['Accuracy\n(Doğruluk)', 'Precision\n(Kesinlik)', 'Recall\n(Duyarlılık)', 'F1-Score']
    x = np.arange(len(metrikler))
    width = 0.35
    vals1 = [df.iloc[0][m] for m in metrikler]
    vals2 = [df.iloc[1][m] for m in metrikler]
    bars1 = ax.bar(x - width/2, vals1, width, label=df.iloc[0]['model'], color='#4A90D9', edgecolor='white')
    bars2 = ax.bar(x + width/2, vals2, width, label=df.iloc[1]['model'], color='#E74C3C', edgecolor='white')
    for bar, val in zip(bars1, vals1):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylabel('Skor')
    ax.set_title('Model Performans Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(metrik_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    yol = os.path.join(GORSEL_CIKTI_YOLU, "model_karsilastirma.png")
    plt.savefig(yol, dpi=150)
    plt.close()
    print(f"\n✅ Karşılaştırma tablosu kaydedildi: {yol}")
if __name__ == "__main__":
    X_cnn_test, X_res_test, y_test = test_verisini_yukle()
    sonuclar = []
    ozgun_model_path = os.path.join(MODEL_KAYIT_YOLU, "ozgun_cnn_modeli.h5")
    if os.path.exists(ozgun_model_path):
        print(f"\n🔹 Özgün CNN yükleniyor: {ozgun_model_path}")
        ozgun_model = tf.keras.models.load_model(ozgun_model_path)
        ozgun_sonuc, ozgun_pred = model_degerlendir(ozgun_model, X_cnn_test, y_test, "Özgün CNN")
        sonuclar.append(ozgun_sonuc)
        confusion_matrix_ciz(y_test, ozgun_pred, "Özgün CNN", "ozgun_confusion_matrix.png")
    else:
        print(f"⚠️ Özgün model bulunamadı: {ozgun_model_path}")
    pretrained_model_path = os.path.join(MODEL_KAYIT_YOLU, "pretrained_model.h5")
    if os.path.exists(pretrained_model_path):
        print(f"\n🔹 Pre-trained model yükleniyor: {pretrained_model_path}")
        try:
            pretrained_model = tf.keras.models.load_model(pretrained_model_path)
        except Exception as e:
            print(f"   ⚠️ Tam yükleme başarısız ({type(e).__name__}), mimari yeniden kuruluyor...")
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras import layers, models
            base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            for layer in base.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
            inputs = layers.Input(shape=(224, 224, 3))
            x = base(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(1, activation='sigmoid')(x)
            pretrained_model = models.Model(inputs, outputs)
            pretrained_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            pretrained_model.load_weights(pretrained_model_path, by_name=False)
            print("   ✅ Ağırlıklar başarıyla yüklendi.")
        pretrained_sonuc, pretrained_pred = model_degerlendir(pretrained_model, X_res_test, y_test, "ResNet50 (Pre-trained)")
        sonuclar.append(pretrained_sonuc)
        confusion_matrix_ciz(y_test, pretrained_pred, "ResNet50 (Pre-trained)", "pretrained_confusion_matrix.png")
    else:
        print(f"⚠️ Pre-trained model bulunamadı: {pretrained_model_path}")
    ozgun_history_path = os.path.join(MODEL_KAYIT_YOLU, "ozgun_history.json")
    if os.path.exists(ozgun_history_path):
        with open(ozgun_history_path, 'r') as f:
            ozgun_hist = json.load(f)
        egitim_grafigi_ciz(ozgun_hist, "Özgün CNN", "egitim_grafikleri_ozgun.png")
    pretrained_history_path = os.path.join(MODEL_KAYIT_YOLU, "pretrained_history.json")
    if os.path.exists(pretrained_history_path):
        with open(pretrained_history_path, 'r') as f:
            pretrained_hist = json.load(f)
        egitim_grafigi_ciz(pretrained_hist, "ResNet50 (Pre-trained)", "egitim_grafikleri_pretrained.png")
    if len(sonuclar) == 2:
        karsilastirma_tablosu_ciz(sonuclar)
    elif len(sonuclar) == 1:
        print("\n⚠️ Sadece 1 model mevcut, karşılaştırma tablosu oluşturulamadı.")
    if sonuclar:
        sonuc_df = pd.DataFrame(sonuclar)
        sonuc_path = os.path.join(GORSEL_CIKTI_YOLU, "model_sonuclari.csv")
        sonuc_df.to_csv(sonuc_path, index=False)
        print(f"\n✅ Sonuçlar CSV olarak kaydedildi: {sonuc_path}")
    print("\n🚀 TÜM METRİKLER VE GÖRSELLER TAMAMLANDI!")
