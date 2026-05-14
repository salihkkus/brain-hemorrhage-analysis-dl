import os
import shutil
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
HAM_IMG_FOLDER = "data/ham_veri/head_ct"
HAM_CSV_YOLU = "data/ham_veri/labels.csv"
ISLENMIS_FOLDER = "data/islenmis_veri"
TEST_DIR = os.path.join(ISLENMIS_FOLDER, "test")
def klasorleri_temizle_ve_olustur():
    print("--- TEMİZLİK: Eski veriler siliniyor ---")
    if os.path.exists(ISLENMIS_FOLDER):
        shutil.rmtree(ISLENMIS_FOLDER)
    os.makedirs(ISLENMIS_FOLDER, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
def goruntuleri_kopyala(df, hedef_klasor):
    list_data = []
    for index, row in df.iterrows():
        img_id = int(row['id'])
        label = int(row['hemorrhage'])
        img_path = os.path.join(HAM_IMG_FOLDER, f"{img_id:03d}.png")
        if not os.path.exists(img_path):
            continue
        dosya_adi = f"{img_id:03d}.png"
        hedef_yol = os.path.join(hedef_klasor, dosya_adi)
        shutil.copy(img_path, hedef_yol)
        list_data.append({'filename': dosya_adi, 'label': label})
    return pd.DataFrame(list_data)
def goruntuleri_isle_ve_kaydet(df, hedef_klasor, arttirma_katsayisi=12):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    list_data = []
    for index, row in df.iterrows():
        img_id = int(row['id'])
        label = int(row['hemorrhage'])
        img_path = os.path.join(HAM_IMG_FOLDER, f"{img_id:03d}.png")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orijinal_adi = f"{img_id:03d}_orj.png"
        orj_hedef_yol = os.path.join(hedef_klasor, orijinal_adi)
        cv2.imwrite(orj_hedef_yol, img)
        list_data.append({'filename': orijinal_adi, 'label': label})
        x = img_to_array(img_rgb)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            uretilmis_img = cv2.cvtColor(batch[0].astype('uint8'), cv2.COLOR_RGB2BGR)
            uretilen_adi = f"{img_id:03d}_aug_{i}.png"
            uretilen_yol = os.path.join(hedef_klasor, uretilen_adi)
            cv2.imwrite(uretilen_yol, uretilmis_img)
            list_data.append({'filename': uretilen_adi, 'label': label})
            i += 1
            if i >= arttirma_katsayisi:
                break
    return pd.DataFrame(list_data)
def tam_is_akisi_calistir():
    klasorleri_temizle_ve_olustur()
    print("\n--- ADIM 1: Veritabanı Okunuyor ---")
    ham_df = pd.read_csv(HAM_CSV_YOLU)
    ham_df.columns = ham_df.columns.str.strip()
    print(f"Toplam Orijinal Görüntü Sayısı: {len(ham_df)}")
    print("NOT: Hasta (Patient ID) meta verisi olmadığı için Grup Split yapılması imkânsızdır.")
    print("\n--- ADIM 2: Bağımsız Test Seti Ayrılıyor ---")
    train_val_df, test_df = train_test_split(ham_df, test_size=0.15, random_state=42, stratify=ham_df['hemorrhage'])
    test_son_df = goruntuleri_kopyala(test_df, TEST_DIR)
    test_son_df.to_csv(os.path.join(ISLENMIS_FOLDER, "test_labels.csv"), index=False)
    print(f"✅ Sabit Test Seti: {len(test_son_df)} resim ayrıldı ve kilitlendi.")
    print("\n--- ADIM 3: 5-Fold Stratified Cross Validation İşlemi ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_val_df = train_val_df.reset_index(drop=True)
    X = train_val_df['id'].values
    y = train_val_df['hemorrhage'].values
    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n>> Fold {fold} işleniyor (K-Fold CV)...")
        fold_train_df = train_val_df.iloc[train_index]
        fold_val_df = train_val_df.iloc[val_index]
        fold_klasor = os.path.join(ISLENMIS_FOLDER, f"fold_{fold}")
        fold_train_klasor = os.path.join(fold_klasor, "train")
        fold_val_klasor = os.path.join(fold_klasor, "val")
        os.makedirs(fold_train_klasor, exist_ok=True)
        os.makedirs(fold_val_klasor, exist_ok=True)
        val_son_df = goruntuleri_kopyala(fold_val_df, fold_val_klasor)
        val_son_df.to_csv(os.path.join(fold_klasor, "val_labels.csv"), index=False)
        train_son_df = goruntuleri_isle_ve_kaydet(fold_train_df, fold_train_klasor, arttirma_katsayisi=10)
        train_son_df.to_csv(os.path.join(fold_klasor, "train_labels.csv"), index=False)
        print(f"Fold {fold} | Train: {len(train_son_df)} | Val: {len(val_son_df)}")
        fold += 1
    print(f"\n🎯 5-Fold Cross Validation veri mimarisi diskte oluşturuldu!")
if __name__ == "__main__":
    tam_is_akisi_calistir()
