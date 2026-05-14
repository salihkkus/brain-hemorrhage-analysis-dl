import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
MODEL_YOLU = "modeller/pretrained_model.h5"
HAM_KLASOR = "data/ham_veri/head_ct"
HAM_CSV = "data/ham_veri/labels.csv"
CIKTI_CSV = "rapor_gorselleri/feature.csv"
IMG_SIZE = (224, 224)
def feature_modelini_olustur(tam_model):
    hedef_katman = None
    for katman in reversed(tam_model.layers):
        if "dense" in katman.name.lower() and katman.output.shape[-1] != 1:
            hedef_katman = katman
            break
    if hedef_katman is None:
        raise RuntimeError("Penultimate Dense katmani bulunamadi.")
    print(f"Feature katmani: {hedef_katman.name} (cikti boyutu: {hedef_katman.output.shape[-1]})")
    return Model(inputs=tam_model.input, outputs=hedef_katman.output)
def goruntuyu_yukle(yol):
    img = cv2.imread(yol)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return img
def main():
    if not os.path.exists(MODEL_YOLU):
        raise FileNotFoundError(f"Model yok: {MODEL_YOLU}")
    if not os.path.exists(HAM_CSV):
        raise FileNotFoundError(f"Ham CSV yok: {HAM_CSV}")
    print("Model yukleniyor...")
    tam_model = load_model(MODEL_YOLU, compile=False)
    feature_model = feature_modelini_olustur(tam_model)
    feature_boyutu = feature_model.output.shape[-1]
    df = pd.read_csv(HAM_CSV)
    df.columns = df.columns.str.strip()
    print(f"Ham veri: {len(df)} goruntu")
    X, kullanilan = [], []
    for _, satir in df.iterrows():
        img_id = int(satir["id"])
        dosya_adi = f"{img_id:03d}.png"
        yol = os.path.join(HAM_KLASOR, dosya_adi)
        img = goruntuyu_yukle(yol)
        if img is None:
            print(f"  Atlandi (okunamadi): {dosya_adi}")
            continue
        X.append(img)
        kullanilan.append({"filename": dosya_adi, "label": int(satir["hemorrhage"])})
    if not X:
        raise RuntimeError("Hicbir goruntu yuklenemedi.")
    X = np.stack(X, axis=0)
    print(f"Feature cikariliyor ({X.shape[0]} goruntu)...")
    features = feature_model.predict(X, batch_size=16, verbose=1)
    sutunlar = [f"f{i}" for i in range(feature_boyutu)]
    feature_df = pd.DataFrame(features, columns=sutunlar)
    meta_df = pd.DataFrame(kullanilan)
    sonuc = pd.concat([meta_df, feature_df], axis=1)
    os.makedirs(os.path.dirname(CIKTI_CSV), exist_ok=True)
    sonuc.to_csv(CIKTI_CSV, index=False)
    print(f"Kaydedildi: {CIKTI_CSV}  (satir={len(sonuc)}, sutun={sonuc.shape[1]})")
if __name__ == "__main__":
    main()
