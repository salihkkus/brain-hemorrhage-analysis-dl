import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

from model_egitimi import seti_yukle, ozgun_cnn_tasarla

MODEL_KAYIT_YOLU = "modeller"
GORSEL_CIKTI_YOLU = "rapor_gorselleri"
CSV_CIKTI_YOLU = os.path.join(GORSEL_CIKTI_YOLU, "hyperparameter_sonuclari.csv")
MD_CIKTI_YOLU = os.path.join(GORSEL_CIKTI_YOLU, "hyperparameter_sonuclari.md")
os.makedirs(MODEL_KAYIT_YOLU, exist_ok=True)
os.makedirs(GORSEL_CIKTI_YOLU, exist_ok=True)

LEARNING_RATES = [1e-3, 1e-4, 5e-5]
BATCH_SIZES = [16, 32]
PATIENCES = [3, 5, 7]
EPOCHS_MAX = 25


def grid_search():
    print("--- HIPERPARAMETRE OPTIMIZASYONU (GENISLETILMIS GRID SEARCH) ---")
    print(f"Grid: LR={LEARNING_RATES} x Batch={BATCH_SIZES} x Patience={PATIENCES}")
    print(f"Toplam deneme: {len(LEARNING_RATES) * len(BATCH_SIZES) * len(PATIENCES)}")
    print("Veri: Fold 1 (zaman tasarrufu icin tek fold uzerinde)")

    X_train, y_train = seti_yukle(1, "train")
    X_val, y_val = seti_yukle(1, "val")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    sonuclar = []
    en_iyi_val_acc = 0.0
    en_iyi_kombinasyon = None
    toplam_baslangic = time.time()

    deneme_no = 0
    toplam_deneme = len(LEARNING_RATES) * len(BATCH_SIZES) * len(PATIENCES)

    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            for pat in PATIENCES:
                deneme_no += 1
                print(f"\n[{deneme_no}/{toplam_deneme}] LR={lr} | Batch={bs} | Patience={pat}")
                t0 = time.time()

                tf.keras.backend.clear_session()
                model = ozgun_cnn_tasarla(learning_rate=lr)

                early_stop = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=pat,
                    restore_best_weights=True
                )

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS_MAX,
                    batch_size=bs,
                    callbacks=[early_stop],
                    verbose=2
                )

                try:
                    best_epoch_idx = early_stop.best_epoch if early_stop.best_epoch is not None else len(history.history['val_loss']) - 1
                except AttributeError:
                    best_epoch_idx = len(history.history['val_loss']) - 1

                best_val_acc = history.history['val_accuracy'][best_epoch_idx]
                best_val_loss = history.history['val_loss'][best_epoch_idx]
                sure = time.time() - t0

                print(f"-> Val Acc: {best_val_acc:.4f} | Val Loss: {best_val_loss:.4f} | Best Epoch: {best_epoch_idx+1} | Sure: {sure:.0f}s")

                sonuclar.append({
                    'Learning Rate': lr,
                    'Batch Size': bs,
                    'Patience': pat,
                    'Best Epoch': best_epoch_idx + 1,
                    'Validation Loss': round(float(best_val_loss), 4),
                    'Validation Accuracy': round(float(best_val_acc), 4),
                    'Sure (sn)': round(sure, 1)
                })

                if best_val_acc > en_iyi_val_acc:
                    en_iyi_val_acc = best_val_acc
                    en_iyi_kombinasyon = (lr, bs, pat)
                    print("** Yeni en iyi model! Kaydediliyor...")
                    model.save(os.path.join(MODEL_KAYIT_YOLU, "ozgun_cnn_en_iyi_model.h5"))

                df_ara = pd.DataFrame(sonuclar).sort_values(by='Validation Accuracy', ascending=False)
                df_ara.to_csv(CSV_CIKTI_YOLU, index=False)

    toplam_sure = time.time() - toplam_baslangic
    df_sonuc = pd.DataFrame(sonuclar).sort_values(by='Validation Accuracy', ascending=False)
    df_sonuc.to_csv(CSV_CIKTI_YOLU, index=False)

    md_metni = "# Hiperparametre Optimizasyonu Sonuclari\n\n"
    md_metni += f"Toplam deneme: {len(sonuclar)} | Toplam sure: {toplam_sure/60:.1f} dk\n\n"
    md_metni += df_sonuc.to_markdown(index=False)
    md_metni += f"\n\n**En iyi kombinasyon:** LR={en_iyi_kombinasyon[0]}, Batch={en_iyi_kombinasyon[1]}, Patience={en_iyi_kombinasyon[2]} -> Val Acc: {en_iyi_val_acc:.4f}\n"
    with open(MD_CIKTI_YOLU, "w") as f:
        f.write(md_metni)

    print("\n=== GRID SEARCH TAMAMLANDI ===")
    print(f"Toplam sure: {toplam_sure/60:.1f} dakika")
    print(f"CSV: {CSV_CIKTI_YOLU}")
    print(f"MD : {MD_CIKTI_YOLU}")
    print(f"En iyi: LR={en_iyi_kombinasyon[0]}, Batch={en_iyi_kombinasyon[1]}, Patience={en_iyi_kombinasyon[2]} -> Val Acc {en_iyi_val_acc:.4f}")
    print("\nOZET TABLO:\n")
    print(df_sonuc.to_markdown(index=False))


if __name__ == "__main__":
    grid_search()
