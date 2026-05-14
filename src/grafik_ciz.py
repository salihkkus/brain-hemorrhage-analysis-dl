import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
GORSEL_CIKTI_YOLU = "rapor_gorselleri"
os.makedirs(GORSEL_CIKTI_YOLU, exist_ok=True)
def kfold_dagilim_grafigi_ciz():
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    setler = ['Hold-Out Test\n(Bağımsız)', 'K-Fold Train\n(Fold Başına - Augmentli)', 'K-Fold Val\n(Fold Başına)']
    normal_sayilar = [15, 750, 17] 
    kanamali_sayilar = [15, 746, 17] 
    x = np.arange(len(setler))
    genislik = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - genislik/2, normal_sayilar, genislik, label='Normal (0)', color='#3498db')
    rects2 = ax.bar(x + genislik/2, kanamali_sayilar, genislik, label='Kanamalı (1)', color='#e74c3c')
    ax.set_ylabel('Görüntü Sayısı')
    ax.set_title('Yeni K-Fold Mimarisine Göre Veri Dağılımı Sınıf Özeti')
    ax.set_xticks(x)
    ax.set_xticklabels(setler)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    yol = os.path.join(GORSEL_CIKTI_YOLU, "islenmis_veri_dagilimi.png")
    plt.savefig(yol, dpi=300)
def ham_veri_dagilimi_ciz():
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    etiketler = ['Normal (0)', 'Kanamalı (1)']
    degerler = [100, 100]
    plt.pie(degerler, labels=etiketler, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    plt.title('Ham Veri Sınıf Dağılımı (Orijinal Kaggle Verisi)')
    yol = os.path.join(GORSEL_CIKTI_YOLU, "ham_veri_dagilimi.png")
    plt.savefig(yol, dpi=300)
if __name__ == "__main__":
    kfold_dagilim_grafigi_ciz()
    ham_veri_dagilimi_ciz()
