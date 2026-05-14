import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
st.set_page_config(
    page_title="Nöroloji - Beyin Kanama Analiz Sistemi",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    /* Streamlit varsayılan arayüz öğelerini gizle (header'ı tamamen gizleme — sidebar açma butonu orada) */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .stAppDeployButton { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }
    /* Header'ı şeffaflaştır ama yüksekliği koru — sidebar açma butonu burada barınır */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 999990 !important;
    }
    /* Sidebar kapalıyken görünen geri açma butonunu belirginleştir */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        z-index: 999999 !important;
        background: #0d2137 !important;
        border: 1px solid #1e4976 !important;
        border-radius: 8px !important;
        padding: 0.3rem !important;
        margin: 0.4rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="collapsedControl"] button svg {
        color: #e6f1ff !important;
        fill: #e6f1ff !important;
    }
    .stApp {
        background: linear-gradient(160deg, #0a192f 0%, #112240 40%, #0a192f 100%);
    }
    .block-container {
        padding-top: 1.5rem !important;
    }
    /* Kurumsal üst şerit (ince) */
    .institutional-strip {
        background: #ffffff;
        border-radius: 12px 12px 0 0;
        padding: 0.9rem 1.6rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        border-bottom: 3px solid #c8102e;
    }
    .institutional-strip img {
        height: 52px;
        width: auto;
    }
    .institutional-strip .inst-text {
        flex: 1;
    }
    .institutional-strip .inst-name {
        color: #0a1f3d;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .institutional-strip .inst-sub {
        color: #5a6e8a;
        font-size: 0.72rem;
        margin: 0.15rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }
    .institutional-strip .inst-meta {
        text-align: right;
        color: #5a6e8a;
        font-size: 0.7rem;
        line-height: 1.5;
    }
    .institutional-strip .inst-meta b {
        color: #0a1f3d;
    }
    /* Ana sistem banner'ı (baskın) */
    .hospital-banner {
        background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 50%, #0d2137 100%);
        border: 1px solid #1e4976;
        border-radius: 0 0 16px 16px;
        padding: 2rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.4rem;
        position: relative;
    }
    .hospital-banner::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #c8102e, #1a3a5c);
        border-radius: 0 0 0 16px;
    }
    .hospital-banner .logo {
        font-size: 3rem;
        line-height: 1;
    }
    .hospital-banner h1 {
        color: #e6f1ff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hospital-banner p {
        color: #8892b0;
        font-size: 0.85rem;
        margin: 0.35rem 0 0 0;
    }
    .hospital-banner .module-badge {
        margin-left: auto;
        background: rgba(200, 16, 46, 0.12);
        border: 1px solid rgba(200, 16, 46, 0.5);
        color: #ff6b80;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    /* Hasta bilgi kartı */
    .patient-card {
        background: #112240;
        border: 1px solid #1e4976;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .patient-card .label {
        color: #4a7fb5;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .patient-card .value {
        color: #ccd6f6;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }
    /* Sonuç kartları */
    .result-card {
        border-radius: 14px;
        padding: 1.8rem;
        margin: 0.8rem 0;
        border: 1px solid transparent;
    }
    .result-critical {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1117 100%);
        border-color: #e74c3c;
        box-shadow: 0 0 25px rgba(231, 76, 60, 0.15);
    }
    .result-normal {
        background: linear-gradient(135deg, #0a1a0f 0%, #112d17 100%);
        border-color: #2ecc71;
        box-shadow: 0 0 25px rgba(46, 204, 113, 0.12);
    }
    .result-card .status-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-critical {
        background: #e74c3c;
        color: white;
    }
    .badge-normal {
        background: #2ecc71;
        color: white;
    }
    .result-card h3 {
        color: #e6f1ff;
        font-size: 1.15rem;
        margin: 0.8rem 0 0.5rem 0;
        font-weight: 600;
    }
    .result-card .yorum {
        color: #a8b2d1;
        font-size: 0.9rem;
        line-height: 1.65;
    }
    .result-card .guven-bar-bg {
        background: #1a2744;
        border-radius: 8px;
        height: 8px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    .guven-bar-fill-red {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #e74c3c, #ff6b6b);
    }
    .guven-bar-fill-green {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #2ecc71, #6bffb8);
    }
    .guven-label {
        color: #4a7fb5;
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }
    /* Konsensüs kutusu */
    .consensus-box {
        background: #0d2137;
        border: 2px solid #1e4976;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .consensus-box h2 {
        color: #e6f1ff;
        font-size: 1.1rem;
        margin: 0 0 0.6rem 0;
    }
    .consensus-box .yorum {
        color: #a8b2d1;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    /* Bekleme ekranı */
    .waiting-screen {
        text-align: center;
        padding: 3rem 1.5rem;
        color: #4a7fb5;
    }
    .waiting-screen .icon { font-size: 3rem; margin-bottom: 1rem; }
    .waiting-screen h3 { color: #8892b0; font-weight: 500; font-size: 1.1rem; }
    .waiting-screen p { color: #4a7fb5; font-size: 0.85rem; line-height: 1.5; }
    /* Footer */
    .footer-bar {
        text-align: center;
        padding: 1.2rem;
        color: #384766;
        font-size: 0.75rem;
        margin-top: 2rem;
        border-top: 1px solid #1e3050;
        line-height: 1.7;
    }
    .footer-bar .inst-line {
        color: #5a6e8a;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .footer-bar .sub-line {
        color: #384766;
        font-size: 0.72rem;
    }
    /* Sidebar — varsayılan olarak görünür ve genişliği sabit */
    section[data-testid="stSidebar"] {
        background: #0d1b2a;
        border-right: 1px solid #1e3050;
        min-width: 300px !important;
        max-width: 360px !important;
        width: 320px !important;
        visibility: visible !important;
        transform: none !important;
        margin-left: 0 !important;
    }
    section[data-testid="stSidebar"] > div {
        visibility: visible !important;
    }
    .sidebar-header {
        color: #8892b0;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    /* Teknik detay tablosu */
    .tech-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .tech-table td {
        padding: 0.45rem 0;
        font-size: 0.8rem;
        border-bottom: 1px solid #1e3050;
    }
    .tech-table .label-cell { color: #4a7fb5; width: 40%; }
    .tech-table .value-cell { color: #ccd6f6; }
    /* Olasılık dağılım paneli */
    .prob-panel {
        background: #0d1b2a;
        border: 1px solid #1e3050;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
    }
    .prob-panel .panel-title {
        color: #4a7fb5;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.6rem;
    }
    .prob-row .prob-label {
        color: #a8b2d1;
        font-size: 0.8rem;
        min-width: 80px;
        flex-shrink: 0;
    }
    .prob-row .prob-bar-bg {
        flex: 1;
        background: #1a2744;
        border-radius: 6px;
        height: 22px;
        overflow: hidden;
        position: relative;
    }
    .prob-row .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
        min-width: 40px;
        transition: width 0.5s ease;
    }
    .prob-fill-red {
        background: linear-gradient(90deg, #c0392b, #e74c3c);
    }
    .prob-fill-green {
        background: linear-gradient(90deg, #27ae60, #2ecc71);
    }
    .prob-row .prob-value {
        color: #ccd6f6;
        font-size: 0.82rem;
        font-weight: 600;
        min-width: 52px;
        text-align: right;
    }
    .seviye-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    .seviye-cok-yuksek { background: #1a472a; color: #2ecc71; border: 1px solid #2ecc71; }
    .seviye-yuksek { background: #1a3d1a; color: #82e0aa; border: 1px solid #82e0aa; }
    .seviye-orta { background: #3d3a1a; color: #f0c040; border: 1px solid #f0c040; }
    .seviye-dusuk { background: #3d1a1a; color: #e74c3c; border: 1px solid #e74c3c; }
    /* Uyarı notu */
    .disclaimer {
        background: #1a1a2e;
        border-left: 3px solid #e6b800;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        color: #a8b2d1;
        font-size: 0.78rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="institutional-strip">
    <img src="https://duzce.edu.tr/assets/theme/images/logo.png" alt="Düzce Üniversitesi" />
    <div class="inst-text">
        <p class="inst-name">T.C. Düzce Üniversitesi Sağlık Uygulama ve Araştırma Merkezi</p>
        <p class="inst-sub">Dahili Bilimler · Nöroloji Anabilim Dalı</p>
    </div>
    <div class="inst-meta">
        <b>İletişim:</b> 0850 259 81 81<br>
        hastane.duzce.edu.tr
    </div>
</div>
<div class="hospital-banner">
    <div>
        <h1>Beyin Kanama Analiz Sistemi</h1>
        <p>Yapay Zekâ Destekli Radyolojik Karar Destek Platformu &nbsp;|&nbsp; v2.0</p>
    </div>
    <span class="module-badge">● Klinik Modül Aktif</span>
</div>
""", unsafe_allow_html=True)
def _pretrained_model_yeniden_kur(model_yolu):
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
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights(model_yolu, by_name=False)
    return model
@st.cache_resource
def model_yukle(model_yolu):
    if not os.path.exists(model_yolu):
        return None
    try:
        return tf.keras.models.load_model(model_yolu, compile=False)
    except Exception:
        return _pretrained_model_yeniden_kur(model_yolu)
PROJE_KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OZGUN_MODEL_YOLU = os.path.join(PROJE_KOK, "modeller", "ozgun_cnn_modeli.h5")
PRETRAINED_MODEL_YOLU = os.path.join(PROJE_KOK, "modeller", "pretrained_model.h5")
ozgun_model = model_yukle(OZGUN_MODEL_YOLU)
pretrained_model = model_yukle(PRETRAINED_MODEL_YOLU)
def goruntu_isle(uploaded_file):
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_array, (224, 224)).astype(np.float32)
    img_cnn = np.expand_dims(img_resized / 255.0, axis=0)
    img_resnet = np.expand_dims(resnet_preprocess(img_resized.copy()), axis=0)
    return img_cnn, img_resnet, image
def tahmin_yap(model, img_batch, temperature=2.5, logit_clip=None):
    if model is None:
        return None, None, None, None
    raw_pred = float(model.predict(img_batch, verbose=0)[0][0])
    raw_pred_clipped = np.clip(raw_pred, 1e-7, 1 - 1e-7)
    logit = np.log(raw_pred_clipped / (1 - raw_pred_clipped))
    if logit_clip is not None:
        logit = float(np.clip(logit, -logit_clip, logit_clip))
    scaled_pred = 1.0 / (1.0 + np.exp(-logit / temperature))
    sinif = "Kanamalı" if scaled_pred > 0.5 else "Normal"
    guven = float(scaled_pred if scaled_pred > 0.5 else (1 - scaled_pred))
    p_kanama = float(scaled_pred)
    p_normal = float(1 - scaled_pred)
    return sinif, guven, p_kanama, p_normal
def insansi_yorum(sinif, guven, model_adi):
    if sinif == "Kanamalı":
        if guven >= 0.95:
            return (f"{model_adi} analizi, yüklenen BT görüntüsünde "
                    f"<b>kanama ile uyumlu bulguya yüksek güvenle işaret etmektedir</b> "
                    f"(%{guven*100:.1f} kesinlik). İntrakraniyal hemoraji şüpheleniyor; "
                    f"nörolojik muayene ve acil radyolojik doğrulama önerilir.")
        elif guven >= 0.85:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>kanama lehine anlamlı bulgular saptamıştır</b> "
                    f"(%{guven*100:.1f} kesinlik). Hastanın klinik tablosuyla birlikte "
                    f"değerlendirilmesi ve ileri tetkik planlanması tavsiye edilir.")
        elif guven >= 0.70:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>kanama olabileceğine dair orta düzeyde bulgu</b> tespit etmiştir "
                    f"(%{guven*100:.1f} kesinlik). Kesin tanı için uzman radyolog "
                    f"değerlendirmesi gereklidir.")
        else:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>kanama yönünde zayıf bir işaret</b> algıladı "
                    f"(%{guven*100:.1f} kesinlik). Bulgu belirsiz olup, klinik korelasyon "
                    f"ve kontrol görüntüleme önerilir.")
    else:
        if guven >= 0.95:
            return (f"{model_adi} analizi, yüklenen BT görüntüsünde "
                    f"<b>kanama ile uyumlu herhangi bir bulguya rastlamamıştır</b> "
                    f"(%{guven*100:.1f} kesinlik). Görüntüler normal sınırlar içinde "
                    f"değerlendirilmiştir.")
        elif guven >= 0.85:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>belirgin bir kanama bulgusuna rastlamamıştır</b> "
                    f"(%{guven*100:.1f} kesinlik). Ancak klinik şüphede devam ediyorsa "
                    f"kontrol görüntüleme planlanabilir.")
        elif guven >= 0.70:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>kanamaya işaret eden belirgin bir bulgu saptamamıştır</b> "
                    f"(%{guven*100:.1f} kesinlik), ancak kesinlik orta düzeydedir. "
                    f"Klinik değerlendirme ile birlikte yorumlanmalıdır.")
        else:
            return (f"{model_adi} analizi, görüntüde "
                    f"<b>net bir sonuca ulaşılamamıştır</b> "
                    f"(%{guven*100:.1f} kesinlik). Düşük güven seviyesi nedeniyle "
                    f"uzman görüşü ile ileri tetkik kesinlikle önerilir.")
def guven_seviyesi(guven):
    if guven >= 0.92:
        return "Çok Yüksek", "seviye-cok-yuksek"
    elif guven >= 0.80:
        return "Yüksek", "seviye-yuksek"
    elif guven >= 0.65:
        return "Orta", "seviye-orta"
    else:
        return "Düşük", "seviye-dusuk"
def olasilik_paneli_html(p_kanama, p_normal, guven):
    seviye_text, seviye_cls = guven_seviyesi(guven)
    return (
f'<div class="prob-panel">'
f'<div class="panel-title">Detaylı Olasılık Dağılımı</div>'
f'<div class="prob-row">'
f'<div class="prob-label">Kanamalı</div>'
f'<div class="prob-bar-bg">'
f'<div class="prob-bar-fill prob-fill-red" style="width:{p_kanama*100:.1f}%">'
f'{("%" + format(p_kanama*100, ".1f")) if p_kanama > 0.12 else ""}'
f'</div></div>'
f'<div class="prob-value">%{p_kanama*100:.1f}</div>'
f'</div>'
f'<div class="prob-row">'
f'<div class="prob-label">Normal</div>'
f'<div class="prob-bar-bg">'
f'<div class="prob-bar-fill prob-fill-green" style="width:{p_normal*100:.1f}%">'
f'{("%" + format(p_normal*100, ".1f")) if p_normal > 0.12 else ""}'
f'</div></div>'
f'<div class="prob-value">%{p_normal*100:.1f}</div>'
f'</div>'
f'<div style="display:flex; align-items:center; margin-top:0.6rem;">'
f'<span class="seviye-badge {seviye_cls}">Güven Seviyesi: {seviye_text}</span>'
f'</div></div>'
    )
def konsensus_yorumu(sinif1, guven1, sinif2, guven2):
    if sinif1 == sinif2:
        ort_guven = (guven1 + guven2) / 2
        if sinif1 == "Kanamalı":
            if ort_guven >= 0.85:
                return ("alert", "Her iki yapay zekâ modeli de bu görüntüde "
                        "<b>kanama bulgusunda hemfikirdir</b>. "
                        f"Ortalama güven oranı %{ort_guven*100:.1f} olup, "
                        "bulgular acil nörolojik değerlendirme gerektirmektedir. "
                        "Uzman onayıyla birlikte tedavi protokolünün başlatılması önerilir.")
            else:
                return ("alert", "Her iki model de kanama yönünde işaret vermekte, "
                        f"ancak ortalama güven oranı (%{ort_guven*100:.1f}) orta düzeydedir. "
                        "Kesin tanı için uzman radyolog değerlendirmesi önerilir.")
        else:
            if ort_guven >= 0.85:
                return ("safe", "Her iki yapay zekâ modeli de bu görüntüde "
                        "<b>kanama bulgusuna rastlamamıştır</b>. "
                        f"Ortalama güven oranı %{ort_guven*100:.1f} olup, "
                        "görüntüler normal sınırlar içinde değerlendirilmiştir. "
                        "Rutin takip planı yeterli görünmektedir.")
            else:
                return ("safe", "Her iki model de kanama saptamamıştır, "
                        f"ancak güven oranı (%{ort_guven*100:.1f}) kesin değildir. "
                        "Klinik şüphede devam ediyorsa kontrol BT planlanması uygun olabilir.")
    else:
        return ("uncertain", "Yapay zekâ modelleri <b>farklı sonuçlara ulaşmıştır</b>. "
                f"Bir model kanama tespit ederken, diğeri normal değerlendirmiştir. "
                "Bu durum, görüntünün sınırda bulgular taşıyabileceğine işaret etmektedir. "
                "<b>Mutlaka uzman radyolog tarafından incelenmeli</b> ve "
                "klinik bulgularla birlikte değerlendirilmelidir.")
with st.sidebar:
    st.markdown('<p class="sidebar-header">Sistem Bilgileri</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="patient-card">
        <div class="label">Analiz Tarihi</div>
        <div class="value">{datetime.now().strftime('%d.%m.%Y - %H:%M')}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p class="sidebar-header">Model Performansları</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="patient-card">
        <table class="tech-table">
            <tr><td class="label-cell" colspan="2" style="color:#e6f1ff; font-weight:600;">Özgün CNN</td></tr>
            <tr><td class="label-cell">Doğruluk</td><td class="value-cell">%89.2</td></tr>
            <tr><td class="label-cell">Kesinlik</td><td class="value-cell">%88.8</td></tr>
            <tr><td class="label-cell">Duyarlılık</td><td class="value-cell">%89.7</td></tr>
            <tr><td class="label-cell">F1-Skor</td><td class="value-cell">%89.3</td></tr>
        </table>
    </div>
    <div class="patient-card">
        <table class="tech-table">
            <tr><td class="label-cell" colspan="2" style="color:#e6f1ff; font-weight:600;">ResNet50 (Transfer Learning)</td></tr>
            <tr><td class="label-cell">Doğruluk</td><td class="value-cell">%85.9</td></tr>
            <tr><td class="label-cell">Kesinlik</td><td class="value-cell">%82.1</td></tr>
            <tr><td class="label-cell">Duyarlılık</td><td class="value-cell">%91.8</td></tr>
            <tr><td class="label-cell">F1-Skor</td><td class="value-cell">%86.7</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        <b>Önemli Uyarı:</b> Bu sistem yalnızca karar destek amacıyla geliştirilmiştir.
        Kesin tanı mutlaka uzman hekim tarafından konulmalıdır. Yapay zekâ çıktıları
        tek başına klinik karar almak için yeterli değildir.
    </div>
    """, unsafe_allow_html=True)
col_img, col_result = st.columns([1, 1.3], gap="large")
with col_img:
    st.markdown('<p class="sidebar-header">Görüntü Yükleme</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Beyin BT görüntüsü seçin",
        type=["png", "jpg", "jpeg"],
        help="Bilgisayarınızdan bir beyin BT görüntüsünü PNG veya JPG olarak yükleyin.",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        img_cnn, img_resnet, original_image = goruntu_isle(uploaded_file)
        st.image(original_image, caption="Yüklenen BT Görüntüsü", use_container_width=True)
        st.markdown(f"""
        <div class="patient-card">
            <div class="label">Dosya Bilgisi</div>
            <div class="value">{uploaded_file.name} &nbsp;|&nbsp; {uploaded_file.size / 1024:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)
with col_result:
    if uploaded_file is not None:
        st.markdown('<p class="sidebar-header">Yapay Zekâ Değerlendirme Raporu</p>',
                    unsafe_allow_html=True)
        sonuclar = []
        if ozgun_model is not None:
            sinif1, guven1, pk1, pn1 = tahmin_yap(ozgun_model, img_cnn)
            yorum1 = insansi_yorum(sinif1, guven1, "Özgün CNN (3 katmanlı evrişim ağı)")
            css1 = "result-critical" if sinif1 == "Kanamalı" else "result-normal"
            badge1 = "badge-critical" if sinif1 == "Kanamalı" else "badge-normal"
            badge_text1 = "KANAMA TESPİT EDİLDİ" if sinif1 == "Kanamalı" else "NORMAL BULGU"
            sonuclar.append((sinif1, guven1))
            st.markdown(
f"""<div class="result-card {css1}">
<span class="status-badge {badge1}">{badge_text1}</span>
<h3>Model 1 — Özgün CNN</h3>
<div class="yorum">{yorum1}</div>
{olasilik_paneli_html(pk1, pn1, guven1)}
</div>""", unsafe_allow_html=True)
        if pretrained_model is not None:
            sinif2, guven2, pk2, pn2 = tahmin_yap(pretrained_model, img_resnet, temperature=2.5, logit_clip=2.5)
            yorum2 = insansi_yorum(sinif2, guven2, "ResNet50 (transfer öğrenme modeli)")
            css2 = "result-critical" if sinif2 == "Kanamalı" else "result-normal"
            badge2 = "badge-critical" if sinif2 == "Kanamalı" else "badge-normal"
            badge_text2 = "KANAMA TESPİT EDİLDİ" if sinif2 == "Kanamalı" else "NORMAL BULGU"
            sonuclar.append((sinif2, guven2))
            st.markdown(
f"""<div class="result-card {css2}">
<span class="status-badge {badge2}">{badge_text2}</span>
<h3>Model 2 — ResNet50 (Pre-trained)</h3>
<div class="yorum">{yorum2}</div>
{olasilik_paneli_html(pk2, pn2, guven2)}
</div>""", unsafe_allow_html=True)
        if len(sonuclar) == 2:
            k_tip, k_yorum = konsensus_yorumu(sinif1, guven1, sinif2, guven2)
            if k_tip == "alert":
                border_color = "#e74c3c"
                icon = "🔴"
            elif k_tip == "safe":
                border_color = "#2ecc71"
                icon = "🟢"
            else:
                border_color = "#e6b800"
                icon = "🟡"
            st.markdown(f"""
            <div class="consensus-box" style="border-color: {border_color};">
                <h2>{icon} Genel Değerlendirme (Model Konsensüs)</h2>
                <div class="yorum">{k_yorum}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="waiting-screen">
            <div class="icon">🩻</div>
            <h3>BT görüntüsü bekleniyor...</h3>
            <p>
                Sol panelden bir beyin bilgisayarlı tomografi görüntüsünü yükleyin.<br>
                Sistem, görüntüyü iki farklı yapay zekâ modeliyle analiz ederek<br>
                kanama olup olmadığını değerlendirip size insansı bir rapor sunacaktır.
            </p>
        </div>
        """, unsafe_allow_html=True)
st.markdown(f"""
<div class="footer-bar">
    <div class="inst-line">🏥 T.C. Düzce Üniversitesi Sağlık Uygulama ve Araştırma Merkezi — Nöroloji Anabilim Dalı</div>
    <div class="sub-line">
        Beyin Kanama Analiz Sistemi · Yapay Zekâ Karar Destek Modülü &nbsp;|&nbsp;
        © {datetime.now().strftime('%Y')} Düzce Üniversitesi Hastanesi
    </div>
</div>
""", unsafe_allow_html=True)
