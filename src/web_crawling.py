import os
import re
import time
import urllib.request
import urllib.parse
from PIL import Image
INDIRME_KLASORU = "data/sunum_gorselleri"
os.makedirs(INDIRME_KLASORU, exist_ok=True)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}
IMG_PATTERN = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
SRCSET_PATTERN = re.compile(r'\bsrcset\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
def _http_get(url, timeout=15):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=timeout)
def _dosya_indir(url, kayit_yolu, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(kayit_yolu, "wb") as f:
        f.write(r.read())
def _en_buyuk_srcset(srcset_value):
    en_iyi = None
    en_iyi_w = -1
    for parca in srcset_value.split(","):
        parca = parca.strip()
        if not parca:
            continue
        bilesen = parca.split()
        u = bilesen[0]
        w = 0
        if len(bilesen) > 1 and bilesen[1].endswith("w"):
            try:
                w = int(bilesen[1][:-1])
            except ValueError:
                w = 0
        if w > en_iyi_w:
            en_iyi_w = w
            en_iyi = u
    return en_iyi
def _wikipedia_orijinal(url):
    if "/thumb/" in url:
        try:
            kok, son = url.split("/thumb/", 1)
            parcalar = son.split("/")
            if len(parcalar) >= 4:
                return f"{kok}/{'/'.join(parcalar[:3])}"
        except Exception:
            pass
    return url
def _gorsel_urllerini_topla(html, sayfa_url):
    urller = []
    for m in IMG_PATTERN.finditer(html):
        urller.append(m.group(1))
    for m in SRCSET_PATTERN.finditer(html):
        u = _en_buyuk_srcset(m.group(1))
        if u:
            urller.append(u)
    sonuc = []
    gorulen = set()
    for u in urller:
        tam = urllib.parse.urljoin(sayfa_url, u)
        if tam not in gorulen:
            gorulen.add(tam)
            sonuc.append(tam)
    return sonuc
UI_GURULTU_DESENLERI = [
    "semi-protection", "edit-icon", "wiki-letter", "ambox", "commons-logo",
    "magnify-clip", "padlock", "ozt-icon", "questionmark", "puzzle",
    "wikipedia/en/thumb/", "/static/", "/skins/", "poweredby",
]
KOMPOZIT_DESENLERI = [
    "large", "montage", "series", "panel", "composite", "collage",
    "grid", "multi", "stack", "_-_", "collection",
]
PATOLOJI_OTOPSI_DESENLERI = [
    "gross", "pathology", "autopsy", "specimen", "cadaver",
    "dissection", "histolog", "microscop", "h%26e", "h_e_stain",
    "mri", "angiogram", "angiograph", "ultrasound", "x-ray", "xray",
    "diagram", "illustration", "schematic", "drawing",
]
CT_ANAHTARLARI = ["ct", "tomograph", "scan", "ct-scan", "ct_scan"]
def _kelime_eslesmesi(metin, anahtar_kelimeler):
    for k in anahtar_kelimeler:
        if len(k) <= 3:
            if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", metin):
                return True
        elif k in metin:
            return True
    return False
def gorsel_indir(sayfa_url, anahtar_kelimeler, max_indirme=3, baslangic_index=1, global_indirilen=None):
    print(f"\nSayfa taraniyor: {sayfa_url}")
    indirme_sayaci = 0
    anahtar_kelimeler = [k.lower() for k in anahtar_kelimeler]
    if global_indirilen is None:
        global_indirilen = set()
    try:
        with _http_get(sayfa_url) as response:
            html_content = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  Sayfa alinamadi: {e}")
        return 0
    aday_urller = _gorsel_urllerini_topla(html_content, sayfa_url)
    print(f"  Toplam {len(aday_urller)} gorsel adayi bulundu.")
    indirilen_hedefler = set()
    for url in aday_urller:
        if indirme_sayaci >= max_indirme:
            break
        url_kucuk = urllib.parse.unquote(url).lower()
        if not (url_kucuk.endswith(".jpg") or url_kucuk.endswith(".jpeg") or url_kucuk.endswith(".png")):
            continue
        if any(g in url_kucuk for g in UI_GURULTU_DESENLERI):
            continue
        if any(k in url_kucuk for k in KOMPOZIT_DESENLERI):
            continue
        if any(p in url_kucuk for p in PATOLOJI_OTOPSI_DESENLERI):
            continue
        if not _kelime_eslesmesi(url_kucuk, CT_ANAHTARLARI):
            continue
        if not _kelime_eslesmesi(url_kucuk, anahtar_kelimeler):
            continue
        hedef_url = _wikipedia_orijinal(url)
        if hedef_url in indirilen_hedefler or hedef_url in global_indirilen:
            continue
        indirilen_hedefler.add(hedef_url)
        global_indirilen.add(hedef_url)
        uzanti = ".png" if url_kucuk.endswith(".png") else ".jpg"
        dosya_adi = f"sunum_test_{baslangic_index + indirme_sayaci}{uzanti}"
        kayit_yolu = os.path.join(INDIRME_KLASORU, dosya_adi)
        try:
            _dosya_indir(hedef_url, kayit_yolu)
            boyut_kb = os.path.getsize(kayit_yolu) / 1024
            if boyut_kb < 5:
                os.remove(kayit_yolu)
                print(f"  Cok kucuk gorsel atlandi ({boyut_kb:.1f} KB): {hedef_url}")
                continue
            try:
                with Image.open(kayit_yolu) as im:
                    w, h = im.size
            except Exception:
                os.remove(kayit_yolu)
                print(f"  Bozuk gorsel atlandi: {hedef_url}")
                continue
            if w < 200 or h < 200:
                os.remove(kayit_yolu)
                print(f"  Cok dusuk cozunurluk atlandi ({w}x{h}): {hedef_url}")
                continue
            oran = w / h
            if oran < 0.6 or oran > 1.7:
                os.remove(kayit_yolu)
                print(f"  Kompozit/montaj oran atlandi ({w}x{h}): {hedef_url}")
                continue
            print(f"  Indirildi: {dosya_adi} ({boyut_kb:.1f} KB, {w}x{h}) <- {hedef_url}")
            indirme_sayaci += 1
            time.sleep(1.5)
        except Exception as e:
            print(f"  Atlandi ({e}): {hedef_url}")
            if os.path.exists(kayit_yolu):
                try:
                    os.remove(kayit_yolu)
                except OSError:
                    pass
    if indirme_sayaci == 0:
        print("  Bu sayfadan uygun gorsel indirilemedi.")
    return indirme_sayaci
if __name__ == "__main__":
    print("Sunum icin Web Crawling (Gorsel Toplama) basliyor...")
    hedefler = [
        ("https://en.wikipedia.org/wiki/Intracranial_hemorrhage", 1),
        ("https://en.wikipedia.org/wiki/Traumatic_brain_injury", 1),
        ("https://en.wikipedia.org/wiki/Subdural_hematoma", 1),
    ]
    anahtar_kelimeler = [
        "ct", "scan", "hemorrhage", "haemorrhage", "brain", "tbi",
        "hematoma", "haematoma", "stroke", "intracranial", "cranial",
        "computed_tomography", "head",
    ]
    toplam = 0
    global_indirilen = set()
    for url, limit in hedefler:
        toplam += gorsel_indir(url, anahtar_kelimeler, max_indirme=limit, baslangic_index=toplam + 1, global_indirilen=global_indirilen)
        time.sleep(2)
    print(f"\nIslem tamam. Toplam {toplam} gorsel '{INDIRME_KLASORU}' klasorune kaydedildi.")
    if toplam == 0:
        print("Hicbir gorsel indirilemedi. Internet baglantinizi veya hedef sayfa erisilebilirligini kontrol edin.")
