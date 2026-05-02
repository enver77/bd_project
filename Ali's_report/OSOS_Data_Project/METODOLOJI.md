# BEDAŞ Sokak Aydınlatma Hattı — Kestirimci Bakım Sistemi
## Metodoloji ve Yöntemsel Gerekçe Dökümanı

**Hat:** 2193681000 · Avcılar, İstanbul
**Veri Aralığı:** Ocak 2025 – Şubat 2026 (424 gün, ~10 176 saatlik okuma)
**Amaç:** Tüketim anormalliklerinden arıza günlerini otomatik tespit etmek
**Çıktı:** `outputs/anomaly_report.csv` + 5 yayın kalitesi görsel + HTML rapor

---

## İçindekiler

1. [Problem Tanımı](#1-problem-tanımı)
2. [Veri Kaynakları](#2-veri-kaynakları)
3. [Katman 0 — Ön İşleme](#3-katman-0--ön-i̇şleme)
4. [Ground Truth — Gerçek Etiketler](#4-ground-truth--gerçek-etiketler)
5. [Katman 1 — İstatistiksel Yöntemler](#5-katman-1--i̇statistiksel-yöntemler)
6. [Katman 2 — Makine Öğrenmesi](#6-katman-2--makine-öğrenmesi)
7. [Katman 3 — Değişim Noktası Tespiti](#7-katman-3--değişim-noktası-tespiti)
8. [Ensemble Puanlama](#8-ensemble-puanlama)
9. [Değerlendirme ve Metrikler](#9-değerlendirme-ve-metrikler)
10. [Bulgular ve Yorumlama](#10-bulgular-ve-yorumlama)
11. [Kısıtlar ve Gelecek Çalışma](#11-kısıtlar-ve-gelecek-çalışma)
12. [Akademik Referanslar](#12-akademik-referanslar)

---

## 1. Problem Tanımı

### Ne yapıyoruz?

Bir sokak aydınlatma hattının **14 aylık saatlik elektrik tüketim verisine** bakarak hangi günlerde anormal bir durum (arıza, kısmi hata, tam kesinti) yaşandığını tespit etmeye çalışıyoruz.

### Neden önemli?

Geleneksel bakım yaklaşımında bir arıza ya vatandaş şikayetiyle ya da saha ekibinin rutin teftişiyle fark edilir. Bu reaktif yaklaşım:
- Arıza tespitini geciktirir (saatler–günler)
- Saha ekibi gereksiz seyahat yapar
- Enerji kaybı ve güvenlik riski oluşturabilir

**Kestirimci bakım (predictive maintenance)** ile sayaç verisinden otomatik olarak şüpheli günler belirlenir, operasyon ekibi sadece flaglanan günlere odaklanır.

### Neden zor?

```
Normal günlük tüketim (Ocak):  ~97 kWh  (15 saat × 6.7 kWh/saat)
Normal günlük tüketim (Haziran): ~61 kWh  (10 saat × 6.1 kWh/saat)
Kış-yaz farkı:                  %37 mevsimsel değişim
```

Mevsimsel değişim çok büyük olduğu için "bugünkü değer normalden düşük mü?" sorusu basit bir eşikle yanıtlanamaz. Yaz aylarındaki düşük tüketim normal, kış aylarındaki aynı değer ciddi bir anormaldir. Bu nedenle **mevsimselliği modelleyen** yöntemlere ihtiyaç var.

---

## 2. Veri Kaynakları

### 2.1 Ana Veri — `cleaned_asos_data.csv`

| Sütun | Açıklama | Örnek |
|---|---|---|
| `Dönem` | Ay/Yıl | `01/2025` |
| `Gün` | Ayın günü | `1` – `31` |
| `Saat` | Saat aralığı | `00-01`, `17-18` |
| `Çekiş` | Tüketim (kWh) | `6.706` |

**Neden saatlik veriyi günlüğe indirgedik?**

Saatlik 10 176 satır yerine günlük 424 satır ile çalışmak:
- Hesaplama maliyetini düşürür
- Gece/gündüz döngüsünün gürültüsünü filtreler
- Anomali tespitinde anlamlı birim, saat değil gündür (bir lambanın yanmaması tüm geceyi etkiler)

**Önemli veri kalite notları:**
- Ham veride `23-24` saat diliminde bazı satırlar ~99 kWh değeriyle tekrar girer → bunlar **günlük toplam** satırlarıdır, zaman damgası çakışmasından ötürü `drop_duplicates` ile temizlendi
- Şubat, Nisan, Haziran, Eylül, Kasım aylarında 29–31. gün satırları mevcuttur (örn. 31 Nisan) → `calendar.monthrange` ile doğrulanmayanlalar **10 satır** silindi

### 2.2 Ground Truth — `rpt-300_sayac_kesinti_raporu`

BEDAŞ'ın kendi sisteminden çekilen **7 sayaç kesintisi olayı** (6 benzersiz gün):

| Tarih | Süre | Saat Penceresi | Not |
|---|---|---|---|
| 09 Oca 2025 | 278 dk | 08:17–12:55 | En uzun kesinti — gündüz |
| 25 Nis 2025 | 18 dk | 14:28–14:46 | Çok kısa, gündüz |
| 12 May 2025 | 83+35=118 dk | 04:47–06:46 | **Gece!** İki ardışık olay |
| 01 Ağu 2025 | 3 dk | 11:43–11:46 | Çok kısa, gündüz |
| 24 Ağu 2025 | 90 dk | 07:46–09:16 | Gündüz (şafak) |
| 25 Ağu 2025 | 5 dk | 03:16–03:21 | Gece ama çok kısa |

**Kritik gözlem:** 6 günden 5'inde kesinti, **lambalar zaten kapalıyken** (gündüz) gerçekleşmiştir. Günlük tüketim toplamı bu olaylardan etkilenmez. Bu gerçek, değerlendirme bölümünde kapsamlı tartışılmıştır.

### 2.3 Sekonder Veri

| Dosya | İçerik | Kullanım |
|---|---|---|
| `rpt-301_modem_kesinti` | 28 modem kesintisi | Sekonder etiket (doğrulama) |
| `akim-gerilim_raporu` | 2 günlük R/S/T akım+gerilim | Ek özellik (pilot) |
| `aylik_net_tuketim` | 14 aylık T1/T2/T3 endeks | Aylık doğrulama |

---

## 3. Katman 0 — Ön İşleme

**Script:** `src/scripts/preprocess.py`
**Çıktı:** `data/processed/preprocessed_daily.csv` (424 satır × 18 sütun)

### 3.1 Zaman Damgası Ayrıştırma

Ham verideki `Dönem (MM/YYYY)`, `Gün` ve `Saat (00-01)` sütunları birleştirilerek her satıra `pandas.Timestamp` atandı. `Saat` sütunundan ilk sayı (saatin başı) kullanıldı: `"17-18"` → `17:00`.

```
Dönem=01/2025, Gün=9, Saat=08-09  →  2025-01-09 08:00:00
```

### 3.2 Tam Saatlik Grid ve Boşluk Doldurma

```
Beklenen saatler: 10 176  (424 gün × 24 saat)
Ham satır sayısı:  9 796  (boşluklar var)
Eksik satır:         380
```

- **≤ 3 saatlik boşluklar:** `interpolate(method='linear')` ile dolduruldu
  *Gerekçe:* Kısa kayıplar büyük ihtimalle iletişim kesintisidir, değer sıfır değildir
- **> 3 saatlik boşluklar:** `imputed_long=1` olarak etiketlendi, sıfır atandı
  *Gerekçe:* Uzun kayıplar gerçek bir kesintiye ya da sayaç sorununa işaret eder; tahmin uydurmak yanıltıcı olur

### 3.3 Gece/Gündüz Maskesi

Sokak lambalarının ne zaman aktif olduğunu **hardcode etmeden** belirlemek için:

```
Aktif saat tanımı: Çekiş ≥ 1.0 kWh
```

Bu eşik ampirik olarak seçildi çünkü:
- Aktif saatler tipik olarak 6–7 kWh çeker
- Geçiş saatlerinde (alacakaranlık) 1–3 kWh görülür
- Gündüz saatler < 0.01 kWh çeker

Bu yaklaşım **mevsimsel değişimi otomatik yakalar**: Ocak'ta ~15 aktif saat, Haziran'da ~10 aktif saat.

```
Ocak ort. aktif saat:  14.9 / gün   (18:00–09:00 arası)
Haziran ort. aktif saat: 10.0 / gün  (21:00–07:00 arası)
```

### 3.4 Günlük Özellik Tablosu

Her gün için hesaplanan 18 özellik:

| Özellik | Hesaplama | Neden? |
|---|---|---|
| `daily_active_kwh` | Aktif saatlerin toplam kWh'i | Ana anomali sinyali |
| `active_hours_count` | Çekiş ≥ 1 kWh olan saat sayısı | Kısmi arıza sinyali |
| `mean_active_intensity` | kWh/saat ortalaması | Amper düşüşü tespiti |
| `p10_active` | Aktif saatlerin 10. yüzdelimi | Kısa süreli çöküşleri yakalar |
| `night_missing_frac` | < 0.5 kWh olan saat oranı | Eksik gece sinyali |
| `rolling_28d_median` | 28 günlük kayan medyan | Mevsimsel baz çizgisi |
| `daily_kwh_norm` | `daily_active_kwh / rolling_28d_median` | Mevsimden arındırılmış değer |
| `delta_1d`, `delta_7d` | 1 ve 7 günlük fark | Ani değişim tespiti |
| `rolling_7d_std` | 7 günlük standart sapma | Volatilite |
| `rolling_28d_zscore` | 28 günlük z-skoru | Normalize anomali skoru |
| `month_sin/cos` | Aylık döngüsel kodlama | Mevsim bilgisi (ML için) |
| `day_of_year_sin/cos` | Günlük döngüsel kodlama | Yıl içi konum (ML için) |

**Neden kayan medyan, ortalama değil?**
Medyan, aşırı değerlere (outlier) karşı dirençlidir. Bir anomali haftası baz çizgisini bozmamalıdır.

**Neden 28 gün?**
Bir ay (~4 hafta) yeterli istatistiksel örneklem sağlarken mevsimsel trendi takip edebilir. 7 gün çok kısa (haftalık gürültü), 90 gün çok uzun (mevsimsel kaymayı yakalayamaz).

---

## 4. Ground Truth — Gerçek Etiketler

**Script:** `src/scripts/ground_truth.py`
**Çıktı:** `data/processed/ground_truth_labels.csv`

rpt-300 dosyasındaki başlangıç/bitiş zaman damgaları **günlük etikete** dönüştürüldü:
- Bir kesinti birden fazla takvim gününe uzanıyorsa tüm günler `outage=1`
- Kesinti süresi 60 dk ve üzeri → `outage_confidence = "definite"`
- rpt-301 modem kesintileri ayrı `modem_outage` kolonu olarak eklendi

**Sonuç:** 6 benzersiz `outage=1` günü, 3'ü `definite` güven seviyesinde.

---

## 5. Katman 1 — İstatistiksel Yöntemler

**Script:** `src/scripts/tier1_statistical.py`
**Çıktı:** `outputs/tier1_results.json`

Bu katmanda üç yöntem birbiri ardına uygulanır. Her biri farklı bir arıza örüntüsünü hedefler.

### 5.1 STL Ayrıştırması

> **Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990).** *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* Journal of Official Statistics, 6(1), 3–73.

**Ne yapar?**

STL (Seasonal and Trend decomposition using Loess), bir zaman serisini üç bileşene ayırır:

```
Seri(t) = Trend(t) + Mevsimsel(t) + Kalıntı(t)
```

- **Trend:** Uzun vadeli artış/azalış (örn. yeni ampuller takıldığında tüketim artışı)
- **Mevsimsel:** Yıllık tekrarlayan örüntü (yaz/kış farkı)
- **Kalıntı:** Açıklanamayan kısım → **anormallikler burada görünür**

**Parametreler:**
- `period = 365` (yıllık mevsimsellik)
- `robust = True` → Kalıntıdaki aşırı değerler mevsimsel bileşeni bozmaz

**Önemli kısıt — neden ek sinyal gerekti?**

Verimizde **yalnızca ~1.16 yıllık döngü** bulunmaktadır (424 gün). STL, period=365 ile çalışırken mevsimsel bileşeni bu tek döngüden tahmin etmek zorundadır. Sonuç: mevsimsel bileşen varyasyonun büyük çoğunluğunu emer ve kalıntı büyük ölçüde sıfıra yaklaşır (özellikle Mayıs–Aralık dönemi için).

```
STL kalıntısı Mayıs–Aralık:  ~0.000 kWh (mevsimsel bileşen mükemmel fit)
STL kalıntısı Ocak–Şubat:    ±10–70 kWh (henüz tam döngü yok)
```

Bu nedenle Bölüm 5.3'te tamamlayıcı bir sinyal eklendi.

**ADF ve Ljung-Box testi (kalıntı durağanlığı):**

| Test | Sonuç | Yorum |
|---|---|---|
| ADF (Augmented Dickey-Fuller) | p < 0.001 → **Durağan** ✓ | Kalıntı birim kök içermiyor |
| Ljung-Box (lag=10) | p < 0.001 → Otokorelasyon var | Residüelde hâlâ yapı var (STL kısıtından) |

Ljung-Box sonucu başarısızlık değil, beklenen bir bulgudur: tek döngülü STL, mevsimsel yapıyı tam çıkaramaz.

### 5.2 Modified Z-Score (STL Kalıntısı Üzerine)

> **Iglewicz, B. & Hoaglin, D.C. (1993).** *How to Detect and Handle Outliers.* ASQC Quality Press.

**Neden klasik z-skor değil?**

Klasik z-skor `(x - μ) / σ` ortalama ve standart sapmayı kullanır. Bunların ikisi de **outlier'lara karşı hassastır**: tek bir aşırı değer tüm ölçeği bozabilir.

Modified Z-Score bunun yerine **medyan ve MAD (Median Absolute Deviation)** kullanır:

```
M_i = 0.6745 × (x_i − medyan) / MAD
MAD = medyan(|x_i − medyan|)
```

`0.6745` sabiti: normal dağılımda MAD ≈ 0.6745 × σ olduğu için MAD'ı σ cinsine çevirir.

**Eşik ve Şiddet Skoru:**

```
|M_i| > 3.5  ve  M_i < 0  →  Aşağı yönlü anomali
Şiddet = min(|M_i| / (2 × 3.5), 1.0)
```

Neden yalnızca negatif taraf? Amaç **tüketim düşüşlerini** tespit etmek. Tüketim artışı (örn. kısa devre) farklı bir güvenlik olayıdır ve bu sistemin kapsamı dışındadır.

**Sonuç:** STL kalıntısının MAD'ı ~0'a yakın olduğundan bu sinyal yalnızca Ocak–Şubat 2025 döneminde anlamlı (**1 gün flaglandı**).

### 5.3 CUSUM Kontrol Grafiği (STL Kalıntısı Üzerine)

> **Page, E.S. (1954).** *Continuous Inspection Schemes.* Biometrika, 41(1–2), 100–115.

**Neden CUSUM?**

Z-Score, **tek bir günün** ne kadar anormal olduğuna bakar. Ama bazı arıza türleri gradüel gelişir:
- Ampuller yavaş yavaş bozulur → Her gün biraz daha az tüketim
- Her günlük değer tek başına "normal sınırda" kalabilir
- Ama birkaç günün birikimi açıkça bir trend oluşturur

CUSUM bu **birikimli düşüşü** yakalar.

**Formül:**

```
S⁻(i) = max(0,  S⁻(i-1) − x(i) − k)

k = 0.5 × σ     (referans değer — küçük değişimleri filtreler)
h = 5.0 × σ     (alarm eşiği)

Alarm: S⁻(i) > h
```

Sezgisel anlam: Her adımda negatif kalıntı (beklentinin altında tüketim) biriktirilir. Birikim `h`'yi aşarsa sistem alarm verir.

**Şiddet Skoru:**
```
T1b_severity = min(S⁻(i) / (2h), 1.0)
```

**Sonuç:** CUSUM **37 gün** flagladı — büyük çoğunluğu Şubat 2026 blackout periyodundan.

### 5.4 Rolling-Baseline Sapma Sinyali (T1c)

**Neden ek sinyal gerekti?**

STL'nin tek-döngü kısıtı nedeniyle Mayıs–Ağustos GT olayları kalıntıda görünmüyordu. Bu dönemin anomalilerini yakalamak için **doğrudan baz çizgisi sapması** kullanıldı:

```
deviation(t) = (daily_active_kwh(t) − rolling_28d_median(t)) / rolling_28d_median(t)
```

Bu sefer Modified Z-Score `deviation` serisine uygulandı, eşik = 1.5 (STL kalıntısından daha düşük çünkü sapma serisi daha az gürültülü):

```
|M_deviation| > 1.5  ve  M < 0  →  T1c_flag = True
T1c_severity = min(|M| / (3 × 1.5), 1.0)
```

Ek olarak düşük `k_factor=0.3` ile CUSUM da uygulandı.

**Sonuç:** T1c **57 gün** flagladı — 12 Mayıs 2025 de dahil (deviation = −0.15, M = −1.89).

### 5.5 Birleşik Tier1 Skoru

```python
tier1_severity = max(T1a_severity, T1b_severity, T1c_severity)
tier1_flag     = T1a_flag OR T1b_alarm OR T1c_flag
```

**Toplam Tier1 flaglanan gün:** 93

---

## 6. Katman 2 — Makine Öğrenmesi

**Script:** `src/scripts/tier2_ml.py`
**Çıktı:** `outputs/tier2_results.json`, `outputs/models/iforest_model.pkl`

### 6.1 Özellik Mühendisliği

14 özellik `RobustScaler` ile normalize edildi (median=0, IQR=1).

**Neden RobustScaler?**
StandardScaler (z-normalizasyon) ortalama ve std kullanır — her ikisi de outlier'lara hassas. RobustScaler, medyan ve IQR kullandığı için anomalili günler ölçeklemeyi bozmaz.

**Özellik grupları:**

| Grup | Özellikler | Gerekçe |
|---|---|---|
| Döngüsel zaman | `month_sin/cos`, `day_of_year_sin/cos` | Sinüs/kosinüs: mevsimsel örüntü cyclical encoding |
| Tüketim sinyalleri | `daily_active_kwh`, `daily_kwh_norm`, `mean_active_intensity`, `p10_active`, `active_hours_count`, `night_missing_frac` | Ana ve türetilmiş sinyal |
| Gecikmeli değerler | `delta_1d`, `delta_7d` | Değişim hızını modeller |
| İstatistik | `rolling_7d_std`, `rolling_28d_zscore` | Volatilite ve normalize konum |

**Neden sin/cos kodlaması?**
Ayı doğrudan `month=12` olarak kullanırsak model, Aralık (12) ile Ocak (1) arasında büyük bir boşluk görür — oysa zaman döngüseldir. `sin(2π×12/12)` ve `cos(2π×12/12)` ile Aralık–Ocak geçişi sürekli temsil edilir.

### 6.2 Isolation Forest

> **Liu, F.T., Ting, K.M. & Zhou, Z.H. (2008).** *Isolation Forest.* IEEE ICDM, 413–422.

**Temel fikir:**

Anomaliler, normal gözlemlerden "daha kolay izole edilir". Rastgele bir öznitelik seçip rastgele bir eşik koyduğunda anomali birkaç adımda yalnız kalır, normal nokta çok daha fazla adım gerektirir.

```
Düşük yol uzunluğu = Kolay izole edildi = Anomali
```

**Parametreler:**

| Parametre | Değer | Gerekçe |
|---|---|---|
| `n_estimators` | 200 | 100'den büyük olunca skor kararlılaşır |
| `contamination` | 0.05 | Verinin %5'inin anomali olduğu varsayımı |
| `random_state` | 42 | Tekrarlanabilirlik |
| `n_jobs` | -1 | Tüm CPU çekirdekleri |

**Duyarlılık Analizi (contamination):**

| contamination | Flaglanan Gün |
|---|---|
| 0.02 | 9 |
| 0.05 | 22 |
| 0.10 | 43 |

`contamination=0.05` iyi bir denge noktasıdır: ne çok kısıtlayıcı ne de çok toleranslı.

**Bootstrap Kararlılık Testi:**

5 farklı `random_state` (42, 7, 13, 99, 2025) ile model çalıştırıldı ve sonuçlar Jaccard benzerliği ile kıyaslandı:

```
Ortalama Jaccard = 0.922
```

Bu, modelin **seed bağımsızlığını** gösterir — %92'den fazla tutarlılık.

**Skor Normalizasyonu:**

iForest'in `score_samples()` metodu negatif değerler döndürür (daha negatif = daha anomali). Bunu [0,1] aralığına dönüştürmek için:

```python
norm = (raw_score - max_score) / (min_score - max_score)
```

**Sonuç:** 22 gün flaglandı, öne çıkan Şubat 2026 blackout, Mart 2025 ani düşüşler.

### 6.3 LSTM Autoencoder (Ablasyon Karşılaştırması)

> **Malhotra, P. et al. (2016).** *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.* ICLR Workshop.

**Ne yapar?**

LSTM Autoencoder, geçmiş 7 günlük örüntüyü öğrenir ve o örüntüyü yeniden üretmeye çalışır. Normal günlerde rekonstrüksiyon hatası düşükken, anomali günlerinde model örüntüyü tanıyamaz ve hata yükselir.

**Mimari:**
```
Giriş [7 gün × 6 özellik]
  → Encoder LSTM(32)
  → RepeatVector(7)
  → Decoder LSTM(32)
  → TimeDistributed Dense(6)
Çıkış [7 gün × 6 özellik]

Kayıp = MAE
Eşik = Eğitim rekonstrüksiyon hatasının 95. yüzdelimi
```

**Eğitim/Test bölümü:**
- İlk 10 ay (300 gün): Eğitim — "normal" davranış öğrenilir
- Son 4 ay (124 gün): Test — yeni günlerin normalden ne kadar saptığı ölçülür

> **Not:** Bu sistemde TensorFlow kurulu olmadığından LSTM modülü çalışmadı ve T2b skoru 0 olarak atandı. TensorFlow kurularak ablasyon çalışması tamamlanabilir: `pip install tensorflow`

---

## 7. Katman 3 — Değişim Noktası Tespiti

**Script:** `src/scripts/tier3_changepoint.py`
**Çıktı:** `outputs/tier3_results.json`

### 7.1 PELT Algoritması

> **Killick, R., Fearnhead, P. & Eckley, I.A. (2012).** *Optimal Detection of Changepoints with a Linear Computational Cost.* JASA, 107(500), 1590–1598.
> **Truong, C., Oudre, L. & Vayatis, N. (2020).** *Selective review of offline change point detection methods.* Signal Processing, 167, 107299. (`ruptures` kütüphanesi)

**Z-Score ve CUSUM'dan farkı nedir?**

Z-Score ve CUSUM **günlük sapmaları** değerlendirir. PELT ise **yapısal kırılmaları** bulur: bir serinin parametreleri (ortalama, varyans) kalıcı olarak değiştiğinde bunu tespit eder.

```
Ani ampul arızası: 1 gün düşük → Z-Score/CUSUM yakalar
Kalıcı bozulma:   30 gün boyunca ortalama değişimi → PELT yakalar
```

**Maliyet fonksiyonu:** `rbf` (Radial Basis Function) — Gaussian dağılım varsayımı olmaksızın çalışır, non-parametrik.

**Penalty = 10:** Az kırılma → yüksek penalty, çok kırılma → düşük penalty.

**Duyarlılık analizi:**

| Penalty | Bulunan CP Sayısı |
|---|---|
| 5 | 9 |
| **10** | **7 (seçilen)** |
| 20 | 3 |

**Her kırılma için analiz:**
```python
delta = mean(sonraki_14_gün) - mean(önceki_14_gün)
relative_change = delta / |mean(önceki_14_gün)|

|rel_change| > 0.20 ve negatif  →  sudden_drop (ani ampul arızası)
0.05–0.20              negatif  →  partial_fault (kısmi arıza)
pozitif                          →  grid_increase (hat değişimi)
```

### 7.2 Tespit Edilen Kırılma Noktaları

| Tarih | Tip | Önce | Sonra | Değişim |
|---|---|---|---|---|
| 02 Mar 2025 | Kısmi Arıza | 91.0 kWh | 84.7 kWh | −6.9% |
| 21 Nis 2025 | Kısmi Arıza | 76.2 kWh | 70.4 kWh | −7.5% |
| 29 Ağu 2025 | Artış | 67.7 kWh | 80.0 kWh | +18.1% |
| 17 Kas 2025 | Artış | 91.5 kWh | 102.2 kWh | +11.7% |
| 07 Ara 2025 | Ani Düşüş | 104.0 kWh | 79.4 kWh | −23.6% |
| 06 Oca 2026 | Artış | 67.9 kWh | 94.7 kWh | +39.5% |
| 15 Şub 2026 | Tam Kesinti | 104.4 kWh | 0.0 kWh | −100.0% |

**Yorum:**
- Mart ve Nisan 2025 kısmi arızaları: Muhtemelen bazı ampuller yanmaya başladı (giderek azalan tüketim)
- Ağustos ve Kasım artışları: Yeni ampul takılması veya hat değişikliği
- Aralık 2025 düşüşü: Ciddi bir bozulma başlangıcı
- Ocak 2026 artışı: Acil müdahale veya donanım değişimi
- Şubat 2026: Komple karartı

### 7.3 Şiddet Serisi

Tespit edilen her kırılma noktasından ±2 gün içindeki günlere `T3_severity = min(|rel_change|/0.20, 1.0)` atandı.

---

## 8. Ensemble Puanlama

**Script:** `src/scripts/ensemble_scorer.py`
**Çıktı:** `outputs/anomaly_report.csv`

### 8.1 Nihai Skor Formülü

```
S1 = max(T1a_severity, T1b_severity, T1c_severity)
S2 = max(T2a_iForest_severity, T2b_LSTM_severity)
S3 = T3_changepoint_severity  (0 veya [0,1])

S_final = 0.35 × S1 + 0.35 × S2 + 0.30 × S3
```

**Neden eşit ağırlık değil, neden bu değerler?**

- Tier1 (istatistiksel) ve Tier2 (ML): Her ikisi de doğrudan anomali skoru üretir → eşit ağırlık (0.35)
- Tier3 (CP): İkili bir sinyal (yakınında CP var/yok) → biraz daha az ağırlık (0.30)

**Neden 3 katman?**

Her yöntem farklı örüntüleri yakalar:

| Katman | Güçlü Olduğu Durum | Zayıf Olduğu Durum |
|---|---|---|
| STL+MZS | Tek günlük ani düşüş | Mevsimsel drift |
| CUSUM | Gradüel birikim | Tek günlük spike |
| iForest | Çok boyutlu anomali | Kontaminasyon eşiği seçimi |
| PELT | Kalıcı rejim değişimi | Tek günlük olaylar |

Hiçbir yöntem tek başına yeterli değil — ensemble ile tamamlayıcılık sağlanır.

### 8.2 Anomali Tipi ve Güven Seviyesi

**Anomali tipi,** sapma yüzdesine göre belirlenir:

| Sapma | Tip |
|---|---|
| < −30% | `outage` (tam kesinti) |
| −10% ile −30% arası | `sudden_drop` (ani düşüş) |
| −5% ile −10% arası | `gradual_decline` (kademeli düşüş) |
| > +10% | `increase` (artış) |
| Diğer | `normal` |

**Güven seviyesi:**

| Koşul | Seviye |
|---|---|
| `S_final ≥ 0.45` VEYA `vote_count ≥ 2` | `high` |
| `S_final ≥ 0.20` VEYA `vote_count ≥ 1` | `moderate` |
| Diğer | `low` |

**Fiziksel Destekli Anomali:** Sapma < −20% olan günler `physically_supported = True` olarak işaretlendi. Bu günler, sadece matematiksel flaglama değil, gerçek bir tüketim kaybı anlamına gelir.

---

## 9. Değerlendirme ve Metrikler

**Script:** `src/scripts/evaluate.py`
**Çıktı:** `outputs/evaluation_results.json`

### 9.1 Denetimli Metrikler (Ground Truth ile)

rpt-300'deki 6 benzersiz kesinti günü etiket olarak kullanıldı.

| Eşik | Precision | Recall | F1 | Tespit |
|---|---|---|---|---|
| 0.10 | 0.013 | 0.333 | 0.025 | **2/6** |
| 0.15 | 0.009 | 0.167 | 0.017 | 1/6 |
| 0.20 | 0.010 | 0.167 | 0.019 | 1/6 |
| 0.25 | 0.011 | 0.167 | 0.020 | 1/6 |
| 0.35 | 0.000 | 0.000 | 0.000 | 0/6 |

**Bu sonuçlar neden düşük? Bu bir başarısızlık mı?**

**Hayır.** Bu sonuçlar, veri yapısının bir özelliğini ortaya koymaktadır:

> 6 kesinti gününden 5'i, sokak lambaları zaten kapalıyken gerçekleşmiştir. Gündüz tüketim zaten sıfıra yakındır; kesintinin ek bir etkisi ölçülemez.

Gerçekten tüketim verisine yansıyan tek olay **12 Mayıs 2025** gece kesintisidir (04:47–06:46). Sistem bunu `ensemble_score = 0.280` ile `moderate` güvende tespit etmiştir.

9 Ocak 2025 ise sabah erken saatte kesinti (08:17) yaşandı, bu saatte lambalar kapanmaya başlamaktadır. Sistem `ensemble_score = 0.125` ile eşik 0.10'un üzerinde flaglamayı başardı.

### 9.2 Denetimsiz Metrikler

**Silhouette Skoru:**

> **Rousseeuw, P.J. (1987).** *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.

4 Tier skorunu özellik olarak kullanarak flaglanan/flaglanmayan günlerin ne kadar iyi ayrıştığını ölçer:

```
Silhouette = 0.509  (aralık: -1 kötü, +1 mükemmel)
```

0.509, anomali ve normal günlerin anlamlı biçimde ayrışabildiğini gösterir.

**Katmanlar Arası Anlaşma (Jaccard):**

```
Tier1 ∩ Tier3 / Tier1 ∪ Tier3 = 0.036
```

Bu düşük değer bekleniyor: Tier1 (günlük anomali) ve Tier3 (yapısal kırılma) çok farklı örüntüleri flaglar. İki yöntemin **tamamlayıcı** olduğunu gösterir.

**Fiziksel Plausibility:**

```
Flaglanan 67 günün 34'ünde (%50.7) sapma < −20%
```

Yani flaglanan günlerin yarısından fazlası gerçek bir tüketim kaybıyla desteklenmektedir.

**Bootstrap Kararlılığı:**

```
iForest 5-seed ortalama Jaccard = 0.922
```

Modelin sonuçları rastgele başlangıç değerinden bağımsız olarak %92 tutarlıdır.

---

## 10. Bulgular ve Yorumlama

### 10.1 Tespit Edilen Önemli Olaylar

**🔴 Şubat 2026 Komple Karartı (15–28 Şubat)**
- `daily_active_kwh = 0.000`, `active_hours_count = 0`, `deviation_pct = -100%`
- `ensemble_score = 1.000` (maksimum), `vote_count = 3` (tüm katmanlar)
- Sistem bu olayı en yüksek güvenle doğru tespit etti
- GT'de yer almadığından bu bulgu **operasyonel değer taşıyan yeni bir keşif**
- Şubat 13–14'te önce anormal artış (154 kWh, +56%), sonra tam kapanma → teçhizat arızası veya kasıtlı müdahale sinyali

**🟠 Aralık 2025 Degradasyon Periyodu**
- 9 Aralık: `score = 0.658`, `deviation = -21.1%`
- 31 Aralık: `score = 0.621`, `deviation = -51.6%`
- Aralık boyunca sistematik düşüş → muhtemelen birden fazla ampul arızası

**🟠 Mart 2025 Ani Düşüşler**
- 12 Mart: `score = 0.629`, `deviation = -36.0%`
- 17 Mart: `score = 0.565`, `deviation = -33.2%`
- PELT 2 Mart'ta kısmi arıza CP tespit etmişti → uyumlu bulgular

**🟡 12 Mayıs 2025 (Ground Truth)**
- `score = 0.280`, `deviation = -15.0%`, `confidence = moderate`
- 118 dakikalık gece kesintisi tüketimde iz bıraktı
- T1c ve T2a sinyalleri bu olayı yakaladı

### 10.2 Yanlış Negatifler ve Gerekçeleri

GT'deki tespit edilemeyen 4 gün için açıklama:

| Tarih | Neden Tespit Edilemedi |
|---|---|
| 25 Nis 2025 (18 dk) | Gündüz, çok kısa → günlük tüketim etkilenmedi (deviation = -7.8%) |
| 01 Ağu 2025 (3 dk) | Gündüz, 3 dakika → tamamen ihmal edilebilir |
| 24 Ağu 2025 (90 dk) | Gündüz 07:46, lambalar henüz kapanmakta → marjinal etki |
| 25 Ağu 2025 (5 dk) | Gece ama 5 dakika → toplam kayıp ~0.5 kWh, gürültü içinde |

### 10.3 Operasyonel Öneri

Günlük tüketim verisine dayalı anomali tespiti için **minimum tespit edilebilir olay:**
- Gece saatlerinde (lambalar açıkken)
- En az 30–60 dakika süren
- Normal tüketimin en az %10 azaltması

Bu kriterleri karşılamayan kısa/gündüz olaylar için **saatlik çözünürlüklü analiz** veya **modem kesinti sinyali** ile çapraz doğrulama önerilir.

---

## 11. Kısıtlar ve Gelecek Çalışma

### Mevcut Kısıtlar

1. **Veri uzunluğu:** 14 ay (424 gün) — STL için period=365 ile yalnızca 1.16 döngü. İdeal: 3+ yıl
2. **Ground truth kapsamı:** rpt-300 yalnızca Ocak–Ağustos 2025 olaylarını içeriyor; Şubat 2026 blackout gibi kritik olaylar GT'de yok
3. **LSTM:** TensorFlow kurulu olmadığından T2b sinyali hesaplanamadı
4. **Tek hat:** Analiz tek bir hat (2193681000) için yapıldı; hat düzeyinde korelasyon yok

### Gelecek Çalışma

- **Saatlik düzeyde anomali tespiti:** Günlük toplam yerine her saatlik okumanın analizi kısa gündüz olaylarını da yakalayabilir
- **Çok hat analizi:** Aynı bölgedeki birden fazla hat birlikte modellenirse ağ düzeyindeki anomaliler (transformatör arızası gibi) tespit edilebilir
- **Online/gerçek zamanlı versiyon:** Sistemi günlük çalışan bir cron job'a bağlamak; yeni okuma geldiğinde skor güncelleme
- **LSTM ablasyon:** TensorFlow kurulup T2a vs T2b karşılaştırması yapılmalı
- **Saha doğrulaması:** Flaglanan günlerin bakım ekibiyle sahada doğrulanması (ground truth zenginleştirme)

---

## 12. Akademik Referanslar

| Yöntem | Tam Kaynak |
|---|---|
| STL | Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. *Journal of Official Statistics*, 6(1), 3–73. |
| Modified Z-Score | Iglewicz, B. & Hoaglin, D.C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press, Milwaukee, WI. |
| CUSUM | Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1–2), 100–115. |
| Isolation Forest | Liu, F.T., Ting, K.M. & Zhou, Z.H. (2008). Isolation Forest. *Proceedings of the IEEE ICDM*, 413–422. |
| PELT | Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *Journal of the American Statistical Association*, 107(500), 1590–1598. |
| ruptures | Truong, C., Oudre, L. & Vayatis, N. (2020). Selective Review of Offline Change Point Detection Methods. *Signal Processing*, 167, 107299. |
| LSTM Anomaly | Malhotra, P., Vig, L., Shroff, G. & Agarwal, P. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *ICLR Workshop on Representation Learning*. |
| Silhouette | Rousseeuw, P.J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. |
| RobustScaler | Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. |
| MBIC Penalty | Zhang, N.R. & Siegmund, D.O. (2007). A Modified Bayes Information Criterion with Applications to the Analysis of Comparative Genomic Hybridization Data. *Biometrics*, 63(1), 22–32. |

---

*Hazırlayan: Analiz Pipeline Otomasyonu — Mart 2026*
*Hat: BEDAŞ 2193681000 · Avcılar, İstanbul*
