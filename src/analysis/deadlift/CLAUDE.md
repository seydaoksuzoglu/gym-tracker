# Deadlift Form Analysis Module — Technical Plan

> Bu dosya, deadlift form analizi modülünün mimari ve uygulama planıdır.
> Claude Code bu dosyayı her oturumda okur ve kararlarını buna göre verir.
> Plan değişikliklerini buradan yap sohbette kısaca anlat.

---

## 1. Project Context

Pose estimation tabanlı bir form analizi uygulamasının **deadlift modülü** geliştiriliyor. Squat ve biceps curl modülleri ayrı olarak gelecek; bu plan **sadece deadlift** içindir.

### Önceki sistemden öğrenilen dersler

Squat modülünde başarısızlığa yol açan hatalar — bu modülde **tekrarlanmayacak**:

- Tek bir açı için sabit eşik (örn. `knee_angle < 120`) kullanmak
- Frame bazlı anlık karar (tek frame anomalisi → hata raporu)
- Ham landmark koordinatlarını filtrelemeden kullanmak
- Kullanıcı boy/oran/kamera mesafesi farklılıklarını normalize etmemek
- Binary (hata var / yok) çıktı

### Bu modülde benimsenen temel ilkeler

1. **Phase-based değerlendirme** — frame-based değil
2. **Açılar arası ilişki (T ↔ K)** — tek açıya bakmak yerine
3. **Z-skor normalizasyonu** — sabit eşik yerine eğitmen dağılımına göre sapma
4. **T-açısı checkpoint örnekleme** — her frame'i değil, gövdenin belirli açılarda olduğu anları değerlendirme
5. **Vücut oranı tabanlı normalizasyon** — kullanıcı boyu ve kamera mesafesine bağımsız
6. **Filtrelenmiş sinyal + multi-frame onay** — titreme dayanıklılığı
7. **Per-error confidence skoru** — binary değil, 0.0–1.0 güven değeri
8. **5 alanlı ağırlıklı puanlama** — Area 1 (mükemmel) → Area 5 (ciddi hata)

---

## 2. Scope (V0 — MVP)

### Kapsam içi
- **Tek bakış açısı:** Side view (yan kamera). Front view bu sürümde yok.
- **Tek hareket:** Standart konvansiyonel deadlift. Sumo, Romanian, stiff-leg yok.
- **Iki giriş kaynağı:** Canlı kamera akışı + video dosyası yükleme.
- **Iki hata kategorisi (V0):** `incomplete_lockout`, `uncontrolled_descent`.
- **V0 referans dağılımı:** Literatür tabanlı tahmini değerler (henüz antrenör verisi toplanmadı).
- **Rep counting:** Squat için yazıldı ve çalışıyor; deadlift için **Sprint 2'de** yeni bir `rep_counter.py` yazılacak. Squat counter şablon olarak kullanılır, mantık deadlift fazlarına adapte edilir (lockout tamamlanması bir rep'i kapatır).

### Kapsam dışı (sonraki sürümler)
- Front view analizi
- LLM doğal dil geri bildirim
- Chaffin L5/S1 kuvvet/sakatlık risk modülü
- `rounded_back`, `hips_shoot_up` hata tespiti (V1'e bırakıldı)
- Sumo / RDL varyasyonları

---

## 3. Pipeline (Frame-by-Frame İşleme)

Her frame için **sırayla** şu katmanlar çalışır:

```
[Frame] → 1.Capture → 1.5.PreChecks → 2.Filter → 3.Normalize → 4.Phase
       → 5.Checkpoint → 6.Z-Score → 7.Confidence → 8.Score
```

### Katman 1: Capture
- MediaPipe Pose ile 33 landmark çıkar.
- Her landmark için `visibility` skorunu sakla.
- `visibility < 0.5` olan landmark `None` olarak işaretlenir.
- Tamamen kaybolan landmark'lar için `[-1, -1]` placeholder kullan (occlusion dayanıklılığı için).

- Tamamen kaybolan landmark'lar için `[-1, -1]` placeholder kullan (occlusion dayanıklılığı için).

### Katman 1.5: Pre-Checks (View Sanity Check)
- Pipeline başında, filtrelemeden önce çalışır.
- **Side view kontrolü:** Sol omuz ve sağ omuz arasındaki x-mesafesi, kalibrasyondan gelen `torso_length_baseline`'ın %20'sinden küçükse → side view kabul edilir, frame geçer.
- Mesafe daha büyükse → `RejectedFrame(reason='not_side_view')` döndürülür, frame pipeline'da ilerlemez.
- UI katmanına uyarı sinyali: "Lütfen kamerayı yandan konumlandırın".
- **Anti-pattern uyarısı:** view tespiti `deadlift_features.py`'ye sızdırılmayacak — bu kontrol burada kalır, başka modüller bilmez.
- **Sprint 1.1 sapması (2026-05):** Plan başlangıçta `shoulder_width × 0.30` öngörüyordu. Side-view'da iki omuz birbirinin arkasında kaldığı için 2D `shoulder_width` projeksiyonu ~0.01-0.02 mertebesine düşüyor; bu da eşiği tersine çeviriyor ve normal side-view frame'lerini reddediyordu. `torso_length` görünüme bağımsız (omuz-mid → kalça-mid dikey mesafe), kararlı bir referans olduğu için onunla değiştirildi. Eşik oranı `0.30` → `0.20` olarak düşürüldü.

### Katman 2: Landmark Filtreleme
- **Asla ham koordinat kullanma.**
- Exponential Moving Average (EMA), önerilen `α = 0.3`.
- Alternatif: 5-frame median filter.
- Filtre her landmark için bağımsız uygulanır.
- `None` işaretli landmark'lar filtreden geçirilmez; ilgili metrikler o frame için hesaplanmaz.

### Katman 3: Vücut Oranı Normalizasyonu (Calibration)
- İlk **30 frame** (kullanıcı bar'ın önünde sabit) `STANDING_BASELINE` olarak topla.
- Hesapla:
  - `torso_length = mean(distance(shoulder_mid, hip_mid))`
  - `shoulder_width = mean(distance(left_shoulder, right_shoulder))`
  - `femur_length = mean(distance(hip, knee))`
- Tüm mesafe metrikleri bu değerlere bölünerek normalize edilir.
- Açılar zaten boy-bağımsız, ek normalizasyon gerekmez.
- Kalibrasyon başarısız olursa (kullanıcı hareketli, landmark eksik) kullanıcıya "Bar'ın önünde 1 saniye sabit dur" uyarısı.

### Katman 4: Faz Tespiti (State Machine)

| Faz | Tetikleyici Sinyal |
|---|---|
| `SETUP` | `hip.y` düşük, `T > 30°`, hareket yok (velocity ≈ 0) |
| `PULL` | `hip.y` yükseliyor, `T` azalıyor |
| `LOCKOUT` | `hip.y` ≈ standing baseline, `T < 5°`, **300ms** boyunca sabit (timestamp delta ile, frame sayısı değil) |
| `DESCENT` | `hip.y` düşüyor, `T` artıyor |

- Geçişleri `hip.y` türevi + `T` açısı eşiği ile yakala. **Türev = Δhip.y / Δtimestamp_ms** (frame index farkı değil).
- Faz geçmişi rep boyunca tutulur (rep tamamlandığında: SETUP → PULL → LOCKOUT → DESCENT → SETUP).
- Faz süreleri `phase_durations_ms` olarak `timestamp_ms` cinsinden kaydedilir; FPS değişkenliğinden bağımsızdır.

### Katman 5: T-Açısı Checkpoint Örnekleme

**Bu modülün squat sisteminden temel farkı.**

- Her frame'i değerlendirme.
- Sadece gövdenin belirli açılarda olduğu anlardaki frame'i yakala:
  - **PULL fazı checkpoint'leri:** `T = {35°, 30°, 25°, 20°, 15°, 10°, 5°}`
  - **DESCENT fazı checkpoint'leri:** Aynı açılar tersten
- Toleranslı yakalama: `|T_current - T_checkpoint| < 1.5°` ise o checkpoint düştü kabul et.
- Her checkpoint'te ölç:
  - `K` (diz açısı): hip → knee → ankle vektörlerinin açısı
  - `back_angle` (sırt açısı): omuz → kalça hattının yatayla açısı
  - `hip_shoulder_velocity_ratio`: kalça-omuz dikey hız oranı

**Edge case kuralları:**
- **Hızlı geçiş (frame atlama):** İki ardışık frame arası `T` değişimi > 3° ise, kaçırılan checkpoint'leri **lineer interpolasyon** ile sentezle. (Örn: frame N'de T=24°, frame N+1'de T=19° ise; T=20° checkpoint'i iki frame'in lineer ara değeriyle oluşturulur.)
- **Çoklu tetikleme:** Aynı checkpoint birden fazla frame'de `|T_current - T_checkpoint| < 1.5°` toleransına girerse, **T'nin checkpoint değerine en yakın olduğu frame** seçilir; diğerleri yok sayılır.

### Katman 6: Z-Skor Değerlendirmesi

- Her checkpoint için referans dağılım: `(T_checkpoint, K_mean, K_std, back_mean, back_std, ...)`
- Referans veriler **`reference_distributions.json`** dosyasında tutulur — kod değişmeden V0 → V1 geçişi yapılabilmeli.
- Hesaplama: `z = (user_value - ref_mean) / ref_std`
- V0 referans değerleri **literatür tabanlı tahmini** (antrenör verisi V1'de toplanacak).

### Katman 7: Hata Bazlı Güven Skoru

- Her hata kategorisi için **bağımsız** confidence skoru (0.0–1.0).
- **V0'da aktif hatalar:**

| Hata | Sinyal | Faz |
|---|---|---|
| `incomplete_lockout` | LOCKOUT'ta `T` min değeri > 5° | LOCKOUT |
| `uncontrolled_descent` | DESCENT süresi (ms) PULL süresinin (ms) <%50'si VEYA hip dikey ivme (Δhip.y / Δts_ms²) > eşik | DESCENT |

- **V1'e bırakılan hatalar:**

| Hata | Sinyal | Faz |
|---|---|---|
| `rounded_back` | (kulak-omuz-kalça) veya (omuz-kalça-diz) açısının baseline'dan sapması | PULL/DESCENT |
| `hips_shoot_up` | PULL erken döneminde `hip_v_y / shoulder_v_y > 1.3` | PULL |

- **Multi-frame onay zorunlu:** Hata, **120–200ms boyunca** (timestamp delta ile ölçülen kesintisiz süre) tetiklenmeden raporlanmaz. Frame sayısı yerine ms kullanılır — 30 FPS'te bu ~4–6 frame'e karşılık gelir, ancak FPS değişirse pencere kendiliğinden uyarlanır.
- Confidence formülü: `min(1.0, |Z| / 4.0)` — Z=4 ve üzeri için max güven.

### Katman 8: 5 Alanlı Ağırlıklı Puanlama (Rep-Level Output)

| Alan | Z-Skor Aralığı | Anlamı |
|---|---|---|
| Area 1 | `\|Z\| < 1` | Mükemmel |
| Area 2 | `1 ≤ \|Z\| < 2` | İyi |
| Area 3 | `2 ≤ \|Z\| < 3` | Uyarı |
| Area 4 | `3 ≤ \|Z\| < 4` | Hata |
| Area 5 | `\|Z\| ≥ 4` | Ciddi hata |

**UI eşikleri:**
- `confidence < 0.3`: Sessiz (gösterme)
- `0.3 ≤ confidence < 0.6`: Sarı uyarı
- `confidence ≥ 0.6`: Kırmızı hata

### Rep çıktı şeması (örnek)

```json
{
  "rep_id": 3,
  "phase_durations_ms": {"setup": 800, "pull": 1200, "lockout": 400, "descent": 1100},
  "errors": {
    "incomplete_lockout": {"area": 1, "confidence": 0.05},
    "uncontrolled_descent": {"area": 3, "confidence": 0.42, "evidence": "descent 480ms vs pull 1200ms"}
  },
  "overall_grade": "needs_attention"
}
```

---

## 4. Modül Yapısı (Önerilen)

```
src/
├── common/                       # YENİ — squat + deadlift ortak yardımcıları
│   ├── __init__.py
│   └── geometry.py               # _angle, _angle_to_vertical, _distance, _choose_best_side
│                                 # (squat_features.py'den çıkarılan saf math util'leri)
│
├── sources/                      # MEVCUT — KORUNACAK
│   ├── video.py                  # video_frames(path) — frame + ts generator
│   └── webcam.py                 # webcam_frames(index) — aynı arayüz
│
├── pose_backends/                # MEVCUT — KORUNACAK
│   └── yolo26_adapter.py         # extract_yolo_pose_frame — deadlift de kullanır
│
├── vis/                          # MEVCUT — KORUNACAK
│   └── skeleton_drawer.py        # draw_landmarks_on_image_mediapipe / _yolo
│                                 # (DOKUNMA — sadece eklenebilir: draw_phase_label, draw_checkpoint_marker)
│
└── analysis/
    └── deadlift/
        ├── __init__.py
        ├── CLAUDE.md
        ├── pipeline.py                    # Ana orkestratör (8 katman sırayla)
        ├── capture.py                     # Katman 1 — MediaPipe + YOLO landmark çıkarımı
        ├── filters.py                     # Katman 2 — EMA / median filter
        ├── calibration.py                 # Katman 3 — standing baseline (TEK NOKTADA)
        ├── pre_checks.py                  # Pipeline başında çalışır — side-view doğrulaması
        │                                  # Sol-sağ omuz x-mesafesi < torso_length × 0.20 → side kabul
        │                                  # Aksi halde RejectedFrame döndürür (frame analiz edilmez)
        ├── deadlift_features.py           # Filtrelenmiş landmark → DeadliftFeatures dataclass
        ├── phase_detector.py              # Katman 4 — state machine (SETUP/PULL/LOCKOUT/DESCENT)
        ├── checkpoints.py                 # Katman 5 — T-açısı eşik yakalama
        ├── z_score.py                     # Katman 6 — referans karşılaştırma
        ├── error_detectors/
        │   ├── __init__.py
        │   ├── base.py                    # ErrorDetector abstract + multi-frame onay (squat _hold pattern'i)
        │   ├── incomplete_lockout.py
        │   └── uncontrolled_descent.py
        ├── rep_counter.py                 # SPRINT 2'de eklenir — squat counter şablonundan adapte
        ├── scoring.py                     # Katman 8 — area + confidence + UI eşikleri
        ├── deadlift_analyzer.py           # LiveDeadliftAnalyzer — pipeline'ı sarar (run_pose interface)
        ├── reference_distributions.json   # V0 literatür değerleri (kod-dışı yapılandırma)
        └── tests/
            ├── fixtures/
            └── test_*.py

inference/                        # MEVCUT — GÜNCELLENECEK
├── run_pose.py                   # --exercise {squat,deadlift} argümanı eklenecek
│                                 # Analyzer factory: SquatAnalyzer | DeadliftAnalyzer
│                                 # Hardcoded mutlak model yolu → göreli yola
│                                 # YOLO/squat import'larını lazy yap
│                                 # HUD/overlay genelleştir (RepFeedback yerine analiz çıktısı objesi)
└── webcam_pose.py                # Sadece skeleton+FPS — KORUNACAK
                                  # Tek değişiklik: hardcoded model yolunu göreli yap
                                  # draw_mediapipe/draw_yolo dublikatları kaldırılıp skeleton_drawer'a yönlendir
```

### Squat'tan miras alınan + yeniden kullanılan parçalar

| Parça | Kaynak | Hedef |
|---|---|---|
| `_angle`, `_angle_to_vertical`, `_distance` | `src/analysis/squat/squat_features.py` | `src/common/geometry.py` (taşı) |
| `_choose_best_side` | `src/analysis/squat/squat_features.py` | `src/common/geometry.py` (taşı) |
| Multi-frame onay pattern (`_hold`) | `src/analysis/squat/squat_rules.py:69-74` | `error_detectors/base.py` (kopyala + soyutlaştır) |
| Faz state machine iskeleti | `src/analysis/squat/squat_counter.py` | `phase_detector.py` (şablon — mantık tamamen farklı) |
| MediaPipe Landmarker oluşturma | `inference/run_pose.py:67-77` | `capture.py` |
| Frame/ts generator (video + webcam) | `src/sources/*.py` | Aynen kullan |
| Skeleton çizimi | `src/vis/skeleton_drawer.py` | Aynen kullan |
| YOLO landmark adapter | `src/pose_backends/yolo26_adapter.py` | Aynen kullan |
| HUD overlay (kutu çizimi, writer) | `inference/run_pose.py:27-64, 264-272` | Aynen kullan, generic hale getir |

### Squat'tan **kaçınılacak** anti-pattern'ler

1. **Kalibrasyonu birden fazla yere yayma.** Squat'ta baseline 3 ayrı dosyada (counter, rules, analyzer) tutuluyor. Deadlift'te SADECE `calibration.py` baseline tutar; counter/rules/detector hep oradan okur.
2. **Feature extractor'a view tespiti / iş mantığı sızdırma.** `deadlift_features.py` sadece geometri çıkarsın, karar vermesin. View doğrulaması → `pre_checks.py` (pipeline başında, feature'dan önce).
3. **Ham landmark'ı feature extractor'a sokma.** Pipeline sırası kesin: `capture → filter → calibration → features`. Filtrelenmemiş landmark'a feature uygulanmaz.
4. **Sabit eşikleri kodda tutma.** Tüm referans değerler `reference_distributions.json`'dan; `RuleThreshold` benzeri dataclass YOK.
5. **Analyzer'a kural mantığı koyma.** `deadlift_analyzer.py` sadece pipeline'ı çağırır + UI için sonuç paketler. Hata mantığı `error_detectors/`'da kalır.
6. **Binary `has_error: bool`.** Çıktı `{area: 1-5, confidence: 0.0-1.0}` formatında.
7. **Tek frame faz geçişi.** Her geçiş minimum 2-3 frame onaylı.
8. **`print()` ile debug.** `logging` modülü, seviyeli (DEBUG/INFO/WARNING).

### Tasarım kuralları
- Her katman **saf fonksiyon** veya **stateless class** — kolay test edilebilir olsun.
- Faz state machine ve calibration **stateful** — sınıf olarak tut.
- Yeni hata eklemek = `error_detectors/` altına yeni dosya, başka yere dokunma.
- Referans değerler **kodda hardcoded olmayacak** — JSON'dan oku.
- `src/vis/skeleton_drawer.py`'ye **dokunma**; yeni overlay'ler için ek fonksiyon yaz, mevcudunu değiştirme.

---

## 5. Yapım Sırası (Build Order)

Claude Code her adımda **Plan Mode**'da başlamalı, plan onaylandıktan sonra implementasyona geçmeli.

### Sprint 1 — İskelet (Katman 1–3) + Mevcut Kod Refactor

**Sprint 1.0 — Hazırlık (mevcut kodu temizle, ortak util'leri ayır)**

0. **Refactor güvenlik ağı oluştur** (DİĞER ADIMLARDAN ÖNCE)
   - `tests/smoke/test_squat_still_works.py` oluştur:
     - Bilinen bir squat videosu (`tests/fixtures/squat_known_good.mp4`) üzerinde mevcut sistemi çalıştır
     - Çıktıyı snapshot olarak kaydet: `tests/fixtures/squat_expected_output.json` (rep sayısı, hata raporları, faz geçişleri, varsa)
     - Test her çalıştığında snapshot ile karşılaştırır; fark varsa fail
   - Bu test refactor sürecinde **regresyon yakalayıcı** olarak çalışır
   - Adım 1-4 **bu test yeşil çalıştıktan sonra** başlar
   - Refactor sırasında her commit öncesi test koşulur

1. **`src/common/geometry.py` oluştur**
   - `src/analysis/squat/squat_features.py`'den şu fonksiyonları taşı:
     - `_angle(a, b, c)` → `angle(a, b, c)` (public)
     - `_angle_to_vertical(top, bottom)` → `angle_to_vertical(top, bottom)`
     - `_distance(a, b)` → `distance(a, b)`
     - `_choose_best_side(landmarks, left_map, right_map)` → genel yapıya çek (sabit `LEFT`/`RIGHT` map'lerini parametre yap)
   - Saf fonksiyonlar, hiçbir state yok, `print` yok.

2. **Squat'ın kırık import'larını düzelt** (deadlift'le ilgisiz ama bu olmadan run_pose çalışmıyor)
   - `src/analysis/squat/__init__.py` ekle (boş).
   - `squat_features.py`, `squat_counter.py`, `squat_rules.py`, `squat_analyzer.py`, `squat_features_yolo.py` içindeki:
     - `from src.analysis.squat_features` → `from src.analysis.squat.squat_features`
     - Diğer kardeş import'lar aynı mantıkla.
   - Squat'ı geometry util'lerini ortak modülden import edecek şekilde güncelle (yerel `_angle` vs.'i kaldır).

3. **`inference/run_pose.py` minimum hijyen** (deadlift entegrasyonuna hazırlık)
   - Hardcoded mutlak model yolu (`C:\Users\...\pose_landmarker_full.task`) → `models/pose_landmarker_full.task` (göreli, repo-root bazlı).
   - YOLO ve `extract_squat_features_yolo` import'larını lazy yap (`--backend yolo26` seçilince).
   - `--exercise {squat,deadlift}` argümanı ekle (default: `squat`). Şimdilik `deadlift` seçilirse `NotImplementedError`.
   - Analyzer factory: `make_analyzer(exercise: str)` — squat/deadlift'i ayır.
   - HUD/overlay (`draw_boxed_lines`, writer mantığı) **olduğu gibi kalsın** — sadece `RepFeedback` yerine generic bir analiz çıktısı objesi alacak şekilde imzayı esnet (ileride deadlift'in `area`+`confidence`'ını gösterecek).

4. **`inference/webcam_pose.py` minimum hijyen**
   - Hardcoded mutlak model yolunu göreli yap.
   - `draw_mediapipe`/`draw_yolo` lokal kopyalarını sil; `src.vis.skeleton_drawer`'dan import et. (skeleton_drawer dosyasına **dokunulmayacak**, sadece kullanım birleştirilecek.)

**Sprint 1.1 — Deadlift Katman 1-3**

1. **`capture.py`**
   - `inference/run_pose.py:67-77`'deki `create_landmarker` fonksiyonunu deadlift modülüne taşı, sınıflaştır:
     ```
     PoseCapture(model_path, running_mode)
       .process(frame_bgr, ts) -> RawLandmarks
     ```
   - 33 landmark + visibility skoru; `visibility < 0.5` → `None`, tamamen kayıp → `[-1, -1]` placeholder.
   - Video ve webcam giriş için iki ayrı **kullanım** (giriş kaynağı `src/sources/*.py`'dan, `capture.py` sadece pose extraction yapar).
   - YOLO yolu için aynı arayüzü `extract_yolo_pose_frame` üzerinden sar.

2. **`filters.py`**
   - `EMAFilter(alpha=0.3)` — landmark stream'ini bağımsız filtreler, `None` işaretliyi atla.
   - Alternatif: `MedianFilter(window=5)`.
   - Saf, stateless girişten stateful çıkışa class.
   - Birim test edilebilir: `[(0,0), (1,1), (2,2)]` → beklenen filtrelenmiş seri.

3. **`calibration.py`**
   - `StandingCalibrator()` — ilk 30 frame topla, hesap:
     - `torso_length`, `shoulder_width`, `femur_length`
   - `is_ready()`, `get_baseline()` arayüzü.
   - Kullanıcı hareketliyse / landmark eksikse "kalibrasyon başarısız" → uyarı sinyali.
   - **Squat'ın 3 yerdeki dağınık baseline'ı bu modüle toplanır; kimse başka yerde baseline tutmaz.**

4. **`deadlift_features.py` (iskelet)**
   - Filtrelenmiş + kalibre edilmiş landmark alır, geometri çıkarır:
     - `T` (gövde-dikey açı), `K` (diz açısı), `back_angle`, `hip.y`
     - `src/common/geometry.py`'dan import.
   - Sadece geometri; iş mantığı, view tespiti, eşik karşılaştırma YOK.

5. **`pre_checks.py`**
   - Pipeline başında, feature extraction'dan önce çalışır.
   - Side-view doğrulaması: `abs(left_shoulder.x - right_shoulder.x) < shoulder_width × 0.30` ise side kabul (kalibrasyondan gelen `shoulder_width` referans alınır).
   - Geçerli değilse `RejectedFrame(reason="not_side_view")` döndürür; pipeline o frame'i analiz etmez, kullanıcıya "Yan kameraya geç" uyarısı verilir.
   - Kalibrasyon henüz hazır değilse `pre_checks` pas geçer (ilk 30 frame için pasif).
   - Saf fonksiyon — state tutmaz.

6. **Test fixture'ları**
   - 1 örnek konvansiyonel deadlift videosu (`tests/fixtures/deadlift_clean_01.mp4`).
   - Beklenen `STANDING_BASELINE` değerleri JSON olarak (`tests/fixtures/baseline_clean_01.json`).
   - `pytest`:
     - `test_geometry.py` → `angle`, `angle_to_vertical` için bilinen vektör → açı eşleşmesi.
     - `test_filters.py` → EMA çıktı eşleşmesi.
     - `test_calibration.py` → fixture video → baseline JSON eşleşmesi (tolerans: ±%5).
     - `test_pre_checks.py` → side / front sentetik landmark girdileriyle accept / RejectedFrame ayrımı.

**Sprint 1 çıktısı:** Bir deadlift videosunu okuyup, filtrelenmiş + kalibre edilmiş landmark stream'i + standing baseline'ı **konsola/JSON'a** dökebilen yapı. Faz tespiti, hata, scoring HENÜZ YOK.

**Sprint 1 kabul kriterleri:**
- [ ] **Smoke test yeşil:** `pytest tests/smoke/test_squat_still_works.py` — refactor öncesi snapshot ile sonrası çıktı eşleşiyor (rep sayısı tam, error_labels seti tam, faz geçişleri ±50ms).
- [ ] `src/common/geometry.py` çalışıyor, squat onu kullanıyor (regresyon yok).
- [ ] `python inference/run_pose.py --exercise squat --mode video --path X.mp4` eskisi gibi çalışıyor.
- [ ] `python inference/run_pose.py --exercise deadlift ...` `NotImplementedError` veriyor (entegrasyon yer tutucusu).
- [ ] `pytest src/analysis/deadlift/tests/` yeşil.
- [ ] Hardcoded mutlak yol kalmadı.
- [ ] `skeleton_drawer.py` değişmedi (`git diff` boş).

### Sprint 2 — Faz tespiti (Katman 4) + Rep counter
1. `phase_detector.py` — state machine
2. `rep_counter.py` — phase_detector çıktısı üzerine bina edilen rep sayacı:
   - SETUP → PULL → LOCKOUT → DESCENT → SETUP tam turu tamamlandığında `rep_count += 1`
   - Squat counter (`src/analysis/squat/squat_counter.py`) şablon olarak alınır, deadlift fazlarına adapte edilir
   - Yarım kalan rep'ler (örn. lockout'a ulaşmadan iniş başlarsa) sayılmaz, `incomplete_rep` olarak loglanır
3. Görsel debug: video üzerine faz etiketi + rep counter overlay
4. **Test:** Bilinen 3 video için (a) faz geçiş zamanları (b) toplam rep sayısı manuel ile karşılaştır

### Sprint 3 — Checkpoint + Z-skor (Katman 5–6)
0. **Literatür taraması (manuel / araştırma işi — kod yok):** Z-skor kodu yazılmadan önce konvansiyonel deadlift biomekanik çalışmaları taranır; her checkpoint (T = 35°, 30°, 25°, 20°, 15°, 10°, 5°) için `K_mean`, `K_std`, `back_mean`, `back_std`, `hip_shoulder_velocity_ratio_mean`, `_std` değerleri **`reference_distributions.json`** dosyasına elle doldurulur. Bu adım tamamlanmadan z_score implementasyonuna başlanmaz.
1. `checkpoints.py` — T-açısı eşik yakalama
2. `reference_distributions.json` — literatür değerleri (Sub-task 0'da doldurulmuş hali bu adımda kod tarafından yüklenir)
3. `z_score.py` — checkpoint başına Z hesaplama

### Sprint 4 — Hata tespiti + skor (Katman 7–8)
1. `error_detectors/incomplete_lockout.py`
2. `error_detectors/uncontrolled_descent.py`
3. `scoring.py` — area + confidence + UI eşikleri
4. `pipeline.py` — hepsini bağlayan orkestratör

### Sprint 5 — Validation
1. **Test video sayısı:** 15–20 konvansiyonel deadlift videosu (kendi çekimi + açık veri karışık).
2. **Manuel etiketleme:** Kullanıcı (sen) her video için frame-bazlı doğru cevabı işaretler — her rep için: `incomplete_lockout` var mı (evet/hayır), `uncontrolled_descent` var mı, faz geçiş timestamp'leri.
3. Sistem çıktısı vs manuel etiket karşılaştırma — confusion matrix (TP/FP/TN/FN) çıkar.
4. False positive / false negative analizi; her hata için confidence eşiği ayarla.
5. **Hedef accuracy:** %70 (manuel etiketle uyum, rep-level).

**V0 → V1 geçiş tetikleyicisi:**
- Confidence eşikleri **%20'den fazla false positive** üretiyorsa V1 antrenör veri toplama başlatılır (literatür-tabanlı referans dağılımları yetersiz demektir).
- Accuracy %70 altındaysa: önce confidence eşik ayarı denenir; düzelmiyorsa V1'e geçilir.

**Sprint 5 kabul kriterleri:**
- [ ] 15–20 video etiketlenmiş.
- [ ] Rep-level accuracy ≥ %70.
- [ ] False positive oranı ≤ %20.
- [ ] Confusion matrix ve hata-bazlı confidence histogramları `tests/validation/report_v0.md` altında.

---

## 6. Teknik Tercihler

- **Dil:** Python 3.11+
- **Pose:** MediaPipe Pose (model_complexity=2, full body)
- **Video I/O:** OpenCV
- **Sayısal:** NumPy, SciPy (filtre için)
- **Test:** pytest
- **Format:** ruff + black
- **Zaman tabanı:** Tüm faz süreleri, hız ve ivme hesapları **`timestamp_ms`** üzerinden — frame sayısı üzerinden değil. Video FPS değişkenliği için zorunlu. (`Δhip.y / Δts_ms` = hız, `Δhız / Δts_ms` = ivme; multi-frame onay penceresi de ms cinsinden.)

### Yapılmayacaklar
- ❌ Sabit eşik (`if angle < X`) — Z-skor kullan
- ❌ Tek frame karar — multi-frame onay zorunlu
- ❌ Ham landmark — daima filtrelenmiş
- ❌ Mutlak piksel mesafesi — daima vücut oranına normalize
- ❌ Hardcoded referans — JSON'dan oku
- ❌ Binary çıktı — confidence skoru üret

---

## 7. Açık Sorular (Implementasyona Başlamadan Önce Cevap Lazım)

- [ ] Canlı akış mı yoksa video upload mu önce yapılacak? (MVP için video upload daha kolay) - Her ikisi de yapılacak kullanıcı seçimine bırakılacak. Bu dosyalar zaten mevcut sadece güncellenecek deadlift modülüne göre
- [ ] Test fixture videoları nereden gelecek? (Kendi çekimi mi, açık veri mi?) - İkisi de
- [ ] V1 antrenör veri toplama planı nedir? (Kaç kişi, kaç tekrar, hangi kameraya?)
- [ ] Çıktı görselleştirme (overlay) bu modülde mi yoksa ayrı UI katmanında mı? - Çıktı ayrı UI katmanında olacak fakat çıktı output klasörüne kaydedilecek.

---

## 8. Sözlük

- **T açısı:** Üst gövdenin (omuz–kalça hattı) dikey eksenle yaptığı açı. SETUP'ta yüksek (~40°+), LOCKOUT'ta sıfıra yakın.
- **K açısı:** Diz açısı — kalça → diz → ayak bileği vektörlerinin arasındaki açı.
- **Checkpoint:** Gövdenin önceden tanımlı bir T açısında olduğu an. Frame değil, "T=20° olduğu an" gibi içerik tabanlı.
- **Z-skor:** Kullanıcı ölçümünün referans dağılımdan kaç standart sapma uzakta olduğu.
- **Area:** Z-skor büyüklüğüne göre hata ciddiyet kademesi (1: mükemmel → 5: ciddi hata).
