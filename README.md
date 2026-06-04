# Gym Tracker

> **Poz Kestirimi Tabanlı Fitness Antrenörü — Gerçek Zamanlı Form Analizi ve Rutin Takibi**

Webcam veya video üzerinden squat ve deadlift egzersizlerini analiz eden, vücut iskeletini çıkaran, **tekrar sayan**, **form hatalarını tespit eden** ve **antrenman geçmişini takip eden** bir web uygulaması (şu an lokal kurulumla çalışır; public deploy yapılmamıştır).

Streamlit arayüzü, SQLite/PostgreSQL kalıcı veri katmanı ve MediaPipe + YOLO çift backend pose kestirimi ile inşa edilmiştir.

---

## Öne Çıkan Özellikler

- **İki egzersiz desteği**: Squat (kural-tabanlı + faz makinesi) ve Deadlift (8-katmanlı pipeline: kalibrasyon → faz → kontrol noktası → z-score → güven skoru).
- **Çift girdi**: Webcam ile **canlı analiz** (Streamlit + WebRTC) veya **video yükleme** ile sonradan analiz.
- **Overlay'li mp4 çıktısı**: İskelet + faz etiketi + rep sayacı + hata uyarısı çizili, tarayıcı-uyumlu H.264.
- **Rutin & set takibi**: Hedef set × tekrar tanımla, gerçekleşeni karşılaştır, oturum ↔ rutin bağlama.
- **Geçmiş & dashboard**: Egzersize göre günlük tekrar grafiği, hata frekansı, oturum bazlı detay inceleme.
- **Çift backend**: MediaPipe Pose Landmarker (varsayılan, düşük gecikme) ve YOLO26-Pose (alternatif benchmark).
- **Hibrit veri toplama altyapısı**: Faz etiketleme GUI'si + landmark çıkarımı + validation karşılaştırma araçları.

---

## Ekran Görüntüleri

### Ana Sayfa

<img width="900" alt="Ana Sayfa" src="https://github.com/user-attachments/assets/83a49ed7-8c2a-4997-8fc8-0b6ccaee4963" />

### Canlı Analiz

<img width="900" alt="Canlı Analiz" src="https://github.com/user-attachments/assets/125804b4-3e7e-4180-a981-7841aa6963a4" />

### Rutinler & Dashboard

| Rutinler | Dashboard |
|:---:|:---:|
| <img width="450" alt="Rutin Ekle" src="https://github.com/user-attachments/assets/8338aa81-7220-484d-803b-cf583e317ccc" /> | <img width="450" alt="Dashboard" src="https://github.com/user-attachments/assets/664d147e-b1d3-4fdf-a9bf-8519e37849e4" /> |

### Form Analizi & Tekrar Sayımı

**Squat**

| Form Analizi | Tekrar Sayımı |
|:---:|:---:|
| <img width="400" alt="Squat form analizi" src="https://github.com/user-attachments/assets/a8f5bff7-2140-4145-8132-fa97f06e7fbd" /> | <img width="400" alt="Squat tekrar sayımı" src="https://github.com/user-attachments/assets/17b4a0bb-619d-493d-af6f-a2d30e88ee7c" /> |

**Deadlift**

| Doğru Form | Hatalı Form |
|:---:|:---:|
| <img width="400" alt="Deadlift doğru form" src="https://github.com/user-attachments/assets/65b392ea-6718-46c2-80df-b1866b7d68b1" /> | <img width="400" alt="Deadlift hatalı form" src="https://github.com/user-attachments/assets/d66452c1-277c-47ed-98ad-c4a68eb1129f" /> |

### Analiz Sonuçları

**Squat**

<img width="900" alt="Squat analiz sonuçları" src="https://github.com/user-attachments/assets/1e69ae4f-3450-4069-8cc2-161b8d1fc72a" />

**Deadlift**

<img width="900" alt="Deadlift analiz sonuçları" src="https://github.com/user-attachments/assets/b031f7a7-4e58-43bf-b26d-3821599781de" />

### Geçmiş Dashboard

<img width="900" alt="Geçmiş Dashboard" src="https://github.com/user-attachments/assets/92d5e01e-b7a0-4fdf-ad01-2d3377062c33" />

---

## Teknoloji Stack

| Katman | Tercih | Neden |
|---|---|---|
| Dil | Python 3.11+ | Tek dil, analiz motoru zaten Python |
| UI | **Streamlit** + multi-page | Pipeline'a sıfır sürtünmeyle bağlanır |
| Canlı video | **streamlit-webrtc** | Tarayıcı kamerasını Python callback'ine taşır |
| Canlı (alternatif/yedek) | **OpenCV penceresi** (`inference/run_pose.py`) | Lokal akıcılık + ağ bağımsız demo |
| Pose backend | **MediaPipe Pose Landmarker** (default), **YOLO26/v8/11-Pose** | Hız ↔ doğruluk kıyası |
| Veritabanı | **SQLite** (default) / **PostgreSQL** opsiyonel | Lokal demo + bulut deploy yolu |
| ORM | SQLAlchemy 2.x | SQLite ↔ Postgres tek `DATABASE_URL` ile |
| Config | pydantic-settings + `.env` | Hardcoded yol yok |
| Sayısal | NumPy, SciPy | EMA / median filtre, geometrik metrikler |
| Video I/O | OpenCV, imageio-ffmpeg | H.264 transcode |
| Test | pytest | Storage + analiz birim testleri |

---

## Kurulum

### 1. Repo'yu klonla ve sanal ortam oluştur

```bash
git clone https://github.com/seydaoksuzoglu/gym-tracker.git
cd gym-tracker

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pose modelini indir

MediaPipe Pose Landmarker (Full) modelini `models/pose_landmarker_full.task` konumuna yerleştir:

- Resmi link: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- YOLO26-Pose ağırlığı opsiyoneldir (`yolo26m-pose.pt`); kullanmıyorsan görmezden gelebilirsin.

### 3. Ortam değişkenleri

```bash
# Windows:
copy .env.example .env
# macOS/Linux:
cp .env.example .env
```

`.env` içeriği (varsayılan SQLite):

```env
DATABASE_URL=sqlite:///data/gymtracker.db
MEDIAPIPE_MODEL_PATH=models/pose_landmarker_full.task
YOLO_MODEL_PATH=yolo26m-pose.pt
LOG_LEVEL=INFO
```

PostgreSQL/Neon'a geçmek istersen `DATABASE_URL`'i değiştirip `psycopg[binary]` paketini ekle.

### 4. Veritabanını oluştur

```bash
python -m src.storage.init_db
```

Bu komut `data/gymtracker.db` dosyasını ve tüm tabloları (`sessions`, `sets`, `reps`, `rep_errors`, `routines`, `routine_items`) oluşturur.

---

## Çalıştırma

### Streamlit arayüzü (önerilen)

```bash
python -m streamlit run app/Home.py
```

Tarayıcıda `http://localhost:8501` adresinde açılır. Sayfalar:

| Sayfa | İşlev |
|---|---|
| **Home** | Genel giriş + hızlı navigasyon kartları |
| **Canlı Analiz** | Webcam üzerinden gerçek zamanlı iskelet + rep + hata, set kontrol + DB persist |
| **Video Yükle** | mp4/mov yükle, overlay'li analiz videosu üret, DB'ye kaydet, rutine bağla |
| **Rutinler** | Rutin oluştur (egzersiz + hedef set × rep), ilerlemeyi gör, tamamla |
| **Geçmiş & Dashboard** | Tüm oturumlar, egzersize göre günlük tekrar grafiği, hata frekansı, oturum detayı |

### CLI / OpenCV penceresi (geliştirme + yedek yol)

WebRTC sorun çıkarırsa veya headless demo gerekirse:

```bash
python inference/run_pose.py --exercise deadlift --mode webcam
python inference/run_pose.py --exercise squat   --mode video --path data/test.mp4
```

---

## Proje Yapısı

```
gym-tracker/
├── app/                          # Streamlit UI
│   ├── Home.py                   # Ana sayfa
│   ├── _engine.py                # Analizör adapter (UI ↔ src/analysis dikişi)
│   ├── _persist.py               # DB yazma yardımcıları
│   ├── _styles.py                # Ortak CSS / tema
│   └── pages/
│       ├── 1_Canli_Analiz.py     # WebRTC canlı analiz
│       ├── 2_Video_Yukle.py      # Video upload + analiz
│       ├── 3_Rutinler.py         # Rutin CRUD + ilerleme
│       └── 4_Gecmis_Dashboard.py # Geçmiş + grafikler
│
├── src/
│   ├── config.py                 # pydantic Settings (.env okuma)
│   │
│   ├── storage/                  # Kalıcı veri katmanı
│   │   ├── models.py             # SQLAlchemy 2.x ORM tabloları
│   │   ├── database.py           # Engine + SessionLocal
│   │   ├── repository.py         # CRUD sözleşmesi
│   │   └── init_db.py            # Tablo oluştur
│   │
│   ├── analysis/
│   │   ├── squat/                # Squat: features + rules + counter
│   │   └── deadlift/             # Deadlift: 8-katmanlı pipeline
│   │
│   ├── pose_backends/            # MediaPipe + YOLO26 adapter'ları
│   ├── sources/                  # video.py, webcam.py (frame + ts generator)
│   ├── vis/                      # skeleton_drawer.py — overlay çizim
│   └── common/                   # Geometri yardımcıları
│
├── inference/                    # CLI / OpenCV penceresi yolu
│   ├── run_pose.py
│   └── webcam_pose.py
│
├── pipeline/                     # Veri etiketleme / dataset hazırlama
│   ├── label_gui.py              # Manuel faz etiketleme GUI'si
│   ├── extract_landmarks.py      # Video → pose koordinatları
│   └── extract_dataset.py        # Etiket + landmark birleştir
│
├── tools/                        # Geliştirici araçları
│   ├── run_validation.py         # Toplu validation: JSON + overlay'li mp4
│   ├── compare_validation.py     # Ground truth ile karşılaştırma
│   ├── label_phases.py           # Faz etiketleme yardımcısı
│   └── debug_side_view.py        # Tek video canlı debug
│
├── models/                       # pose_landmarker_full.task
├── data/                         # gymtracker.db (gitignored)
├── outputs/                      # Analiz çıktıları (overlay video)
├── tests/                        # pytest birim testleri
├── .streamlit/config.toml        # Tema (emerald)
├── requirements.txt
└── .env.example
```

---

## Mimari

Üç katmanlı kesin sorumluluk ayrımı:

```
┌──────────────────────────────────────────────────────────┐
│ app/  (Streamlit)                                        │
│   - sadece sunum + DB yazma çağrıları                    │
│   - analizör çağrısı: app/_engine.py adapter üzerinden   │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ src/analysis/  (saf analiz)                              │
│   - sadece JSON / dataclass döner                        │
│   - DB bilmez, UI bilmez                                 │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ src/storage/  (kalıcılık)                                │
│   - analiz çıktısını okur, ORM ile yazar                 │
│   - UI bilmez                                            │
└──────────────────────────────────────────────────────────┘
```

Bu sınır, ileride FastAPI veya ayrı frontend (React/Next) eklenmek istenirse hazır dikiştir: `app/` katmanı HTTP istemcisine, analizör çağrısı bir endpoint'e dönüştürülebilir.

---

## Veritabanı Şeması

```
sessions      (id, started_at, ended_at, source, video_path?, routine_id?)
sets          (id, session_id, exercise_key, set_index,
               target_reps, completed_reps, backend, started_at?, ended_at?)
reps          (id, set_id, rep_index, overall_grade,
               phase_durations_ms JSON, video_ts_ms?, created_at)
rep_errors    (id, rep_id, error_type, area, confidence, evidence JSON)

routines      (id, name, created_at, completed_at?)
routine_items (id, routine_id, exercise_key, target_sets, target_reps, order_index)
```

İlişkiler: `session 1—N set 1—N rep 1—N rep_error`. Rep formatı analizörün ürettiği JSON ile birebir eşleşir — tek doğruluk kaynağı.

---

## Deadlift Pipeline (8 Katman)

Deadlift modülü kendi `CLAUDE.md`'sinde detaylanmış 8 katmanlı bir pipeline kullanır:

1. **Capture** — frame + timestamp
2. **PreChecks** — yan görünüm, vücut tamlığı
3. **Filter** — EMA / median pürüzsüzleştirme
4. **Normalize** — torso uzunluğuna oranlama
5. **Phase** — `setup → pull → lockout → descent` durum makinesi
6. **Checkpoint** — faz geçişlerinde kural kontrolü
7. **Z-Score** — kullanıcının kendi standing kalibrasyonuna göre sapma
8. **Confidence & Score** — `{area: 1–5, confidence: 0.0–1.0, phase_durations_ms}` çıktısı

Hata türleri: `incomplete_lockout` (kilit tamamlanmamış), `uncontrolled_descent` (iniş kontrolsüz), `incomplete` (yarım tekrar).

---

## Validation Araçları

Deadlift analizinin doğruluğunu ölçmek için:

```bash
# Tüm validation videolarını analiz et, JSON + overlay'li mp4 üret
python tools/run_validation.py

# Manuel ground truth ile sistem çıktısını karşılaştır
python tools/compare_validation.py
```

Çıktılar:
- JSON: `tests/fixtures/validation/system_output/*.json`
- Overlay mp4: `outputs/validation/*_overlay.mp4`

---

## Pose Backend Benchmark

Farklı pose modelleri karşılaştırıldı:

| Backend | Hız | Doğruluk | Not |
|---|---|---|---|
| **MediaPipe Pose Landmarker** | Yüksek | Orta-Yüksek | Düşük gecikme, canlı için varsayılan |
| **YOLO26-Pose** | Orta | Yüksek | Tekrar sayımında daha güçlü |
| YOLOv8-Pose | Orta | Orta | Genel kullanım |
| YOLO11-Pose | Orta-Düşük | Yüksek | Daha güncel |

CPU üzerinde YOLO tabanlı modeller gerçek zamanlı kullanımda yetersiz kaldı; canlı analizde MediaPipe varsayılan.

---

## Test

```bash
pytest                    # Tüm testler
pytest tests/test_storage.py -v   # Sadece storage testleri
```

Storage testleri bellek-içi SQLite ile çalışır — gerçek `data/gymtracker.db` etkilenmez.

---

## Yol Haritası

- [x] Squat: kural-tabanlı analiz + tekrar sayımı
- [x] Deadlift: 8-katmanlı pipeline + faz tespiti + hata detektörleri
- [x] Streamlit UI: video upload + canlı webrtc + dashboard + rutinler
- [x] SQLAlchemy storage + rutin ↔ oturum bağlama
- [x] Overlay'li mp4 çıktı + H.264 transcode
- [ ] Deadlift validation accuracy ≥ %70
- [ ] Push-up / biceps curl modülleri
- [ ] Hibrit model (zaman-serisi ML) entegrasyonu
- [ ] React / Next.js frontend + FastAPI ayrımı
- [ ] Multi-user (user tablosu + auth)

---

## Konvansiyonlar

- Hardcoded yol yok — tüm yollar `src/config.py` üzerinden `.env`'den okunur.
- `print()` yerine `logging` (DEBUG/INFO/WARNING).
- Sırlar commit edilmez — yalnızca `.env.example`.
- `src/vis/skeleton_drawer.py`'nin **mevcut fonksiyonlarının imza/davranışı değiştirilmez**, yeni overlay'ler için yeni fonksiyon eklenir.
- Analiz ↔ Storage ↔ UI sınırı korunur (tek yön import).

---

## Lisans

Bu proje bir bitirme projesi kapsamında geliştirilmektedir. Lisans bilgisi proje sahibine aittir.

---

## Modül Dokümanları

- [`CLAUDE.md`](CLAUDE.md) — Proje kök planı (mimari + sprint yapısı)
- [`src/analysis/deadlift/CLAUDE.md`](src/analysis/deadlift/CLAUDE.md) — Deadlift pipeline detayı (8 katman)
