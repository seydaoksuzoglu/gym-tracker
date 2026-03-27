# Poz Kestirimi Tabanlı Fitness Antrenörü  
## Gerçek Zamanlı Form Analizi ve Rutin Takibi

Bu proje, poz kestirimi tabanlı görüntü işleme yöntemleri kullanarak kullanıcının egzersizlerini **webcam** veya **video** üzerinden analiz eden bir dijital fitness antrenörü geliştirmeyi amaçlamaktadır. Sistem, egzersiz sırasında kullanıcının vücut noktalarını çıkarır, hareketi değerlendirir, **tekrar sayımı** yapar, **form hatalarını** tespit eder ve kullanıcıya **anlık geri bildirim** sunar.

Projenin ana hedefi, kullanıcıların bir antrenör desteği olmadan da egzersizlerini daha kontrollü, güvenli ve ölçülebilir şekilde yapabilmesini sağlamaktır.

---

## Proje Amacı

Bu proje kapsamında, poz kestirimi temelli bir analiz sistemi geliştirilerek özellikle form hatasına açık egzersizlerde kullanıcının hareket kalitesinin değerlendirilmesi hedeflenmiştir. Sistem yalnızca tekrar sayımı yapan bir yapı olarak değil, aynı zamanda hatalı form durumlarında kullanıcıyı açıklayıcı uyarılarla yönlendiren bir yardımcı antrenör olarak tasarlanmıştır.

Bunun yanında proje, yapılan antrenmanların set, tekrar ve rutin düzeyinde takip edilebileceği bir altyapı kurmayı amaçlamaktadır. Böylece kullanıcı hem anlık geri bildirim alabilmekte hem de zaman içindeki gelişimini izleyebilmektedir.

---

## Kapsam

Proje mimarisi farklı egzersizleri destekleyecek şekilde düşünülmüştür. Şu an geliştirme odağı özellikle **squat** egzersizi üzerindedir. Sistem yapısı ilerleyen aşamalarda **deadlift** ve **push-up** gibi hareketlere de genişletilecek şekilde tasarlanmıştır.

---

## Temel Özellikler

- Webcam ve video girdisi ile çalışma
- İnsan vücudu landmark / keypoint çıkarımı
- Gerçek zamanlı iskelet görselleştirmesi
- Squat için tekrar sayımı
- Squat için form-hata tespiti
- Önden / yandan görünüm ayrımı
- Kural tabanlı analiz altyapısı
- Benchmark ve performans karşılaştırmaları
- Veri etiketleme ve veri seti hazırlama altyapısı
- Hibrit analiz yaklaşımı için hazırlık

---

## Kullanılan Yaklaşımlar

Projede iki temel yaklaşım bir araya getirilmektedir:

### 1. Kural tabanlı analiz
Açıklanabilir ve anlık geri bildirim verebilen yapı bu kısımdır. Eklem açıları, hareket evreleri, görünüm yönü ve belirli biyomekanik kurallar üzerinden hata tespiti yapılır.

### 2. Makine öğrenmesi destekli hibrit yaklaşım
Sadece sabit eşiklere dayalı sistemlerin sınırlı kaldığı durumlarda daha esnek analiz yapabilmek için zaman serisi tabanlı öğrenme modelleri planlanmıştır. Bu doğrultuda veri hazırlama ve etiketleme süreci başlatılmıştır.

---

## Sistem Akışı

Projenin temel işleyişi aşağıdaki gibidir:

1. Egzersiz seçimi yapılır  
2. Girdi kaynağı belirlenir (`webcam` veya `video`)  
3. Poz / landmark çıkarımı yapılır  
4. Kullanıcının görüş yönü belirlenir  
5. Form analizi gerçekleştirilir  
6. Tekrar sayımı güncellenir  
7. Kullanıcıya geri bildirim sunulur  
8. Sonuçlar gösterilir ve kayıt altına alınır  

---

## Kural Tabanlı Squat Analizi

Squat özelinde geliştirilen kural tabanlı yapı üç temel modül üzerinden çalışmaktadır:

### `squat_features.py`
Ham pose koordinatlarını analizde kullanılabilecek özniteliklere dönüştürür.

Örnek öznitelikler:
- Diz açısı
- Kalça açısı
- Topuk kalkma oranı
- Diz hizası sapması
- Görüş yönüne göre güvenilir taraf seçimi

### `squat_rules.py`
Squat hareketine ait biyomekanik kuralları uygular ve form hatalarını tespit eder.

Örnek kontroller:
- Yetersiz derinlik
- Topuk kalkması
- Gövde eğimi
- Diz hizasının bozulması

### `squat_counter.py`
Squat hareketini bir durum makinesi olarak takip eder ve tekrar sayımını yapar.

Takip edilen fazlar:
- `standing`
- `descent`
- `bottom`
- `ascent`

Bu yapı sayesinde yalnızca tam hareket döngüsünü tamamlayan tekrarlar geçerli sayılır.

---

## Veri Hazırlama Süreci

Projenin hibrit analiz kısmı için squat videoları toplanmış ve etiketleme süreci başlatılmıştır.

### `label_gui.py`
Her tekrarın başlangıç, dip nokta ve bitiş karelerini manuel olarak işaretlemek için geliştirilmiştir.

### `extract_landmarks.py`
Videolardan MediaPipe veya YOLO tabanlı pose çıktıları alınarak iskelet koordinatları dışa aktarılır.

### `extract_dataset.py`
Etiketler ile pose verilerini birleştirerek zaman serisi modellerinde kullanılabilecek yapılandırılmış veri setini üretir.

---

## Kullanılan Teknolojiler

- Python
- OpenCV
- MediaPipe
- YOLO tabanlı pose modelleri
- NumPy
- Matplotlib
- Özel kural tabanlı analiz modülleri

---

## Benchmark Sonuçları

Projede farklı poz kestirimi yöntemleri performans açısından karşılaştırılmıştır. Yapılan testlerde özellikle aşağıdaki yapılar değerlendirilmiştir:

- MediaPipe
- YOLO26-Pose
- YOLOv8-Pose
- YOLO11-Pose

### Genel gözlemler
- **MediaPipe**, hız ve gecikme açısından en yüksek performansı vermiştir.
- **YOLO26-Pose**, tekrar sayımı ve bazı analiz senaryolarında daha güçlü sonuçlar sunmuştur.
- CPU üzerinde çalışan YOLO tabanlı yapıların gerçek zamanlı kullanım açısından yetersiz kaldığı gözlemlenmiştir.
- Form-hata tespiti probleminin, tekrar sayımına göre daha zor olduğu görülmüştür.

Bu sonuçlar, projede neden **hibrit bir yapıya** ihtiyaç duyulduğunu desteklemektedir.

---

## Mevcut Durum

Şu an proje kapsamında:

- Squat için pose tabanlı analiz akışı kurulmuştur
- Squat tekrar sayımı geliştirilmiştir
- Squat için kural tabanlı form analizi oluşturulmuştur
- MediaPipe ve YOLO tabanlı benchmark çalışmaları yapılmıştır
- Veri hazırlama ve etiketleme altyapısı başlatılmıştır

---

## Planlanan Geliştirmeler

- Deadlift ve push-up egzersizleri için analiz modüllerinin eklenmesi
- Hibrit model eğitiminin tamamlanması
- Form-hata tespit doğruluğunun artırılması
- Rutin ve antrenman geçmişi takibi
- Web arayüzü / kullanıcı paneli entegrasyonu
- Daha güçlü kullanıcı geri bildirim sistemi
- Çok egzersizli tek platform yapısı

---

## Proje Dizini

```bash
project/
│
├── src/
│   ├── analysis/
│   │   ├── squat_features.py
│   │   ├── squat_rules.py
│   │   ├── squat_counter.py
│   │   └── ...
│   │
│   ├── pipeline/
│   │   ├── label_gui.py
│   │   ├── extract_landmarks.py
│   │   ├── extract_dataset.py
│   │   └── ...
│   │
│   ├── vis/
│   ├── sources/
│   └── ...
├── requirements.txt
└── README.md
