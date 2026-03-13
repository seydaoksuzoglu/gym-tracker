import os
import json
import glob
import math
import argparse


# --------------------------------------------------
# KLASÖR OLUŞTURMA
# --------------------------------------------------
def ensure_dir(path: str):
    """
    Verilen klasör yoksa oluşturur.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# JSON OKUMA
# --------------------------------------------------
def load_json(json_path: str):
    """
    Verilen JSON dosyasını okuyup Python dict olarak döndürür.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# ETİKET DOSYALARINI BULMA
# --------------------------------------------------
def get_label_files(labels_root: str):
    """
    labels_root altında bulunan tüm .json dosyalarını bulur.
    """
    search_pattern = os.path.join(labels_root, "**", "*.json")
    return glob.glob(search_pattern, recursive=True)


# --------------------------------------------------
# LABEL JSON -> LANDMARK JSON EŞLEŞTİRME
# --------------------------------------------------
def get_landmark_json_path(label_json_path: str, labels_root: str, landmarks_root: str):
    """
    Etiket dosyasının relative path'ini kullanarak ilgili landmark JSON yolunu üretir.

    Örnek:
    labels_root    = video_labels/squat
    label_json     = video_labels/squat/set1/video_001.json
    landmarks_root = landmarks/squat

    çıktı:
    landmarks/squat/set1/video_001_landmarks.json
    """
    relative_label_path = os.path.relpath(label_json_path, labels_root)
    relative_without_ext = os.path.splitext(relative_label_path)[0]
    return os.path.join(landmarks_root, relative_without_ext + "_landmarks.json")


# --------------------------------------------------
# TEMEL GEOMETRİ FONKSİYONLARI
# --------------------------------------------------
def vector_sub(a, b):
    """
    a - b vektörünü döndürür.
    a ve b aynı boyutta liste/tuple olmalıdır.
    """
    return [a[i] - b[i] for i in range(len(a))]


def vector_norm(v):
    """
    Bir vektörün uzunluğunu döndürür.
    """
    return math.sqrt(sum(x * x for x in v))


def angle_between_vectors(v1, v2):
    """
    İki vektör arasındaki açıyı derece cinsinden döndürür.
    """
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)

    if norm1 == 0 or norm2 == 0:
        return None

    dot = sum(v1[i] * v2[i] for i in range(len(v1)))
    cos_theta = dot / (norm1 * norm2)

    # Sayısal taşmaları önlemek için clamp
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)


def angle_3pt(a, b, c):
    """
    Üç nokta verildiğinde B noktasındaki açıyı hesaplar.
    Yani angle ABC döner.
    """
    ba = vector_sub(a, b)
    bc = vector_sub(c, b)
    return angle_between_vectors(ba, bc)


# --------------------------------------------------
# LANDMARK ALMA
# --------------------------------------------------
def get_landmark_point(frame_record, landmark_idx, prefer_world=True):
    """
    Bir frame içinden istenen landmark'ı alır.
    Önce world_landmarks denenir, yoksa image_landmarks kullanılır.

    Dönen değer:
    {
        "point": [x, y, z] veya [x, y],
        "visibility": ...,
        "presence": ...,
        "source": "world" veya "image"
    }
    """
    world_landmarks = frame_record.get("world_landmarks", [])
    image_landmarks = frame_record.get("image_landmarks", [])

    if prefer_world and len(world_landmarks) > landmark_idx:
        lm = world_landmarks[landmark_idx]
        return {
            "point": [lm["x"], lm["y"], lm["z"]],
            "visibility": lm.get("visibility", 0.0),
            "presence": lm.get("presence", 0.0),
            "source": "world"
        }

    if len(image_landmarks) > landmark_idx:
        lm = image_landmarks[landmark_idx]
        return {
            "point": [lm["x"], lm["y"]],
            "visibility": lm.get("visibility", 0.0),
            "presence": lm.get("presence", 0.0),
            "source": "image"
        }

    return None


def get_image_landmark_point(frame_record, landmark_idx):
    """
    Özellikle image koordinatı gerektiğinde kullanılır.
    Örneğin gövde eğimini ekran dikeyine göre hesaplamak için.
    """
    image_landmarks = frame_record.get("image_landmarks", [])
    if len(image_landmarks) > landmark_idx:
        lm = image_landmarks[landmark_idx]
        return {
            "point": [lm["x"], lm["y"]],
            "visibility": lm.get("visibility", 0.0),
            "presence": lm.get("presence", 0.0),
            "source": "image"
        }
    return None


# --------------------------------------------------
# FRAME BAZLI FEATURE ÇIKARIMI
# --------------------------------------------------
def compute_frame_features(frame_record):
    """
    Tek bir frame için squat açısından temel feature'ları çıkarır.

    Kullanılan MediaPipe landmark indeksleri:
    11 left_shoulder
    12 right_shoulder
    23 left_hip
    24 right_hip
    25 left_knee
    26 right_knee
    27 left_ankle
    28 right_ankle
    """
    # Diz açısı için world varsa world kullan, yoksa image
    l_shoulder = get_landmark_point(frame_record, 11, prefer_world=True)
    r_shoulder = get_landmark_point(frame_record, 12, prefer_world=True)
    l_hip = get_landmark_point(frame_record, 23, prefer_world=True)
    r_hip = get_landmark_point(frame_record, 24, prefer_world=True)
    l_knee = get_landmark_point(frame_record, 25, prefer_world=True)
    r_knee = get_landmark_point(frame_record, 26, prefer_world=True)
    l_ankle = get_landmark_point(frame_record, 27, prefer_world=True)
    r_ankle = get_landmark_point(frame_record, 28, prefer_world=True)

    # Gövde eğimi için image koordinatı daha mantıklı
    l_shoulder_img = get_image_landmark_point(frame_record, 11)
    r_shoulder_img = get_image_landmark_point(frame_record, 12)
    l_hip_img = get_image_landmark_point(frame_record, 23)
    r_hip_img = get_image_landmark_point(frame_record, 24)

    left_knee_angle = None
    right_knee_angle = None
    left_hip_angle = None
    right_hip_angle = None
    left_torso_lean_deg = None
    right_torso_lean_deg = None

    # Sol diz açısı: hip-knee-ankle
    if l_hip and l_knee and l_ankle:
        left_knee_angle = angle_3pt(l_hip["point"], l_knee["point"], l_ankle["point"])

    # Sağ diz açısı: hip-knee-ankle
    if r_hip and r_knee and r_ankle:
        right_knee_angle = angle_3pt(r_hip["point"], r_knee["point"], r_ankle["point"])

    # Sol kalça açısı: shoulder-hip-knee
    if l_shoulder and l_hip and l_knee:
        left_hip_angle = angle_3pt(l_shoulder["point"], l_hip["point"], l_knee["point"])

    # Sağ kalça açısı: shoulder-hip-knee
    if r_shoulder and r_hip and r_knee:
        right_hip_angle = angle_3pt(r_shoulder["point"], r_hip["point"], r_knee["point"])

    # Sol gövde eğimi: hip->shoulder vektörü ile görüntü dikeyi arasındaki açı
    if l_shoulder_img and l_hip_img:
        torso_vec = vector_sub(l_shoulder_img["point"], l_hip_img["point"])
        vertical_up = [0.0, -1.0]  # Görüntüde yukarı yön
        left_torso_lean_deg = angle_between_vectors(torso_vec, vertical_up)

    # Sağ gövde eğimi
    if r_shoulder_img and r_hip_img:
        torso_vec = vector_sub(r_shoulder_img["point"], r_hip_img["point"])
        vertical_up = [0.0, -1.0]
        right_torso_lean_deg = angle_between_vectors(torso_vec, vertical_up)

    return {
        "frame_index": frame_record["frame_index"],
        "timestamp_sec": frame_record["timestamp_sec"],
        "detected": frame_record.get("detected", False),
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_torso_lean_deg": left_torso_lean_deg,
        "right_torso_lean_deg": right_torso_lean_deg,
    }


# --------------------------------------------------
# REP ARALIĞINI LANDMARK DOSYASINDAN KESME
# --------------------------------------------------
def slice_rep_frames(landmark_data, start_frame, end_frame):
    """
    Landmark JSON içindeki tüm frames listesinden,
    sadece start_frame-end_frame aralığını seçer.
    """
    selected = []
    for frame_record in landmark_data.get("frames", []):
        idx = frame_record["frame_index"]
        if start_frame <= idx <= end_frame:
            selected.append(frame_record)
    return selected


# --------------------------------------------------
# REP İÇİN ÖZET İSTATİSTİK
# --------------------------------------------------
def safe_min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def safe_max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def value_at_frame(frame_features, target_frame_idx, key):
    """
    İstenen frame_index'teki feature değerini döndürür.
    """
    for item in frame_features:
        if item["frame_index"] == target_frame_idx:
            return item.get(key)
    return None


def compute_rep_summary(frame_features, rep_info):
    """
    Bir rep'in tüm frame feature'larından özet metrik üretir.
    """
    left_knees = [f["left_knee_angle"] for f in frame_features]
    right_knees = [f["right_knee_angle"] for f in frame_features]
    left_hips = [f["left_hip_angle"] for f in frame_features]
    right_hips = [f["right_hip_angle"] for f in frame_features]
    left_lean = [f["left_torso_lean_deg"] for f in frame_features]
    right_lean = [f["right_torso_lean_deg"] for f in frame_features]

    bottom_frame = rep_info.get("bottom_frame")

    summary = {
        "num_frames": len(frame_features),
        "min_left_knee_angle": safe_min(left_knees),
        "min_right_knee_angle": safe_min(right_knees),
        "min_left_hip_angle": safe_min(left_hips),
        "min_right_hip_angle": safe_min(right_hips),
        "max_left_torso_lean_deg": safe_max(left_lean),
        "max_right_torso_lean_deg": safe_max(right_lean),
        "bottom_left_knee_angle": value_at_frame(frame_features, bottom_frame, "left_knee_angle"),
        "bottom_right_knee_angle": value_at_frame(frame_features, bottom_frame, "right_knee_angle"),
        "bottom_left_hip_angle": value_at_frame(frame_features, bottom_frame, "left_hip_angle"),
        "bottom_right_hip_angle": value_at_frame(frame_features, bottom_frame, "right_hip_angle"),
        "start_left_knee_angle": value_at_frame(frame_features, rep_info["start_frame"], "left_knee_angle"),
        "start_right_knee_angle": value_at_frame(frame_features, rep_info["start_frame"], "right_knee_angle"),
        "end_left_knee_angle": value_at_frame(frame_features, rep_info["end_frame"], "left_knee_angle"),
        "end_right_knee_angle": value_at_frame(frame_features, rep_info["end_frame"], "right_knee_angle"),
    }

    return summary


# --------------------------------------------------
# TEK REP DOSYASI ÜRETME
# --------------------------------------------------
def build_single_rep_record(label_data, rep_info, rep_frames, frame_features):
    """
    Tek rep için kaydedilecek JSON yapısını oluşturur.
    """
    rep_summary = compute_rep_summary(frame_features, rep_info)

    return {
        "video_name": label_data["video_name"],
        "video_path": label_data["video_path"],
        "exercise": label_data["exercise"],
        "fps": label_data["fps"],
        "total_frames": label_data["total_frames"],
        "rep_id": rep_info["rep_id"],
        "start_frame": rep_info["start_frame"],
        "bottom_frame": rep_info["bottom_frame"],
        "end_frame": rep_info["end_frame"],
        "start_sec": rep_info.get("start_sec"),
        "bottom_sec": rep_info.get("bottom_sec"),
        "end_sec": rep_info.get("end_sec"),
        "labels": rep_info["labels"],
        "summary_features": rep_summary,
        "frame_features": frame_features,
        # İstersen bunu sonra kapatabilirsin.
        # Şimdilik doğrulama için ham frame bilgisi de saklıyoruz.
        "raw_rep_frames": rep_frames,
    }


# --------------------------------------------------
# ÇIKTI DOSYASI YOLU
# --------------------------------------------------
def get_rep_output_path(label_json_path, labels_root, output_root, rep_id):
    """
    Her rep için ayrı JSON dosyası üretir.

    Örnek:
    video_labels/squat/set1/video_001.json
    ->
    dataset/squat/set1/video_001_rep_1.json
    """
    relative_label_path = os.path.relpath(label_json_path, labels_root)
    relative_without_ext = os.path.splitext(relative_label_path)[0]
    return os.path.join(output_root, relative_without_ext + f"_rep_{rep_id}.json")


# --------------------------------------------------
# ANA İŞLEM
# --------------------------------------------------
def process_all(labels_root: str, landmarks_root: str, output_root: str):
    """
    Tüm etiket dosyalarını okuyup landmark dosyaları ile eşleştirir.
    Sonra rep bazlı dataset üretir.
    """
    label_files = get_label_files(labels_root)

    if not label_files:
        print(f"[HATA] '{labels_root}' içinde etiket JSON bulunamadı.")
        return

    print(f"Toplam {len(label_files)} label dosyası bulundu.")

    dataset_index = []

    for label_json_path in label_files:
        print("\n" + "=" * 60)
        print(f"Label JSON: {label_json_path}")

        landmark_json_path = get_landmark_json_path(label_json_path, labels_root, landmarks_root)

        if not os.path.isfile(landmark_json_path):
            print(f"[UYARI] Landmark JSON bulunamadı: {landmark_json_path}")
            continue

        label_data = load_json(label_json_path)
        landmark_data = load_json(landmark_json_path)

        # Güvenlik kontrolü: aynı video mu?
        if label_data.get("video_path") != landmark_data.get("video_path"):
            print("[UYARI] video_path eşleşmiyor.")
            print("  Label   :", label_data.get("video_path"))
            print("  Landmark:", landmark_data.get("video_path"))

        reps = label_data.get("reps", [])
        print(f"Rep sayısı: {len(reps)}")

        for rep_info in reps:
            rep_id = rep_info["rep_id"]
            start_frame = rep_info["start_frame"]
            end_frame = rep_info["end_frame"]

            # Rep frame aralığını uzun landmark dosyasından kes
            rep_frames = slice_rep_frames(landmark_data, start_frame, end_frame)

            if not rep_frames:
                print(f"[UYARI] rep_id={rep_id} için frame bulunamadı.")
                continue

            # Her frame için feature hesapla
            frame_features = [compute_frame_features(frame_record) for frame_record in rep_frames]

            # Tek rep kaydı oluştur
            rep_record = build_single_rep_record(label_data, rep_info, rep_frames, frame_features)

            # Dosyaya yaz
            rep_output_path = get_rep_output_path(label_json_path, labels_root, output_root, rep_id)
            ensure_dir(os.path.dirname(rep_output_path))

            with open(rep_output_path, "w", encoding="utf-8") as f:
                json.dump(rep_record, f, ensure_ascii=False, indent=2)

            print(f"[KAYIT] rep_id={rep_id} -> {rep_output_path}")

            dataset_index.append({
                "rep_output_path": os.path.relpath(rep_output_path, output_root),
                "video_name": label_data["video_name"],
                "video_path": label_data["video_path"],
                "exercise": label_data["exercise"],
                "rep_id": rep_id,
                "labels": rep_info["labels"]
            })

    # Bütün rep dosyalarının özet index'i
    index_path = os.path.join(output_root, "dataset_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(dataset_index, f, ensure_ascii=False, indent=2)

    print("\nBitti.")
    print(f"Dataset index kaydedildi: {index_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landmark JSON + label JSON dosyalarından rep bazlı dataset üretir."
    )

    parser.add_argument(
        "--labels_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\data\squat\video_labels",
        help="Etiket JSON dosyalarının olduğu klasör"
    )

    parser.add_argument(
        "--landmarks_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\data\squat\landmarks",
        help="Landmark JSON dosyalarının olduğu klasör"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\data\squat\dataset",
        help="Rep bazlı dataset çıktılarının yazılacağı klasör"
    )

    args = parser.parse_args()

    ensure_dir(args.labels_root)
    ensure_dir(args.landmarks_root)
    ensure_dir(args.output_root)

    process_all(args.labels_root, args.landmarks_root, args.output_root)