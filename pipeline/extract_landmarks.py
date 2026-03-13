import cv2
import os
import json
import glob
import argparse
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# KLASÖR OLUŞTURMA
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)
    
# VİDEO DOSYALARINI BULMA
def get_video_files(folder_path: str):
    """
    Kök klasör içindeki tüm .mp4 videolarını bulur.
    Alt klasörlere de recursive=True sayesinde girer
    """
    search_pattern = os.path.join(folder_path, "**", "*.mp4")
    return glob.glob(search_pattern, recursive=True)

# ÇIKTI DOSYASI YOLU OLUŞTURMA
def get_output_json_path(video_path: str, input_root: str, output_root: str):
    """
    Her video için ayrı landmark JSON dosyası oluşturur.

    Örnek: 
    input_root  = videos/squat
    video_path  = videos/squat/set1/video_01.mp4
    output_root = landmarks/squat

    çıktı:
    landmarks/squat/set1/video_01_landmarks.json
    """
    relative_video_path = os.path.relpath(video_path, input_root)
    relative_without_ext = os.path.splitext(relative_video_path)[0]
    return os.path.join(output_root, relative_without_ext + "_landmarks.json")

# LANDMARK'LARI JSON'A UYGUN HALE GETİRME
def landmark_to_dict(lm):
    """
    MediaPipe image landmark nesnesini Python dict'e çevirir.
    Bu landmark görüntü koordinat sistemindedir.
    x ve y genelde 0-1 arası normalize değerdir.
    z derinlik benzeri göreli bilgidir.
    """
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(getattr(lm, "visibility", 0.0)),
        "presence": float(getattr(lm, "presence", 0.0)),
    }

def world_landmark_to_dict(lm):
    """
    MediaPipe world landmark nesnesini dict'e çevirir.
    Bu koordinatlar 3B uzayda daha anlamlı pozisyon bilgisi verir.
    Squat analizi yaparken ilerde daha faydalı olabilir.
    """
    return {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
        "visibility": float(getattr(lm, "visibility", 0.0)),
        "presence": float(getattr(lm, "presence", 0.0)),
    }

# MEDIAPIPE POSE LANDMARKER OLUŞTURMA
def build_landmarker(
    model_path: str,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
):
    """
    MediaPipe Pose Landmarker nesnesini oluşturur.

    running_mode=VIDEO seçiyoruz çünkü video üzerinde çalışıyoruz.
    num_poses=1 çünkü bu projede tek kişi varsayıyoruz.
    confidence eşikleri düşük ya da yüksek ayarlanabilir.
    """
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)



# TEK BİR VİDEODAN LANDMARK ÇIKARMA
def process_video(video_path: str, input_root: str, output_root: str, model_path: str, landmarker):
    """
    Tek bir videoyu işler.
    Her frame için pose landmark çıkarır ve JSON dosyasına yazar.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[UYARI] Video açılamadı: {video_path}")
        return

    # FPS bilgisi
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0

    # Toplam frame sayısı, genişlik ve yükseklik bilgisi
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Bu video için landmark çıktısının nereye yazılacağını hesapla
    output_json = get_output_json_path(video_path, input_root, output_root)
    ensure_dir(os.path.dirname(output_json))

    print(f"\nİşleniyor: {video_path}")
    print(f"Çıktı:     {output_json}")

    # JSON içine yazacağımız ana veri yapısı
    data = {
        "video_name": os.path.basename(video_path),
        "video_path": os.path.relpath(video_path, input_root),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "pose_model": os.path.basename(model_path),
        "frames": []
    }

    frame_idx = 0
    detected_count = 0

    # Video sonuna kadar frame frame ilerle
    while True:
        ret, frame_bgr = cap.read()

        # Frame okunamazsa video bitmiş ya da okuma başarısız olmuş demektir
        if not ret or frame_bgr is None:
            break

        # OpenCV frame'i BGR verir
        # MediaPipe ise RGB bekler
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Image nesnesine çeviriyoruz
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Video modunda timestamp vermek gerekir
        timestamp_ms = int((frame_idx / fps) * 1000.0)

        # Pose tespiti burada yapılıyor
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Her frame için başlangıç kayıt yapısı
        frame_record = {
            "frame_index": frame_idx,
            "timestamp_ms": timestamp_ms,
            "timestamp_sec": round(frame_idx / fps, 6),
            "detected": False,
            "num_poses": 0,
            "image_landmarks": [],
            "world_landmarks": []
        }

        # Pose bulunduysa landmarkları kaydet
        if result.pose_landmarks:
            frame_record["detected"] = True
            frame_record["num_poses"] = len(result.pose_landmarks)

            # Tek kişi beklediğimiz için ilk kişiyi alıyoruz
            pose_landmarks = result.pose_landmarks[0]
            frame_record["image_landmarks"] = [landmark_to_dict(lm) for lm in pose_landmarks]

            # 3B world landmarks varsa onları da kaydediyoruz
            if result.pose_world_landmarks:
                pose_world_landmarks = result.pose_world_landmarks[0]
                frame_record["world_landmarks"] = [
                    world_landmark_to_dict(lm) for lm in pose_world_landmarks
                ]

            detected_count += 1

        # Bu frame'in kaydını ana listeye ekle
        data["frames"].append(frame_record)
        frame_idx += 1

        # Her 100 framede bir ilerleme göster
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total_frames} frame işlendi...")

    cap.release()

    # Özet bilgi
    data["processed_frames"] = frame_idx
    data["detected_frames"] = detected_count
    data["detection_rate"] = round(detected_count / max(frame_idx, 1), 6)

    # JSON olarak kaydet
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Tamamlandı. Tespit oranı: {detected_count}/{frame_idx} = {data['detection_rate']:.3f}")


# KLASÖRDEKİ TÜM VİDEOLARI İŞLEME
def process_folder(input_root: str, output_root: str, model_path: str):
    """
    Girdi klasöründeki tüm videoları bulur ve sırayla işler.
    """
    video_files = get_video_files(input_root)

    if not video_files:
        print(f"[HATA] '{input_root}' içinde .mp4 bulunamadı.")
        return

    print(f"Toplam {len(video_files)} video bulundu.")

    # Her video için ayrı landmarker 
    for video_path in video_files:
        with build_landmarker(model_path=model_path) as landmarker:
            process_video(video_path, input_root, output_root, model_path, landmarker)



# ANA ÇALIŞMA BLOĞU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Videolardan MediaPipe Pose landmark çıkarır."
    )

    # Videoların bulunduğu klasör
    parser.add_argument(
        "--input_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\videos\squat",
        help="Videoların bulunduğu kök klasör"
    )

    # Landmark çıktılarının kaydedileceği klasör
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\landmarks\squat",
        help="Landmark JSON çıktılarının yazılacağı klasör"
    )

    # MediaPipe .task model dosyası zorunlu
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task",
        required=False,
        help="Pose landmarker .task model dosyasının tam yolu"
    )

    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    model_path = args.model_path

    ensure_dir(input_root)
    ensure_dir(output_root)

    process_folder(input_root, output_root, model_path)

