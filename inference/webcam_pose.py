"""
webcam_pose.py  –  Ham pose çizimi + FPS / latency ölçümü
Squat analizi YOK, sadece skeleton overlay ve performans metrikleri.

Kullanım:
  python inference/webcam_pose.py --backend mediapipe
  python inference/webcam_pose.py --backend yolo
  python inference/webcam_pose.py --backend mediapipe --scale 1.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from ultralytics import YOLO


# ──────────────────────────────────────────────
# MediaPipe çizim yardımcısı
# ──────────────────────────────────────────────
def draw_mediapipe(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    annotated = np.copy(rgb_image)
    landmark_style = drawing_styles.get_default_pose_landmarks_style()
    connection_style = drawing_utils.DrawingSpec(thickness=2)
    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )
    return annotated


# ──────────────────────────────────────────────
# YOLO çizim yardımcısı
# ──────────────────────────────────────────────
YOLO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def draw_yolo(bgr_image: np.ndarray, yolo_result) -> np.ndarray:
    annotated = bgr_image.copy()
    if yolo_result.keypoints is None:
        return annotated
    kps = yolo_result.keypoints.xy.cpu().numpy()   # (N, 17, 2)
    for person_kps in kps:
        # Eklem noktaları
        for x, y in person_kps:
            if x > 0 and y > 0:
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
        # Bağlantılar
        for a, b in YOLO_SKELETON:
            xa, ya = person_kps[a]
            xb, yb = person_kps[b]
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(annotated, (int(xa), int(ya)), (int(xb), int(yb)),
                         (255, 100, 0), 2)
    return annotated


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ham pose çizimi + FPS ölçümü")
    parser.add_argument("--backend", choices=["mediapipe", "yolo"], default="mediapipe",
                        help="Kullanılacak pose backend'i")
    parser.add_argument("--model",
                        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task",
                        help="MediaPipe .task dosyası (sadece mediapipe backend'inde kullanılır)")
    parser.add_argument("--yolo-model", default="yolo26m-pose.pt",
                        help="YOLO model adı/yolu")
    parser.add_argument("--index", type=int, default=0, help="Kamera index'i")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Görüntü gösterim ölçeği (1.0 = orijinal)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="YOLO için cihaz (cuda:0 veya cpu)")
    args = parser.parse_args()

    # ── Model yükle ──
    landmarker = None
    yolo_model = None

    if args.backend == "mediapipe":
        base_options = python.BaseOptions(model_asset_path=args.model)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.65,
            min_pose_presence_confidence=0.65,
            min_tracking_confidence=0.70,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print(f"[INFO] MediaPipe yüklendi: {args.model}")
    else:
        yolo_model = YOLO(args.yolo_model)
        print(f"[INFO] YOLO yüklendi: {args.yolo_model}")

    # ── Kamera ──
    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {args.index} açılamadı.")

    # ── Metrikler ──
    last_ts = 0
    frame_count      = 0
    total_infer_ms   = 0.0
    total_draw_ms    = 0.0
    fps_display      = 0.0
    fps_timer        = time.perf_counter()
    fps_frame_count  = 0

    title = f"Pose — {args.backend.upper()}"
    print(f"[INFO] Başlatıldı. Çıkmak için 'q' tuşuna bas.\n")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Kameradan görüntü alınamadı.")
                break

            frame_count += 1
            fps_frame_count += 1

            # ── Inference ──
            infer_t0 = time.perf_counter()

            if args.backend == "mediapipe":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts = int(time.time() * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, ts)
            else:
                result = yolo_model(frame_bgr, device=args.device, verbose=False)

            infer_ms = (time.perf_counter() - infer_t0) * 1000.0
            total_infer_ms += infer_ms

            # ── Çizim ──
            draw_t0 = time.perf_counter()

            if args.backend == "mediapipe":
                annotated_rgb = draw_mediapipe(frame_rgb, result)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            else:
                annotated_bgr = draw_yolo(frame_bgr, result[0])

            draw_ms = (time.perf_counter() - draw_t0) * 1000.0
            total_draw_ms += draw_ms

            # ── FPS güncelle (her 0.5 sn) ──
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 0.5:
                fps_display     = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer       = time.perf_counter()

            # ── Overlay ──
            overlay = (
                f"FPS: {fps_display:.1f}  "
                f"Infer: {infer_ms:.1f}ms  "
                f"Draw: {draw_ms:.1f}ms  "
                f"[{args.backend.upper()}]"
            )
            cv2.putText(
                annotated_bgr, overlay,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

            # ── Göster ──
            if args.scale != 1.0:
                h, w = annotated_bgr.shape[:2]
                annotated_bgr = cv2.resize(
                    annotated_bgr,
                    (int(w * args.scale), int(h * args.scale))
                )

            cv2.imshow(title, annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker:
            landmarker.close()

        if frame_count > 0:
            print(f"\n===== WEBCAM POSE ÖZET [{args.backend.upper()}] =====")
            print(f"Toplam frame        : {frame_count}")
            print(f"Ort. inference      : {total_infer_ms / frame_count:.2f} ms")
            print(f"Ort. çizim          : {total_draw_ms  / frame_count:.2f} ms")
            print(f"Ort. toplam         : {(total_infer_ms + total_draw_ms) / frame_count:.2f} ms")
            print(f"Son FPS             : {fps_display:.1f}")


if __name__ == "__main__":
    main()