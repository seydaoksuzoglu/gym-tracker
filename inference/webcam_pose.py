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
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.vis.skeleton_drawer import (
    draw_landmarks_on_image_mediapipe,
    draw_landmarks_on_image_yolo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT

def main():
    parser = argparse.ArgumentParser(description="Ham pose çizimi + FPS ölçümü")
    parser.add_argument("--backend", choices=["mediapipe", "yolo"], default="mediapipe")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--yolo-model", default="yolo26m-pose.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

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
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_model)
        print(f"[INFO] YOLO yüklendi: {args.yolo_model}")

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {args.index} açılamadı.")

    last_ts = 0
    frame_count = 0
    total_infer_ms = 0.0
    total_draw_ms = 0.0
    fps_display = 0.0
    fps_timer = time.perf_counter()
    fps_frame_count = 0

    title = f"Pose — {args.backend.upper()}"
    print("[INFO] Başlatıldı. Çıkmak için 'q' tuşuna bas.\n")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[WARN] Kameradan görüntü alınamadı.")
                break

            frame_count += 1
            fps_frame_count += 1

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

            draw_t0 = time.perf_counter()
            if args.backend == "mediapipe":
                annotated_rgb = draw_landmarks_on_image_mediapipe(frame_rgb, result)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            else:
                annotated_rgb = draw_landmarks_on_image_yolo(frame_bgr, result[0])
                annotated_bgr = annotated_rgb  # yolo zaten BGR'ye yakın iç dönüşüm yapıyor
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0
            total_draw_ms += draw_ms

            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 0.5:
                fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.perf_counter()

            overlay = (
                f"FPS: {fps_display:.1f}  "
                f"Infer: {infer_ms:.1f}ms  "
                f"Draw: {draw_ms:.1f}ms  "
                f"[{args.backend.upper()}]"
            )
            cv2.putText(
                annotated_bgr, overlay, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )

            if args.scale != 1.0:
                h, w = annotated_bgr.shape[:2]
                annotated_bgr = cv2.resize(
                    annotated_bgr,
                    (int(w * args.scale), int(h * args.scale)),
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
            print(f"Ort. çizim          : {total_draw_ms / frame_count:.2f} ms")
            print(f"Ort. toplam         : {(total_infer_ms + total_draw_ms) / frame_count:.2f} ms")
            print(f"Son FPS             : {fps_display:.1f}")


if __name__ == "__main__":
    main()