import argparse
import cv2
import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.vis.skeleton_drawer import draw_landmarks_on_image
from src.sources.webcam import webcam_frames
from src.sources.video import video_frames


def create_landmarker(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.70,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "video"], required=True)
    parser.add_argument("--model", default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task")
    parser.add_argument("--index", type=int, default=0)      # webcam
    parser.add_argument("--path", type=str, default=None)    # video
    parser.add_argument("--output", type=str, default=None)  # çıktı video
    parser.add_argument("--scale", type=float, default=0.6)  # ekran boyutu
    args = parser.parse_args()

    landmarker = create_landmarker(args.model)

    if args.mode == "webcam":
        frame_iter = webcam_frames(args.index)
    else:
        if not args.path:
            raise ValueError("--mode video için --path zorunlu.")
        frame_iter = video_frames(args.path)

    # Video kaydedicisi
    writer = None
    output_path = None
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    try:
        for frame_bgr, ts in frame_iter:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts)

            annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # VideoWriter ilk frame ile oluştur
            if writer is None and output_path:
                h, w = annotated_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
            
            # Video dosyasına yaz
            if writer:
                writer.write(annotated_bgr)

            # Ekranda göster (ölçeklendirilmiş)
            display_frame = cv2.resize(annotated_bgr, 
                                      (int(annotated_bgr.shape[1] * args.scale),
                                       int(annotated_bgr.shape[0] * args.scale)))
            cv2.imshow("MediaPipe Pose Landmarker", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        landmarker.close()


if __name__ == "__main__":
    main()