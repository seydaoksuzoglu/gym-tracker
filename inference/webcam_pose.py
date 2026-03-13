import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

# Modelin bulduğu landmark'ları kameradan gelen görüntünün üstüne çizip, ekranda "iskeletli" haliyle göstermek
def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result):
    annotated_image = np.copy(rgb_image)
    # detection_result: PoseLandmarker'ın ürettiği sonuç
    # pose_landmarks: tespit edilen iskelet noktaları listesi
    pose_landmarks_list = detection_result.pose_landmarks

    landmark_style = drawing_styles.get_default_pose_landmarks_style()
    connection_style = drawing_utils.DrawingSpec(thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )
    return annotated_image

def main():
    model_path = r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.70,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı.")
    
    last_ts = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("Kameradan görüntü alınamadı.")
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
        ts = int(time.time() * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, ts)

        annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("Mediapipe Pose Landmarker", annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()