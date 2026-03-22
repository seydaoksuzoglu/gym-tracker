import numpy as np
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles


def draw_landmarks_on_image_mediapipe(rgb_image: np.ndarray, detection_result):
    """
    MediaPipe PoseLandmarker sonucundaki landmark'ları görüntünün üstüne çizer.
    """
    annotated_image = np.copy(rgb_image)
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


def draw_landmarks_on_image_yolo(rgb_image: np.ndarray, yolo_result):
    """
    Ultralytics YOLO pose sonucunu görüntü üzerine çizer.
    YOLO'nun kendi plot() metodunu kullanır.
    plot() BGR döndürdüğü için tekrar RGB'ye çeviriyoruz.
    """
    if yolo_result is None:
        return np.copy(rgb_image)

    plotted_bgr = yolo_result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    return plotted_rgb