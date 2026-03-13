import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result):
    """
    PoseLandmarker sonucundaki landmark'ları görüntünün üstüne çizer.
    rgb_image: RGB numpy image
    detection_result: landmarker.detect_for_video(...) çıktısı
    """

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