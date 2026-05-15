"""
Katman 1: Pose extraction (MediaPipe).
33 landmark + visibility skoru çıkarır.
visibility < 0.5 -> valid=False
Tamamen kayıp landmark -> Landmark.missing().placeholder
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.analysis.deadlift.landmarks import (
    Landmark,
    LandmarkFrame,
    NUM_LANDMARKS,
)

class PoseCapture:
    """MediaPipe PoseLandmarker wrapper (num_poses=1)"""

    def __init__(
            self,
            model_path: str,
            min_visibility: float = 0.5,
            running_mode: str = "VIDEO",
    ):
        self.min_visibility = min_visibility

        mode = (
            vision.RunningMode.VIDEO
            if running_mode.upper() == "VIDEO"
            else vision.RunningMode.IMAGE
        )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.65,
            min_pose_presence_confidence=0.65,
            min_tracking_confidence=0.70,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mode = mode
    
    def process(self, frame_rgb, ts_ms: int) -> LandmarkFrame:
        """Bir RGB frame icin landmark cikarimi."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
        else:
            result = self._landmarker.detect(mp_image)
        
        if not result.pose_landmarks:
            return LandmarkFrame(
                landmarks=[Landmark.missing() for _ in range(NUM_LANDMARKS)],
                ts_ms=ts_ms,
            )
        pose = result.pose_landmarks[0]
        landmarks = []
        for lm in pose:
            vis = float(getattr(lm, "visibility", 1.0))
            valid = vis >= self.min_visibility
            landmarks.append(Landmark(
                x=lm.x, y=lm.y, visibility=vis, valid=valid,
            ))
        return LandmarkFrame(landmarks=landmarks, ts_ms=ts_ms)   # <- for'un DIŞINDA

    def close(self) -> None:

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None