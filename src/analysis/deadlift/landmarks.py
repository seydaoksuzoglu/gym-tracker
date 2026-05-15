"""
Deadlift modülü ortak veri tipleri ve MediaPipe landmark indeksleri.
Tüm pipeline katmanları (capture, filter, calibration, features) bu tipleri kullanır.
"""
from dataclasses import dataclass
from typing import List, Mapping

# MediaPipe Pose 33-landmark indeksleri (deadlift için ilgili olanlar)
LEFT: Mapping[str, int] = {
    "ear": 7,
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
    "heel": 29,
    "foot_index": 31,
}

RIGHT: Mapping[str, int] = {
    "ear": 8,
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "heel": 30,
    "foot_index": 32,
}

NUM_LANDMARKS = 33

@dataclass
class Landmark:
    x: float
    y: float
    visibility: float
    valid: bool

    @classmethod
    def missing(cls) -> "Landmark":
        """Tamamen tespit edilemeyen landmark için placeholder."""
        return cls(x=-1.0, y=-1.0, visibility=0.0, valid=False)
    
    def as_xy(self):
        return (self.x, self.y)
    
@dataclass
class LandmarkFrame:
    landmarks: List[Landmark] # Uzunluk = NUM_LANDMARKS
    ts_ms: int

    def get(self, idx: int) -> Landmark:
        return self.landmarks[idx]