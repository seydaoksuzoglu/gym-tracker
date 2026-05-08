"""
Saf geometri util'leri. Squat ve deadlift modüllerinin ortak yardımcıları.
State yok, side effect yok, print yok.
"""
import math
from typing import Tuple, Mapping

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """İki nokta arası Öklid mesafesi."""
    return math.sqrt((a[0] - b[0]) **2 + (a[1] - b[1]) **2)

def angle(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
) -> float:
    """
    b köşesinden a ve c noktalarına çizilen vektörlerin arasındaki açı (derece).
    Geçersiz girdide 0.0 döner.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_value = dot / (mag_ba * mag_bc)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))

def angle_to_vertical(
        top: Tuple[float, float],
        bottom: Tuple[float, float],
) -> float:
    """
    bottom->top vektörünün dikey eksenle (yukarı yönlü) yaptığı açı (derece).
    Görüntü koordinatlarında dikey = (0, -1).
    """
    v = (top[0] - bottom[0], top[1] - bottom[1])
    vertical = (0.0, -1.0)

    dot = v[0] * vertical[0] + v[1] * vertical[1]
    mag_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

    if mag_v == 0:
        return 0.0

    cos_value = dot / mag_v
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))

def average_visibility(landmarks, side_map: Mapping[str, int]) -> float:
    """side_map'teki landmark indekslerinin ortalama visibility skoru."""
    vis_values = [
        float(getattr(landmarks[idx], "visibility", 1.0))
        for idx in side_map.values()
    ]
    return sum(vis_values) / len(vis_values)

def choose_best_side(
        landmarks,
        left_map: Mapping[str, int],
        right_map: Mapping[str, int],
) -> Tuple[str, Mapping[str, int], float]:
    """
    Hangi taraf (sol/sağ) daha iyi görünüyorsa onu döner.
    LEFT/RIGHT map'leri parametre - squat ve deadlift'te ortak.

    Returns:
        (side_name, side_map, avg_visibility)
    """
    left_vis = average_visibility(landmarks, left_map)
    right_vis = average_visibility(landmarks, right_map)

    if left_vis >= right_vis:
        return "left", left_map, left_vis
    return "right", right_map, right_vis