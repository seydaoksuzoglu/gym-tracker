# src/analysis/deadlift/scoring.py
"""
Katman 8 - Z-skor / sinyal buyuklugunden area + confidence + UI seviyesi.

Saf fonksiyonlar. Karar mantigi yok - sadece donusum.
Squat'tan kacis: binary cikti DEGIL, [1-5] + [0-1] iki kanali.
"""
from __future__ import annotations


def area_from_z(z: float) -> int:
    """
    |Z| -> Area (1-5).
      |z| < 1   -> 1 (mukemmel)
      |z| < 2   -> 2 (iyi)
      |z| < 3   -> 3 (uyari)
      |z| < 4   -> 4 (hata)
      |z| >= 4  -> 5 (ciddi)
    """
    a = abs(z)
    if a < 1.0:
        return 1
    if a < 2.0:
        return 2
    if a < 3.0:
        return 3
    if a < 4.0:
        return 4
    return 5


def confidence_from_z(z: float) -> float:
    """|Z| -> [0.0, 1.0] guven. Z>=4 icin tavan."""
    return min(1.0, abs(z) / 4.0)


def ui_level(confidence: float) -> str:
    """UI gosterim esigi.
      confidence < 0.3  -> "silent"
      < 0.6             -> "yellow"
      >= 0.6            -> "red"
    """
    if confidence < 0.3:
        return "silent"
    if confidence < 0.6:
        return "yellow"
    return "red"


def overall_grade(areas: list[int]) -> str:
    """
    Rep icindeki tum hatalarin area'larini birlestirip rep-level etiket uret.
    En kotu area'ya gore karar verilir (max).
      max area <= 1: "perfect"
      <= 2: "good"
      <= 3: "needs_attention"
      <= 4: "form_issue"
      >= 5: "severe"
    """
    if not areas:
        return "perfect"
    worst = max(areas)
    return {
        1: "perfect", 2: "good", 3: "needs_attention",
        4: "form_issue", 5: "severe",
    }[worst]
