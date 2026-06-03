from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "sqlite:///data/gymtracker.db"
    MEDIAPIPE_MODEL_PATH: str = "models/pose_landmarker_full.task"
    YOLO_MODEL_PATH: str = "yolo26m-pose.pt"
    LOG_LEVEL: str = "INFO"

    @property
    def mediapipe_model_abs(self) -> Path:
        return (REPO_ROOT / self.MEDIAPIPE_MODEL_PATH).resolve()

    @property
    def yolo_model_abs(self) -> Path:
        return (REPO_ROOT / self.YOLO_MODEL_PATH).resolve()


settings = Settings()
