import logging
from pathlib import Path

from src.config import REPO_ROOT, settings
from src.storage.database import Base, engine
from src.storage import models  # noqa: F401  — Base.metadata'yi doldurmak icin import

logging.basicConfig(level=settings.LOG_LEVEL)
log = logging.getLogger(__name__)


def init_db() -> None:
    if settings.DATABASE_URL.startswith("sqlite"):
        db_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
        (REPO_ROOT / db_path).parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=engine)
    log.info("DB ready at %s", settings.DATABASE_URL)


if __name__ == "__main__":
    init_db()
