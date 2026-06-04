from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, JSON, String, Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.database import Base


class Exercise(Base):
    __tablename__ = "exercises"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(32), unique=True)
    display_name: Mapped[str] = mapped_column(String(64))


class Routine(Base):
    __tablename__ = "routines"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    items: Mapped[list["RoutineItem"]] = relationship(
        back_populates="routine", cascade="all, delete-orphan"
    )


class RoutineItem(Base):
    __tablename__ = "routine_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    routine_id: Mapped[int] = mapped_column(ForeignKey("routines.id"))
    exercise_key: Mapped[str] = mapped_column(String(32))
    target_sets: Mapped[int] = mapped_column(Integer)
    target_reps: Mapped[int] = mapped_column(Integer)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    routine: Mapped["Routine"] = relationship(back_populates="items")


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(16))  # "webcam" | "video"
    video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    routine_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("routines.id"), nullable=True
    )
    routine: Mapped[Optional["Routine"]] = relationship()

    sets: Mapped[list["SetRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SetRow(Base):
    __tablename__ = "sets"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    exercise_key: Mapped[str] = mapped_column(String(32))
    set_index: Mapped[int] = mapped_column(Integer)
    target_reps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completed_reps: Mapped[int] = mapped_column(Integer, default=0)
    backend: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    session: Mapped["Session"] = relationship(back_populates="sets")
    reps: Mapped[list["Rep"]] = relationship(
        back_populates="set", cascade="all, delete-orphan"
    )


class Rep(Base):
    __tablename__ = "reps"
    id: Mapped[int] = mapped_column(primary_key=True)
    set_id: Mapped[int] = mapped_column(ForeignKey("sets.id"))
    rep_index: Mapped[int] = mapped_column(Integer)
    overall_grade: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    phase_durations_ms: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    video_ts_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    set: Mapped["SetRow"] = relationship(back_populates="reps")
    errors: Mapped[list["RepError"]] = relationship(
        back_populates="rep", cascade="all, delete-orphan"
    )


class RepError(Base):
    __tablename__ = "rep_errors"
    id: Mapped[int] = mapped_column(primary_key=True)
    rep_id: Mapped[int] = mapped_column(ForeignKey("reps.id"))
    error_type: Mapped[str] = mapped_column(String(64))
    area: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    evidence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    rep: Mapped["Rep"] = relationship(back_populates="errors")
