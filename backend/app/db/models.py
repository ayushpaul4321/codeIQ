"""
SprintGuard - SQLAlchemy ORM Models

Four tables:
  - bugs        : incoming bug reports with their BERT embedding vector
  - developers  : team members and their sprint workload metadata
  - sprints     : sprint definition and velocity data
  - assignments : triage decisions linking bugs → developers within a sprint

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.sql import func

from app.db.base import Base


class Bug(Base):
    """
    Persisted bug report.

    embedding_vector stores the 384-dim L2-normalised BERT vector so that
    future re-ranking or similarity search can be done directly in SQL.
    """

    __tablename__ = "bugs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    reporter = Column(String(255))
    sprint_id = Column(String(100))
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)  # length = 384


class Developer(Base):
    """
    Team member profile — load and historical fix-time are updated after each assignment.
    """

    __tablename__ = "developers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False)
    current_sprint_load_hours = Column(Float, default=0.0, nullable=False)
    avg_fix_time_hours = Column(Float, default=4.0, nullable=False)


class Sprint(Base):
    """
    Sprint definition.  id is a human-readable key (e.g. "SPRINT-001")
    so that it can be used directly in API paths.
    """

    __tablename__ = "sprints"

    id = Column(String(100), primary_key=True)
    name = Column(String(255))
    start_date = Column(Date)
    end_date = Column(Date)
    velocity_last_sprint = Column(Float)
    velocity_current = Column(Float)


class Assignment(Base):
    """
    Triage decision — links a bug to the developer it was assigned to.

    Foreign key constraints enforce referential integrity:
      assignments.bug_id       → bugs.id
      assignments.developer_id → developers.id
    """

    __tablename__ = "assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bug_id = Column(
        UUID(as_uuid=True),
        ForeignKey("bugs.id", ondelete="CASCADE"),
        nullable=False,
    )
    developer_id = Column(
        UUID(as_uuid=True),
        ForeignKey("developers.id", ondelete="SET NULL"),
        nullable=True,
    )
    sprint_id = Column(String(100))
    confidence = Column(Float)
    effort_hours_estimated = Column(Float)
    assigned_at = Column(DateTime, server_default=func.now(), nullable=False)
