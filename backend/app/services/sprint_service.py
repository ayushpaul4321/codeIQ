"""
SprintGuard - Sprint Service

Provides sprint data access methods used by the sprint risk and replan endpoints.
Uses supabase-py (HTTPS) for all DB queries since task 5 (SQLAlchemy) is deferred.

Methods:
    get_sprint(sprint_id)        -> dict | None
    get_velocity_trend(sprint_id) -> float
    get_bug_hours_today(sprint_id) -> float
    get_days_remaining(sprint_id)  -> float

Requirements: 8.8, 8.10
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


class SprintService:
    """
    Data-access layer for sprint-related queries.

    Args:
        supabase: A supabase-py Client instance (from get_supabase() dependency).
    """

    def __init__(self, supabase) -> None:
        self._sb = supabase

    # ------------------------------------------------------------------
    # Sprint lookup
    # ------------------------------------------------------------------

    def get_sprint(self, sprint_id: str) -> dict | None:
        """
        Query the sprints table for the given sprint_id.

        Returns:
            Sprint row as a dict, or None if not found.
        """
        try:
            result = (
                self._sb.table("sprints")
                .select("*")
                .eq("id", sprint_id)
                .limit(1)
                .execute()
            )
            rows = result.data
            return rows[0] if rows else None
        except Exception as exc:
            logger.error("get_sprint(%r) failed: %s", sprint_id, exc)
            return None

    # ------------------------------------------------------------------
    # Velocity trend
    # ------------------------------------------------------------------

    def get_velocity_trend(self, sprint: dict) -> float:
        """
        Compute velocity_current - velocity_last_sprint from a sprint row.

        Args:
            sprint: Sprint row dict (from get_sprint).

        Returns:
            Float delta. Returns 0.0 if either velocity field is missing.
        """
        v_current = sprint.get("velocity_current") or 0.0
        v_last    = sprint.get("velocity_last_sprint") or 0.0
        return float(v_current) - float(v_last)

    # ------------------------------------------------------------------
    # Bug hours added today
    # ------------------------------------------------------------------

    def get_bug_hours_today(self, sprint_id: str) -> float:
        """
        Sum effort_hours_estimated from assignments for this sprint created today.

        Uses the assignments table joined with the sprint_id filter and
        a date filter on assigned_at.

        Returns:
            Total estimated hours added today, or 0.0 on error/no data.
        """
        today_str = date.today().isoformat()  # "YYYY-MM-DD"

        try:
            result = (
                self._sb.table("assignments")
                .select("effort_hours_estimated")
                .eq("sprint_id", sprint_id)
                .gte("assigned_at", f"{today_str}T00:00:00")
                .lt("assigned_at",  f"{today_str}T23:59:59")
                .execute()
            )
            rows = result.data or []
            total = sum(
                float(row.get("effort_hours_estimated") or 0.0)
                for row in rows
            )
            return total
        except Exception as exc:
            logger.error("get_bug_hours_today(%r) failed: %s", sprint_id, exc)
            return 0.0

    # ------------------------------------------------------------------
    # Days remaining
    # ------------------------------------------------------------------

    def get_days_remaining(self, sprint: dict) -> float:
        """
        Compute (end_date - today).days for the sprint.

        Args:
            sprint: Sprint row dict (from get_sprint).

        Returns:
            Number of days remaining (can be negative if sprint has ended).
            Returns 7.0 as a safe default if end_date is missing.
        """
        end_date_raw = sprint.get("end_date")
        if not end_date_raw:
            logger.warning("Sprint %r has no end_date — defaulting to 7 days.", sprint.get("id"))
            return 7.0

        try:
            if isinstance(end_date_raw, str):
                # Supabase returns dates as ISO strings "YYYY-MM-DD"
                end_date = date.fromisoformat(end_date_raw[:10])
            elif isinstance(end_date_raw, date):
                end_date = end_date_raw
            else:
                return 7.0

            return float((end_date - date.today()).days)
        except Exception as exc:
            logger.error("get_days_remaining failed for sprint %r: %s", sprint.get("id"), exc)
            return 7.0
