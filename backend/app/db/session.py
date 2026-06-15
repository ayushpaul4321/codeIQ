"""
SprintGuard - Database Session / Client Factory

Two connection strategies are provided:

1. `get_supabase_client()` — returns a supabase-py Client that connects over
   HTTPS (port 443). This is the primary strategy used at runtime because it
   works through firewalls that block raw PostgreSQL ports (5432 / 6543).

2. `engine` / `SessionLocal` — standard SQLAlchemy engine kept for Alembic
   migration commands and environments where direct TCP access is available.

Environment variables (set in .env or Docker):
  SUPABASE_URL  — e.g. https://haqjdstdlyhrapvhdynf.supabase.co
  SUPABASE_KEY  — anon or service-role key from Supabase → Settings → API
  DATABASE_URL  — standard PostgreSQL DSN (used by Alembic only)
"""

from __future__ import annotations

import os
from functools import lru_cache

# ---------------------------------------------------------------------------
# Supabase client (HTTPS — works through firewalls)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_supabase_client():
    """
    Return a singleton supabase-py Client.

    Cached so the same client object is reused across requests (the client
    maintains an internal httpx connection pool).
    """
    from supabase import create_client, Client  # type: ignore

    url: str = os.environ.get("SUPABASE_URL", "")
    key: str = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env. "
            "Find them in your Supabase project → Settings → API."
        )

    return create_client(url, key)


# ---------------------------------------------------------------------------
# SQLAlchemy engine (direct TCP — used by Alembic; may be blocked on some
# networks; not used by the FastAPI app at runtime)
# ---------------------------------------------------------------------------

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/sprintguard",
    )

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 5},
    )

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
except Exception:
    # SQLAlchemy not needed when running purely via the Supabase client
    engine = None       # type: ignore[assignment]
    SessionLocal = None # type: ignore[assignment]
