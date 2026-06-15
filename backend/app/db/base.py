"""
SprintGuard - SQLAlchemy Declarative Base

All ORM model classes must inherit from `Base` defined here.
Alembic's env.py also imports Base.metadata to autogenerate migrations.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
