-- SprintGuard — Initial Schema
-- Run this once in your Supabase SQL Editor:
-- Project → SQL Editor → New query → paste and run

-- Enable the uuid-ossp extension so gen_random_uuid() works
create extension if not exists "pgcrypto";

-- ----------------------------------------------------------------
-- bugs
-- ----------------------------------------------------------------
create table if not exists bugs (
    id               uuid primary key default gen_random_uuid(),
    title            text not null,
    description      text not null,
    reporter         varchar(255),
    sprint_id        varchar(100),
    created_at       timestamptz not null default now(),
    embedding_vector float8[] not null          -- 384-element BERT vector
);

-- ----------------------------------------------------------------
-- developers
-- ----------------------------------------------------------------
create table if not exists developers (
    id                         uuid primary key default gen_random_uuid(),
    username                   varchar(255) unique not null,
    current_sprint_load_hours  float8 not null default 0.0,
    avg_fix_time_hours         float8 not null default 4.0
);

-- ----------------------------------------------------------------
-- sprints
-- ----------------------------------------------------------------
create table if not exists sprints (
    id                   varchar(100) primary key,
    name                 varchar(255),
    start_date           date,
    end_date             date,
    velocity_last_sprint float8,
    velocity_current     float8
);

-- ----------------------------------------------------------------
-- assignments
-- ----------------------------------------------------------------
create table if not exists assignments (
    id                     uuid primary key default gen_random_uuid(),
    bug_id                 uuid not null references bugs(id) on delete cascade,
    developer_id           uuid references developers(id) on delete set null,
    sprint_id              varchar(100),
    confidence             float8,
    effort_hours_estimated float8,
    assigned_at            timestamptz not null default now()
);
