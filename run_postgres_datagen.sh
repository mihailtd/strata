#!/usr/bin/env bash
set -e
cd /home/mihai/gnn-experiment
EPUB="data/postgresql/AI-Ready PostgreSQL 18_ Buildin - Vibhor Kumar.epub"

uv run scripts/run_datagen.py \
  --epub "AI-Ready PostgreSQL 18_ Buildin - Vibhor Kumar=${EPUB}" \
  --domain-description "PostgreSQL 18 and AI-assisted database design" \
  --source-label book \
  --out data/postgresql \
  --no-unload
