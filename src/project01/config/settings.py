from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.getenv("PROJECT01_DATA_DIR", ROOT_DIR / "data" / "raw"))
if not DATA_DIR.is_absolute():
    DATA_DIR = ROOT_DIR / DATA_DIR
ARTIFACT_DIR = Path(os.getenv("PROJECT01_ARTIFACT_DIR", ROOT_DIR / "models"))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = ROOT_DIR / ARTIFACT_DIR
CONFIG_PATH = ROOT_DIR / "config_datasets.yaml"
