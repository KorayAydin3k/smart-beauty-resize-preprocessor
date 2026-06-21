from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STRING = str(PROJECT_ROOT)

if PROJECT_ROOT_STRING not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STRING)
