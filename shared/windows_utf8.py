"""Force UTF-8 mode on Windows before reading ir-datasets files."""
from __future__ import annotations

import os
import sys


def ensure_utf8_mode() -> None:
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        import subprocess

        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))
