import json
from pathlib import Path
from typing import Any, Dict

from config import settings

# Path to the on-disk cache file
CACHE_PATH = Path(settings.CACHE_FILE)

# In-memory cache dict
CACHE: Dict[str, Any] = {}

# cache.py
def load_cache() -> Dict[str, Any]:
    global CACHE
    try:
        data = json.load(open(CACHE_PATH, "r", encoding="utf-8"))
        CACHE.clear()
        CACHE.update(data)
    except Exception:
        CACHE.clear()
    return CACHE



def save_cache() -> None:
    """
    Atomically write the in-memory CACHE dict back to disk.
    """
    # Ensure parent directory exists
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            try:
                json.dump(CACHE, f, ensure_ascii=False, indent=2)
            except UnicodeEncodeError:
            # rewind and retry with ensure_ascii=True
                f.seek(0)
                f.truncate()
                json.dump(CACHE, f, ensure_ascii=True, indent=2)
        tmp.replace(CACHE_PATH)
    except OSError as e:
        print(f"Error saving cache to {CACHE_PATH}: {e}")

# Auto-load cache on import so CACHE is ready immediately
load_cache()
