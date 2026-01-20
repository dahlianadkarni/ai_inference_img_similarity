"""Incremental scan cache for photo metadata and hashes."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .image_utils import ImageMetadata

logger = logging.getLogger(__name__)


class ScanCache:
    """Disk-backed cache keyed by file path + (size, mtime)."""

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {"version": 1, "entries": {}}

    def load(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "r") as f:
                self._data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {self.cache_file}: {e}")
            self._data = {"version": 1, "entries": {}}

    def save(self) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {self.cache_file}: {e}")

    def _key(self, file_path: Path) -> str:
        return str(file_path)

    def get_entry(self, file_path: Path) -> Optional[dict]:
        return self._data.get("entries", {}).get(self._key(file_path))

    def is_fresh(self, file_path: Path) -> bool:
        entry = self.get_entry(file_path)
        if not entry:
            return False
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            return False
        return entry.get("file_size") == stat.st_size and entry.get("mtime") == stat.st_mtime

    def to_metadata(self, file_path: Path) -> Optional[ImageMetadata]:
        entry = self.get_entry(file_path)
        if not entry:
            return None

        # Only keep fields we know are stable/needed. created_at/modified_at are optional.
        return ImageMetadata(
            file_path=str(file_path),
            file_size=int(entry.get("file_size", 0)),
            width=int(entry.get("width", 0)),
            height=int(entry.get("height", 0)),
            format=str(entry.get("format", "UNKNOWN")),
            md5_hash=entry.get("md5_hash"),
            perceptual_hash=entry.get("perceptual_hash"),
            camera_make=entry.get("camera_make"),
            camera_model=entry.get("camera_model"),
            is_screenshot=bool(entry.get("is_screenshot", False)),
            orientation=int(entry.get("orientation", 1)),
        )

    def update_from_metadata(self, file_path: Path, metadata: ImageMetadata) -> None:
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            return

        entry = asdict(metadata)
        # Normalize datetimes (we don't rely on them for freshness)
        entry["created_at"] = None
        entry["modified_at"] = None
        entry["mtime"] = stat.st_mtime
        entry["file_size"] = stat.st_size

        self._data.setdefault("entries", {})[self._key(file_path)] = entry

    def update_md5(self, file_path: Path, md5_hash: str) -> None:
        entry = self.get_entry(file_path)
        if not entry:
            return
        entry["md5_hash"] = md5_hash

    def size_bytes(self) -> int:
        try:
            return self.cache_file.stat().st_size
        except FileNotFoundError:
            return 0

    def entry_count(self) -> int:
        return len(self._data.get("entries", {}))
