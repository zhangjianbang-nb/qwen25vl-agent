"""Archive and retention.

A 24/7 monitor accumulates two things that grow without bound: episodes (small, worth keeping)
and raw observations plus saved frames (large, only useful for a few days).  This module moves
aged rows out of SQLite into per-day JSONL files, then prunes the database, and finally trims
the archive itself to a size budget so a long run cannot fill the disk.

JSONL is the archive format on purpose: append-only, one record per line, greppable with plain
shell tools, and readable without this package.  If the agent is gone in a year, the archive
still opens with ``grep``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from .store import MemoryStore

__all__ = ["Archiver"]


class Archiver:
    def __init__(self, store: MemoryStore, archive_dir: str, *,
                 episode_days: int = 30, observation_days: int = 7,
                 max_mb: int = 2048, frames_dir: Optional[str] = None,
                 frame_days: int = 3) -> None:
        self.store = store
        self.archive_dir = os.path.expanduser(archive_dir)
        self.episode_days = int(episode_days)
        self.observation_days = int(observation_days)
        self.max_mb = int(max_mb)
        self.frames_dir = os.path.expanduser(frames_dir) if frames_dir else None
        self.frame_days = int(frame_days)
        os.makedirs(self.archive_dir, exist_ok=True)

    # -- helpers --------------------------------------------------------------------
    @staticmethod
    def _day(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    def _append_jsonl(self, path: str, rows: List[dict]) -> int:
        if not rows:
            return 0
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return len(rows)

    # -- main step -------------------------------------------------------------------
    def run(self, now: Optional[float] = None) -> Dict[str, object]:
        """Archive aged rows, prune the database, trim frames and the archive.  Idempotent."""
        now = now if now is not None else time.time()
        report: Dict[str, object] = {"episodes": 0, "observations": 0, "frames": 0, "files_removed": 0}

        report["episodes"] = self._archive_table(
            table="episodes", cutoff=now - self.episode_days * 86400, prefix="episodes")
        report["observations"] = self._archive_table(
            table="observations", cutoff=now - self.observation_days * 86400, prefix="observations")

        if self.frames_dir and os.path.isdir(self.frames_dir):
            report["frames"] = self._prune_frames(now)
        report["files_removed"] = self._enforce_budget()
        report["archive_bytes"] = self._archive_size()
        return report

    def _archive_table(self, table: str, cutoff: float, prefix: str) -> int:
        """Write rows older than ``cutoff`` to per-day JSONL, then delete them."""
        rows = self.store.query(f"SELECT * FROM {table} WHERE ts < ? ORDER BY ts", (cutoff,))
        if not rows:
            return 0

        by_day: Dict[str, List[dict]] = {}
        for row in rows:
            record = {k: row[k] for k in row.keys()}
            by_day.setdefault(self._day(float(row["ts"])), []).append(record)

        written = 0
        for day, records in sorted(by_day.items()):
            written += self._append_jsonl(
                os.path.join(self.archive_dir, f"{prefix}-{day}.jsonl"), records)
        self.store.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
        return written

    def _prune_frames(self, now: float) -> int:
        cutoff = now - self.frame_days * 86400
        removed = 0
        assert self.frames_dir is not None
        for name in os.listdir(self.frames_dir):
            path = os.path.join(self.frames_dir, name)
            try:
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
        return removed

    def _archive_size(self) -> int:
        total = 0
        for name in os.listdir(self.archive_dir):
            path = os.path.join(self.archive_dir, name)
            if os.path.isfile(path):
                total += os.path.getsize(path)
        return total

    def _enforce_budget(self) -> int:
        """Delete the oldest archive files until the total is under ``max_mb``."""
        if self.max_mb <= 0:
            return 0
        limit = self.max_mb * 1024 * 1024
        files = []
        for name in os.listdir(self.archive_dir):
            path = os.path.join(self.archive_dir, name)
            if os.path.isfile(path):
                files.append((os.path.getmtime(path), os.path.getsize(path), path))
        total = sum(size for _, size, _ in files)
        removed = 0
        for _, size, path in sorted(files):        # oldest first
            if total <= limit:
                break
            try:
                os.remove(path)
                total -= size
                removed += 1
            except OSError:
                continue
        return removed
