"""Archive and retention: JSONL migration, frame pruning, and the size budget."""

from __future__ import annotations

import json
import os
import time

from qwen25vl_agent.memory import Memory


def test_archive_moves_aged_episodes_to_jsonl(cfg):
    now = time.time()
    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        mem.episodic.record("scene", "old episode", ts=now - 40 * 86400)
        mem.episodic.record("scene", "fresh episode", ts=now)

        report = mem.archiver.run(now)
        assert report["episodes"] == 1
        # The aged row left the database; the fresh one stayed.
        assert mem.episodic.count() == 1

        files = [f for f in os.listdir(cfg.archive_dir) if f.startswith("episodes-")]
        assert len(files) == 1
        with open(os.path.join(cfg.archive_dir, files[0]), encoding="utf-8") as fh:
            record = json.loads(fh.read().strip())
        assert record["summary"] == "old episode"


def test_archive_is_idempotent(cfg):
    now = time.time()
    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        mem.episodic.record("scene", "old", ts=now - 40 * 86400)
        assert mem.archiver.run(now)["episodes"] == 1
        # A second pass finds nothing left to move.
        assert mem.archiver.run(now)["episodes"] == 0


def test_archive_moves_aged_observations(cfg):
    now = time.time()
    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        mem.episodic.observe("frame", ts=now - 10 * 86400)     # older than the 7-day retention
        mem.episodic.observe("frame", ts=now)
        report = mem.archiver.run(now)
        assert report["observations"] == 1
        assert mem.store.stats()["observations"] == 1
        assert any(f.startswith("observations-") for f in os.listdir(cfg.archive_dir))


def test_archive_budget_prunes_the_oldest_files(make_cfg):
    cfg = make_cfg(memory={"archive_max_mb": 1})
    os.makedirs(cfg.archive_dir, exist_ok=True)
    big = os.path.join(cfg.archive_dir, "episodes-2000-01-01.jsonl")
    with open(big, "wb") as fh:
        fh.write(b"x" * (2 * 1024 * 1024))       # 2 MB against a 1 MB budget

    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        report = mem.archiver.run()
        assert report["files_removed"] >= 1

    assert not os.path.exists(big)


def test_frames_older_than_retention_are_pruned(cfg):
    frames = cfg.frames_dir
    os.makedirs(frames, exist_ok=True)

    old = os.path.join(frames, "old.jpg")
    with open(old, "wb") as fh:
        fh.write(b"x")
    ten_days_ago = time.time() - 10 * 86400
    os.utime(old, (ten_days_ago, ten_days_ago))       # frame retention is 3 days by default

    fresh = os.path.join(frames, "fresh.jpg")
    with open(fresh, "wb") as fh:
        fh.write(b"x")

    with Memory(cfg, frames_dir=frames) as mem:
        report = mem.archiver.run()
        assert report["frames"] == 1

    assert not os.path.exists(old)
    assert os.path.exists(fresh)


def test_archive_report_shape(cfg):
    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        report = mem.archiver.run()
        assert set(report) >= {"episodes", "observations", "frames", "files_removed",
                              "archive_bytes"}
