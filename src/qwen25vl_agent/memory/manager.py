"""The memory facade.

``Memory`` wires the five stores onto one SQLite connection and exposes the context builder the
agent uses.  Keeping the five types as separate classes matters because they have genuinely
different lifecycles: working memory dies with the process, episodic/semantic/person persist,
and procedural memory is configuration the user owns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import Config
from .archive import Archiver
from .episodic import EpisodicMemory
from .person import PersonMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory
from .store import MemoryStore
from .working import WorkingMemory

__all__ = ["Memory", "MemoryStore", "WorkingMemory", "EpisodicMemory", "SemanticMemory",
           "ProceduralMemory", "PersonMemory", "Archiver"]


class Memory:
    """All five memories over one SQLite file."""

    def __init__(self, cfg: Config, *, frames_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.store = MemoryStore(cfg.db_path, cfg.archive_dir)
        self.store.connect()

        self.working = WorkingMemory(cfg.memory.working_size, cfg.memory.working_ttl_s)
        self.person = PersonMemory(
            self.store,
            face_threshold=cfg.face.match_threshold,
            voice_threshold=cfg.speaker.threshold,
            identity_mode=cfg.identity.mode,
        )
        self.episodic = EpisodicMemory(self.store)
        self.semantic = SemanticMemory(self.store)
        self.procedural = ProceduralMemory(self.store)
        self.archiver = Archiver(
            self.store,
            cfg.archive_dir,
            episode_days=cfg.memory.episode_retention_days,
            observation_days=cfg.memory.observation_retention_days,
            max_mb=cfg.memory.archive_max_mb,
            frames_dir=frames_dir,
            frame_days=cfg.monitor.frame_retention_days,
        )
        self.procedural.seed_defaults()

    # -- lifecycle ------------------------------------------------------------------
    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def stats(self) -> Dict[str, Any]:
        out = self.store.stats()
        out["working_items"] = len(self.working)
        out["db_path"] = self.store.path
        out["archive_dir"] = self.archive_dir
        return out

    @property
    def archive_dir(self) -> str:
        return self.archiver.archive_dir

    # -- context for the agent -------------------------------------------------------
    def context(self, *, episodes: int = 8, facts: int = 12, working: int = 12) -> Dict[str, Any]:
        """Assemble the memory slice handed to the model on each reasoning step.

        Deliberately small: a 3B model degrades with a long context, and the point of memory is
        to hand it the *relevant* slice, not everything.  Known people, the last few episodes,
        the most recent facts, and the working-memory tail are enough for the agent to describe
        a scene and decide what is worth writing back.
        """
        return {
            "known_people": [
                {
                    "name": p.name,
                    "seen_count": p.seen_count,
                    "last_seen": p.last_seen,
                    "notes": p.notes,
                }
                for p in self.person.all()
            ],
            "recent_episodes": [e.as_dict() for e in self.episodic.recent(limit=episodes)],
            "facts": [f.sentence() for f in self.semantic.get(limit=facts)],
            "working": self.working.snapshot(working),
            "rules": [r.name for r in self.procedural.all(enabled_only=True)],
        }
