"""The five memories: person, working, episodic, semantic, procedural, over one SQLite file."""

from __future__ import annotations

import time

import numpy as np
import pytest

from qwen25vl_agent.memory import Memory, MemoryStore, PersonMemory, WorkingMemory


def test_store_creates_schema_and_reports_counts(cfg):
    with Memory(cfg) as mem:
        stats = mem.store.stats()
        assert stats["schema_version"] == 1
        assert stats["persons"] == 0
        assert stats["episodes"] == 0
        assert stats["facts"] == 0
        assert stats["db_bytes"] > 0


def test_store_rejects_a_bad_blob_dimension(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    with store:
        from qwen25vl_agent.vectors import pack, unpack

        blob = pack(np.zeros(3, dtype=np.float32))
        with pytest.raises(ValueError):
            unpack(blob, dim=4)


# -- person memory ----------------------------------------------------------------------------
def test_person_ensure_is_idempotent(cfg):
    with Memory(cfg) as mem:
        first = mem.person.ensure("小明")
        again = mem.person.ensure("小明")
        assert first.id == again.id
        assert mem.person.get("小明").id == first.id
        assert mem.person.get(first.id).name == "小明"
        assert mem.person.get("不存在") is None
        assert [p.name for p in mem.person.all()] == ["小明"]


def test_person_needs_a_non_empty_name(cfg):
    with Memory(cfg) as mem:
        with pytest.raises(ValueError):
            mem.person.ensure("   ")


def test_face_match_found_and_unknown(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        vec = np.ones(128, dtype=np.float32)
        mem.person.add_face(person, vec)

        hit = mem.person.match_face(vec.copy())
        assert hit.known
        assert hit.name == "小明"
        assert hit.modality == "face"
        assert hit.score == pytest.approx(1.0)

        # An orthogonal vector scores ~1/sqrt(128) ≈ 0.09, well below the SFace threshold.
        orthogonal = np.zeros(128, dtype=np.float32)
        orthogonal[0] = 1.0
        miss = mem.person.match_face(orthogonal)
        assert not miss.known
        assert miss.modality == "face"


def test_match_on_an_empty_gallery_is_unknown(cfg):
    with Memory(cfg) as mem:
        assert not mem.person.match_face(np.ones(128, dtype=np.float32)).known
        assert not mem.person.match_voice(np.ones(192, dtype=np.float32)).known


def test_fuse_prefers_face_then_falls_back_to_voice(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        face_vec = np.ones(128, dtype=np.float32)
        voice_vec = np.ones(192, dtype=np.float32)
        mem.person.add_face(person, face_vec)
        mem.person.add_voice(person, voice_vec)

        both = mem.person.fuse(face_vec=face_vec, voice_vec=voice_vec)
        assert both.known and both.modality == "face"        # prefer_face is the default

        voice_only = mem.person.fuse(voice_vec=voice_vec)
        assert voice_only.known and voice_only.modality == "voice"
        assert voice_only.person_id == person.id


def test_fuse_both_requires_two_modalities_to_agree(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        face_vec = np.ones(128, dtype=np.float32)
        voice_vec = np.ones(192, dtype=np.float32)
        mem.person.add_face(person, face_vec)
        mem.person.add_voice(person, voice_vec)

        strict = PersonMemory(mem.store, identity_mode="both")
        agreed = strict.fuse(face_vec=face_vec, voice_vec=voice_vec)
        assert agreed.known and agreed.modality == "both"
        # One modality alone cannot satisfy "both".
        assert not strict.fuse(face_vec=face_vec).known
        assert not strict.fuse().known


def test_fuse_any_accepts_either_modality(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        voice_vec = np.ones(192, dtype=np.float32)
        mem.person.add_voice(person, voice_vec)

        loose = PersonMemory(mem.store, identity_mode="any")
        hit = loose.fuse(voice_vec=voice_vec)
        assert hit.known and hit.modality == "voice"


def test_touch_and_history(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        mem.person.touch(person.id)
        mem.person.touch(person.id)
        refreshed = mem.person.get(person.id)
        assert refreshed.seen_count == 2
        assert refreshed.first_seen is not None
        assert refreshed.last_seen is not None

        mem.person.note(person.id, "说了一句话", kind="speech")
        history = mem.person.history(person.id)
        assert history[0]["text"] == "说了一句话"
        assert history[0]["kind"] == "speech"


def test_embedding_counts_rename_and_delete(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        mem.person.add_face(person, np.ones(128, dtype=np.float32))
        mem.person.add_face(person, np.full(128, 2.0, dtype=np.float32))
        mem.person.add_voice(person, np.ones(192, dtype=np.float32))
        assert mem.person.embedding_counts(person.id) == {"faces": 2, "voices": 1}

        mem.person.rename(person.id, "小红")
        assert mem.person.get(person.id).name == "小红"

        mem.person.delete(person.id)
        assert mem.person.get(person.id) is None
        # The cascade removes the embeddings too.
        assert mem.person.embedding_counts(person.id) == {"faces": 0, "voices": 0}


def test_person_gallery_cache_is_invalidated_by_a_new_embedding(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        assert not mem.person.match_face(np.ones(128, dtype=np.float32)).known
        mem.person.add_face(person, np.ones(128, dtype=np.float32))
        # The gallery was cached as empty before the insert; it must not stay stale.
        assert mem.person.match_face(np.ones(128, dtype=np.float32)).known


def test_gallery_refresh_crosses_processes(cfg):
    """Enrolment in another process must be visible without restarting the monitor.

    ``enroll`` runs as its own process (it needs the camera, which the monitor may be holding),
    so an in-process cache invalidation alone is not enough: the monitor would keep matching
    against the gallery it loaded at startup and never recognise the newly enrolled person.
    """
    with Memory(cfg) as mem:
        # Cache the gallery as empty first, so staleness would be observable.
        assert not mem.person.match_face(np.ones(128, dtype=np.float32)).known

        with Memory(cfg) as other:            # a second connection, as a second process would
            person = other.person.ensure("建邦")
            other.person.add_face(person, np.ones(128, dtype=np.float32))

        hit = mem.person.match_face(np.ones(128, dtype=np.float32))
        assert hit.known
        assert hit.name == "建邦"


# -- semantic memory --------------------------------------------------------------------------
def test_semantic_upsert_keeps_the_max_confidence(cfg):
    with Memory(cfg) as mem:
        semantic = mem.semantic
        first = semantic.remember("房间", "是", "安静的", confidence=0.4)
        second = semantic.remember("房间", "是", "安静的", confidence=0.9)
        assert first == second
        assert semantic.count() == 1

        fact = semantic.about("房间")[0]
        assert fact.confidence == pytest.approx(0.9)
        assert fact.sentence() == "房间 是 安静的"

        # Re-learning with lower confidence must not lower it.
        semantic.remember("房间", "是", "安静的", confidence=0.1)
        assert semantic.about("房间")[0].confidence == pytest.approx(0.9)


def test_semantic_rejects_empty_parts(cfg):
    with Memory(cfg) as mem:
        with pytest.raises(ValueError):
            mem.semantic.remember("", "p", "o")
        with pytest.raises(ValueError):
            mem.semantic.remember("s", "p", "   ")


def test_semantic_search_and_forget(cfg):
    with Memory(cfg) as mem:
        semantic = mem.semantic
        fact_id = semantic.remember("门", "在", "左边", confidence=0.6)
        assert semantic.search("左边")[0].id == fact_id
        assert semantic.forget(fact_id) is True
        assert semantic.forget(fact_id) is False
        assert semantic.count() == 0


# -- episodic memory --------------------------------------------------------------------------
def test_episodic_record_read_and_search(cfg):
    with Memory(cfg) as mem:
        person = mem.person.ensure("小明")
        episode_id = mem.episodic.record("scene", "小明在书桌前", importance=0.5,
                                         person_ids=[person.id], detail={"x": 1})
        recent = mem.episodic.recent(limit=5)
        assert recent[0].id == episode_id
        assert recent[0].summary == "小明在书桌前"
        assert recent[0].person_ids == [person.id]
        assert mem.episodic.search("书桌")[0].id == episode_id
        assert mem.episodic.for_person(person.id)[0].id == episode_id
        assert mem.episodic.for_person(9999) == []
        assert mem.episodic.count() == 1


def test_episodic_min_importance_filter(cfg):
    with Memory(cfg) as mem:
        mem.episodic.record("scene", "low", importance=0.1)
        mem.episodic.record("alert", "high", importance=0.9)
        got = mem.episodic.recent(limit=10, min_importance=0.5)
        assert [e.summary for e in got] == ["high"]


def test_observe_writes_raw_rows(cfg):
    with Memory(cfg) as mem:
        mem.episodic.observe("frame", payload={"faces": []})
        assert mem.store.stats()["observations"] == 1


# -- working memory ---------------------------------------------------------------------------
def test_working_size_cap_and_latest():
    working = WorkingMemory(size=3, ttl_s=300.0)
    now = time.time()
    for i in range(5):
        working.add("tick", str(i), ts=now + i)
    assert len(working) == 3
    assert [item.text for item in working.recent()] == ["2", "3", "4"]
    assert working.latest("tick").text == "4"
    assert working.recent(kind="missing") == []


def test_working_ttl_prunes_on_read():
    working = WorkingMemory(size=10, ttl_s=1.0)
    working.add("tick", "a", ts=100.0)
    assert len(working.recent(now=100.5)) == 1
    assert len(working.recent(now=200.0)) == 0


def test_working_snapshot_shape():
    working = WorkingMemory(size=4, ttl_s=300.0)
    working.add("scene", "hello", ts=time.time(), importance=0.3)
    snapshot = working.snapshot()
    assert snapshot[0]["kind"] == "scene"
    assert snapshot[0]["text"] == "hello"
    assert snapshot[0]["importance"] == 0.3


# -- the facade -------------------------------------------------------------------------------
def test_memory_facade_stats_and_context(cfg):
    with Memory(cfg) as mem:
        mem.person.ensure("小明")
        mem.semantic.remember("房间", "是", "安静的", confidence=0.7)
        mem.episodic.record("scene", "有人在", importance=0.4)

        stats = mem.stats()
        assert stats["persons"] == 1
        assert stats["facts"] == 1
        assert stats["episodes"] == 1
        assert stats["procedures"] == 4          # seed_defaults ran on first open
        assert stats["db_path"].endswith("memory.db")

        context = mem.context()
        assert context["known_people"][0]["name"] == "小明"
        assert any("安静的" in fact for fact in context["facts"])
        assert len(context["recent_episodes"]) == 1
        assert "unknown_face" in context["rules"]


def test_memory_seed_defaults_is_not_duplicated(cfg):
    with Memory(cfg) as mem:
        assert mem.store.stats()["procedures"] == 4
    with Memory(cfg) as mem:
        assert mem.store.stats()["procedures"] == 4


# -- the agent's context window ----------------------------------------------------------------
def test_context_collapses_repeated_episodes(cfg):
    """Ten identical "1张未知人脸" rows must not crowd out every scene and alert.

    This was the live failure: a flickering unknown-face episode was rewritten on every tick, so
    the eight episodes handed to the model were all the same sentence, and the model then echoed
    that sentence back as its description.
    """
    with Memory(cfg) as mem:
        for _ in range(10):
            mem.episodic.record("face", "画面中出现：1张未知人脸")
        mem.episodic.record("scene", "一个人坐在书桌前看书")
        mem.episodic.record("alert", "陌生人出现在画面中", importance=0.8)

        episodes = mem.context()["recent_episodes"]
        summaries = [e["summary"] for e in episodes]

        assert summaries.count("画面中出现：1张未知人脸") == 1     # collapsed to one
        assert summaries.count("一个人坐在书桌前看书") == 1
        assert "陌生人出现在画面中" in summaries                    # the alert survived


def test_context_caps_each_kind(cfg):
    """A busy stream cannot monopolise the budget: scenes are capped even when all are distinct."""
    with Memory(cfg) as mem:
        for i in range(6):
            mem.episodic.record("scene", f"第{i}个不同的场景")
        mem.episodic.record("face", "画面中出现：小明")

        episodes = mem.context()["recent_episodes"]
        assert sum(1 for e in episodes if e["kind"] == "scene") <= 4
        assert any(e["kind"] == "face" for e in episodes)
        assert len(episodes) <= 8


def test_context_keeps_the_newest_of_each_kind(cfg):
    with Memory(cfg) as mem:
        for i in range(6):
            mem.episodic.record("scene", f"场景{i}", ts=1000.0 + i)

        scenes = [e for e in mem.context()["recent_episodes"] if e["kind"] == "scene"]
        assert [e["summary"] for e in scenes] == ["场景5", "场景4", "场景3", "场景2"]
