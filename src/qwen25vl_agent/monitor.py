"""The 24/7 monitoring loop.

The loop is split across two threads on purpose:

* an **audio thread** continuously captures, runs VAD, transcribes and extracts voiceprints,
  and hands finished utterances to the main thread through a queue.  Speech work is bursty and
  CPU-heavy (a 936 MB ASR model on CPU), so it must not stall the tick loop.
* the **main thread** owns everything else: frame sampling, face recognition, identity fusion,
  the agent call, memory writes, rule evaluation and archiving.  It also owns the SQLite
  connection -- all database writes happen here, because a sqlite3 connection is not safe to
  share across threads and this loop's correctness matters more than shaving a lock.

The tick loop never raises out: a bad frame, a failed model call or a malformed reply degrades
one tick and the next one proceeds.  A 24/7 agent that dies on the first hiccup is not a 24/7
agent.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, Decision, Observation
from .config import Config
from .imaging import rgb_to_jpeg
from .llm import LLMError
from .memory import Memory, TickContext
from .perception.audio import AudioCapture, AudioUnavailable
from .perception.face import FaceEngine, FaceUnavailable
from .perception.vad import EnergyVAD
from .perception.video import CameraSource

__all__ = ["Monitor", "MonitorEvent"]


@dataclass
class MonitorEvent:
    """Something worth telling the operator about."""

    ts: float
    kind: str                  # 'scene' | 'alert' | 'speech' | 'person' | 'error' | 'tick'
    message: str
    importance: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "time": datetime.fromtimestamp(self.ts).strftime("%H:%M:%S"),
            "kind": self.kind,
            "message": self.message,
            "importance": round(self.importance, 3),
            **self.data,
        }


@dataclass
class _Utterance:
    """What the audio thread produces: text, language, and a raw voiceprint."""

    ts: float
    text: str
    language: str
    emotion: str
    duration_s: float
    voice_embedding: Any = None


class Monitor:
    def __init__(self, cfg: Config, *, memory: Optional[Memory] = None,
                 camera: Optional[CameraSource] = None,
                 face: Optional[FaceEngine] = None,
                 asr: Optional[Any] = None, speaker: Optional[Any] = None,
                 agent: Optional[Agent] = None) -> None:
        self.cfg = cfg
        self.memory = memory or Memory(cfg, frames_dir=cfg.frames_dir)
        self.camera = camera or CameraSource(device=cfg.camera.device, width=cfg.camera.width)
        self.face = face or FaceEngine(
            cfg.detector_model_path, cfg.recognizer_model_path,
            detect_score=cfg.face.detect_score, min_face_px=cfg.face.min_face_px,
        )
        self.asr = asr
        self.speaker = speaker
        self.agent = agent or Agent(cfg)

        self._face_ready = False
        self._stop = threading.Event()
        self._speech_q: "queue.Queue[_Utterance]" = queue.Queue(maxsize=64)
        self._audio_thread: Optional[threading.Thread] = None
        self._recent_voices: List[Dict[str, Any]] = []
        self._last_speech_ts: Optional[float] = None
        self._last_describe = 0.0
        self._last_archive = 0.0
        self._started_ts = 0.0
        self._pending_event = False
        self._frame_counter = 0
        self._stats: Dict[str, Any] = {
            "ticks": 0, "frames": 0, "describes": 0, "describe_errors": 0,
            "faces": 0, "utterances": 0, "alerts": 0, "episodes": 0,
        }
        self._listeners: List[Callable[[MonitorEvent], None]] = []

    # -- setup ----------------------------------------------------------------------
    def on_event(self, fn: Callable[[MonitorEvent], None]) -> None:
        self._listeners.append(fn)

    def _emit(self, event: MonitorEvent) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:
                continue

    def _setup(self) -> None:
        os.makedirs(self.cfg.frames_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.alert_log) or ".", exist_ok=True)
        self._started_ts = time.time()

        # Face models are optional: without them the agent still describes scenes.
        if self.face.models_present(self.cfg.detector_model_path, self.cfg.recognizer_model_path):
            self._face_ready = self.face.available()
        else:
            self._face_ready = False

        self.camera.open()

        # Warm the galleries in this thread so the audio thread never touches SQLite.
        self.memory.person.all()

        if self.cfg.audio.enabled:
            self._start_audio()

    def _start_audio(self) -> None:
        if self.asr is None and self.speaker is None:
            return
        self._audio_thread = threading.Thread(target=self._audio_loop, name="audio", daemon=True)
        self._audio_thread.start()

    # -- audio thread ----------------------------------------------------------------
    def _audio_loop(self) -> None:
        """Capture -> VAD -> ASR -> voiceprint, publishing utterances onto a queue."""
        capture = AudioCapture(
            device=self.cfg.audio.device, rate=self.cfg.audio.rate,
            channels=self.cfg.audio.channels, chunk_ms=self.cfg.audio.chunk_ms,
        )
        vad = EnergyVAD(
            start_db=self.cfg.audio.vad.start_db,
            hangover_ms=self.cfg.audio.vad.hangover_ms,
            min_speech_ms=self.cfg.audio.vad.min_speech_ms,
            max_segment_s=self.cfg.audio.vad.max_segment_s,
            rate=self.cfg.audio.rate,
        )
        backoff = 1.0
        while not self._stop.is_set():
            try:
                capture.start()
            except AudioUnavailable as exc:
                self._emit(MonitorEvent(time.time(), "error", f"audio unavailable: {exc}"))
                return

            try:
                while not self._stop.is_set():
                    chunk = capture.read_float()
                    if chunk is None:            # arecord exited; re-open below
                        break
                    segment = vad.push(chunk, time.time())
                    if segment is None:
                        continue
                    self._handle_segment(segment, vad)
                # Stream ended: flush whatever was buffered, then reconnect.
                tail = vad.flush()
                if tail is not None:
                    self._handle_segment(tail, vad)
            except Exception as exc:  # keep the thread alive across any per-chunk failure
                self._emit(MonitorEvent(time.time(), "error", f"audio loop: {exc}"))
            finally:
                capture.stop()

            if not self._stop.is_set():
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2)

    def _handle_segment(self, segment, vad: EnergyVAD) -> None:
        text, language, emotion = "", "unknown", "unknown"
        if self.asr is not None:
            try:
                transcript = self.asr.transcribe(segment.audio, rate=self.cfg.audio.rate)
            except Exception as exc:
                self._emit(MonitorEvent(time.time(), "error", f"asr: {exc}"))
                transcript = None
            if transcript is None:
                vad.reset()
                return
            text, language, emotion = transcript.text, transcript.language, transcript.emotion

        embedding = None
        if self.speaker is not None:
            try:
                embedding = self.speaker.embed(segment.audio, rate=self.cfg.audio.rate)
            except Exception as exc:
                self._emit(MonitorEvent(time.time(), "error", f"speaker: {exc}"))

        if not text and embedding is None:
            return
        try:
            self._speech_q.put_nowait(_Utterance(
                ts=segment.end_ts, text=text, language=language, emotion=emotion,
                duration_s=segment.duration_s, voice_embedding=embedding,
            ))
        except queue.Full:
            pass    # the main thread is behind; dropping an utterance is better than blocking

    # -- main thread helpers ---------------------------------------------------------
    def _drain_speech(self, now: float) -> List[_Utterance]:
        """Consume utterances from the audio thread and match/record them here."""
        out: List[_Utterance] = []
        while True:
            try:
                item = self._speech_q.get_nowait()
            except queue.Empty:
                break
            out.append(item)

            match = self.memory.person.fuse(voice_vec=item.voice_embedding) \
                if item.voice_embedding is not None else None
            speaker_name = match.name if (match and match.known) else "unknown"
            person_ids = [match.person_id] if (match and match.known) else []

            self.memory.episodic.observe(
                "speech", item.text, ts=item.ts,
                payload={"language": item.language, "emotion": item.emotion,
                         "speaker_name": speaker_name, "known": bool(match and match.known)},
                person_ids=person_ids,
            )
            self.memory.working.add(
                "speech", item.text, ts=item.ts,
                language=item.language, speaker_name=speaker_name,
                known=bool(match and match.known),
            )
            if match and match.known:
                self.memory.person.touch(match.person_id, ts=item.ts)
                self.memory.person.note(match.person_id, item.text, kind="speech", ts=item.ts)

            self._last_speech_ts = max(self._last_speech_ts or 0.0, item.ts)
            self._recent_voices.append({
                "ts": item.ts, "name": speaker_name,
                "known": bool(match and match.known),
                "person_id": match.person_id if match else None,
                "score": match.score if match else 0.0,
            })
            self._stats["utterances"] += 1
            self._pending_event = True
            self._emit(MonitorEvent(item.ts, "speech",
                                    f"[{speaker_name}] {item.text}",
                                    data={"language": item.language}))

        cutoff = now - 10.0
        self._recent_voices = [v for v in self._recent_voices if v["ts"] >= cutoff]
        return out

    def _identities(self, faces: List[Any], now: float) -> List[Dict[str, Any]]:
        """Fuse face matches with voice matches from the same time window."""
        identities: List[Dict[str, Any]] = []
        for face in faces:
            match = self.memory.person.match_face(face.embedding) if face.embedding is not None \
                else None
            entry: Dict[str, Any] = {
                "box": [int(v) for v in face.box],
                "det_score": round(float(face.score), 3),
                "known": bool(match and match.known),
                "name": match.name if (match and match.known) else "unknown",
                "score": round(match.score, 4) if match else 0.0,
                "modality": "face",
            }
            if match and match.known:
                # If a voice in the last 10 s matched the same person, the identity is confirmed
                # by two modalities -- worth recording as such.
                for voice in self._recent_voices:
                    if voice["known"] and voice["person_id"] == match.person_id:
                        entry["modality"] = "both"
                        entry["voice_score"] = round(voice["score"], 4)
                        break
                self.memory.person.touch(match.person_id, ts=now)
            identities.append(entry)
        return identities

    def _save_frame(self, rgb, now: float) -> Optional[str]:
        if not self.cfg.monitor.save_frames:
            return None
        try:
            jpeg = rgb_to_jpeg(rgb, self.cfg.camera.jpeg_quality)
            name = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S-%f") + ".jpg"
            path = os.path.join(self.cfg.frames_dir, name)
            with open(path, "wb") as fh:
                fh.write(jpeg)
            return path
        except OSError:
            return None

    def _should_describe(self, now: float) -> bool:
        since = now - self._last_describe
        if since < self.cfg.monitor.min_describe_gap:
            return False
        if since >= self.cfg.monitor.describe_interval:
            return True
        return self.cfg.monitor.describe_on_event and self._pending_event

    def _apply_decision(self, decision: Decision, obs: Observation, frame_path: Optional[str],
                        identities: List[Dict[str, Any]]) -> None:
        person_ids = sorted({i["person_id"] for i in identities
                             if i.get("known") and i.get("person_id") is not None})
        # A decision that names people but has no ids still needs them for person-scoped queries.
        for entry in decision.people:
            name = str(entry.get("name", "")).strip()
            if name and name != "unknown":
                person = self.memory.person.get(name)
                if person is not None:
                    person_ids.append(person.id)
        person_ids = sorted(set(person_ids))

        importance = decision.importance
        if decision.scene:
            self.memory.working.add("scene", decision.scene, ts=obs.ts,
                                    importance=importance, ok=decision.ok)
            self.memory.episodic.record(
                "scene", decision.scene, ts=obs.ts, importance=importance,
                person_ids=person_ids, frame_path=frame_path,
                detail={"people": decision.people, "parse_ok": decision.ok},
            )
            self._stats["episodes"] += 1
            self._emit(MonitorEvent(obs.ts, "scene", decision.scene,
                                    importance=importance, data={"frame": frame_path}))

        for event in decision.events:
            summary = str(event.get("summary", "")).strip()
            if not summary:
                continue
            self.memory.episodic.record(
                "event", summary, ts=obs.ts,
                detail={"kind": event.get("kind", "event")},
                importance=float(event.get("importance", 0.0) or 0.0),
                person_ids=person_ids, frame_path=frame_path,
            )
            self._stats["episodes"] += 1

        for fact in decision.remember:
            try:
                self.memory.semantic.remember(
                    str(fact.get("subject", "")), str(fact.get("predicate", "")),
                    str(fact.get("object", "")),
                    confidence=float(fact.get("confidence", 0.4) or 0.4),
                    source="agent",
                )
            except ValueError:
                continue

        for note in decision.notes:
            name = str(note.get("name", "")).strip()
            text = str(note.get("text", "")).strip()
            if not name or name == "unknown" or not text:
                continue
            person = self.memory.person.get(name)
            if person is not None:
                self.memory.person.note(person.id, text, kind="agent", ts=obs.ts)

    def _evaluate_rules(self, identities: List[Dict[str, Any]], now: float,
                        frame_path: Optional[str]) -> None:
        ctx = TickContext(
            ts=now,
            identities=identities,
            speech=[{"text": u.text, "language": u.language,
                     "speaker_name": (self._recent_voices[-1]["name"] if self._recent_voices else "unknown")}
                    for u in self.memory.working.recent(kind="speech", n=5)],
            last_speech_ts=self._last_speech_ts,
            person_last_seen={p.name: (p.last_seen or 0.0) for p in self.memory.person.all()},
            started_ts=self._started_ts or None,
        )
        for fired in self.memory.procedural.evaluate(ctx):
            action = fired.action
            kind = action.get("type", "alert")
            if kind == "remember":
                try:
                    self.memory.semantic.remember(
                        str(action.get("subject", "")), str(action.get("predicate", "")),
                        str(action.get("object", "")),
                        confidence=float(action.get("confidence", 0.3) or 0.3),
                        source=f"rule:{fired.rule_name}",
                    )
                except ValueError:
                    pass
                self.memory.episodic.record("rule", fired.message, ts=now,
                                            detail={"rule": fired.rule_name},
                                            importance=fired.importance, frame_path=frame_path)
                continue

            self._alert(fired.message, fired.importance, now,
                        data={"rule": fired.rule_name, "frame": frame_path})

    def _alert(self, message: str, importance: float, now: float, data: Dict[str, Any]) -> None:
        self._stats["alerts"] += 1
        self.memory.episodic.record("alert", message, ts=now, importance=importance,
                                    detail=data, frame_path=data.get("frame"))
        self._emit(MonitorEvent(now, "alert", message, importance, data))

        # Append to the alert log so alerts survive a restart and can be tailed.
        try:
            os.makedirs(os.path.dirname(self.cfg.alert_log) or ".", exist_ok=True)
            with open(self.cfg.alert_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": now,
                    "time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                    "message": message, "importance": importance, **data,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

        if self.cfg.alerts.webhook:
            try:
                payload = json.dumps({"message": message, "importance": importance,
                                      "ts": now, **data}, ensure_ascii=False).encode()
                req = urllib.request.Request(
                    self.cfg.alerts.webhook, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5).close()
            except Exception as exc:      # a broken webhook must not stop the monitor
                self._emit(MonitorEvent(now, "error", f"webhook failed: {exc}"))

    # -- the loop -------------------------------------------------------------------
    def run(self, *, max_ticks: int = 0, duration_s: float = 0.0) -> int:
        """Run the monitor.  Returns the number of ticks executed.

        ``max_ticks``/``duration_s`` exist for tests and for ``selftest``; a real 24/7 run
        passes neither and stops only on SIGINT/SIGTERM.
        """
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                continue    # not the main thread, or unsupported on this platform

        ticks = 0
        started = 0.0
        try:
            # Inside the try so a startup failure -- no camera, an unreadable database -- still
            # releases the camera and the SQLite connection instead of leaking them.
            self._setup()
            started = time.time()
            self._emit(MonitorEvent(time.time(), "tick", "monitor started",
                                    data={"face": self._face_ready,
                                          "audio": self._audio_thread is not None,
                                          "db": self.cfg.db_path}))
            while not self._stop.is_set():
                t0 = time.monotonic()
                now = time.time()
                try:
                    self._tick(now)
                except Exception as exc:     # one bad tick must not end a 24/7 run
                    self._emit(MonitorEvent(time.time(), "error", f"tick failed: {exc}"))
                ticks += 1
                self._stats["ticks"] = ticks

                if max_ticks and ticks >= max_ticks:
                    break
                if duration_s and (time.time() - started) >= duration_s:
                    break
                sleep = self.cfg.monitor.tick_interval - (time.monotonic() - t0)
                if sleep > 0:
                    self._stop.wait(sleep)
        finally:
            self._stop.set()
            if self._audio_thread is not None:
                self._audio_thread.join(timeout=3.0)
            self.camera.close()
            self.memory.close()
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
        self._emit(MonitorEvent(time.time(), "tick", "monitor stopped",
                                data={"ticks": ticks}))
        return ticks

    def _on_signal(self, signum, frame) -> None:   # noqa: ARG002
        self._stop.set()

    def _tick(self, now: float) -> None:
        frame = self.camera.grab()
        if frame is None:
            self._emit(MonitorEvent(now, "error", "camera returned no frame"))
            return
        self._stats["frames"] += 1

        faces = self.face.analyze(frame.rgb) if self._face_ready else []
        self._stats["faces"] += len(faces)
        self._drain_speech(now)
        identities = self._identities(faces, now)

        # Record what was seen even when the agent is not called, so the raw log is complete.
        self.memory.episodic.observe(
            "frame", ts=now,
            payload={"faces": [i for i in identities], "face_count": len(faces)},
            person_ids=[i["person_id"] for i in identities
                        if i.get("known") and i.get("person_id") is not None],
        )
        if faces:
            self.memory.working.add("face", ts=now, identities=identities)
            self._pending_event = True

        due = self._should_describe(now)
        if due:
            jpeg = rgb_to_jpeg(frame.rgb, self.cfg.camera.jpeg_quality)
            # The frame is saved whenever an episode is about to be written, so a summary in the
            # database can always be checked against what the camera saw.
            frame_path = self._save_frame(frame.rgb, now)
            obs = Observation(ts=now, frame_jpeg=jpeg, faces=identities,
                              speech=[], frame_path=frame_path)
            decision = self.agent.decide(obs, self.memory.context())
            self._last_describe = now
            self._pending_event = False
            self._stats["describes"] += 1
            if not decision.ok:
                self._stats["describe_errors"] += 1
                self._emit(MonitorEvent(now, "error",
                                        f"agent reply unparsed: {decision.error}"))
            self._apply_decision(decision, obs, frame_path, identities)

        self._evaluate_rules(identities, now, None)

        # Hourly archive/retention pass.
        if now - self._last_archive >= 3600.0:
            self._last_archive = now
            try:
                report = self.memory.archiver.run(now)
                self._emit(MonitorEvent(now, "tick", "archived memory", data=report))
            except Exception as exc:
                self._emit(MonitorEvent(now, "error", f"archive failed: {exc}"))

    # -- introspection ---------------------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def stop(self) -> None:
        self._stop.set()
