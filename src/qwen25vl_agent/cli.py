"""Command-line interface.

Subcommands, grouped by what the operator is doing:

* watching -- ``run`` (the 24/7 loop), ``selftest`` (a short end-to-end check)
* enrolling -- ``enroll`` (face + voice), ``voiceprint`` (compare or enroll a voice)
* inspecting -- ``persons``, ``mem``, ``rules``, ``transcribe``
* diagnosing -- ``doctor``

Every command takes ``--config`` and ``--db``; ``run`` additionally accepts the cadence and
vision-budget flags.  Nothing here loads the Qwen model: the agent always talks to a running
``qwen25vl-lite serve``, because a second copy would not fit in the card's VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config
from .llm import LLMClient, LLMError

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Translate CLI flags into a config override mapping."""
    out: Dict[str, Any] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            out.setdefault(section, {})[key] = value

    put("llm", "url", getattr(args, "url", None))
    put("llm", "max_vision_tokens", getattr(args, "max_vision_tokens", None))
    put("llm", "timeout", getattr(args, "timeout", None))
    put("llm", "max_tokens", getattr(args, "max_tokens", None))
    put("camera", "device", getattr(args, "device", None))
    put("camera", "width", getattr(args, "width", None))
    put("camera", "jpeg_quality", getattr(args, "jpeg_quality", None))
    put("monitor", "tick_interval", getattr(args, "interval", None))
    put("monitor", "describe_interval", getattr(args, "describe_interval", None))
    put("monitor", "save_frames", getattr(args, "save_frames", None))
    put("memory", "path", getattr(args, "db", None))
    if getattr(args, "no_audio", False):
        out.setdefault("audio", {})["enabled"] = False
    put("audio", "device", getattr(args, "audio_device", None))
    put("asr", "backend", getattr(args, "asr_backend", None))
    put("speaker", "backend", getattr(args, "speaker_backend", None))
    return out


def _make_engines(cfg):
    """Build the optional perception engines, honoring the configured backends."""
    asr = speaker = None
    if cfg.asr.backend != "none":
        from .perception.asr import ASREngine

        asr = ASREngine(model=cfg.asr.model, language=cfg.asr.language, use_itn=cfg.asr.use_itn)
    if cfg.speaker.backend != "none":
        from .perception.speaker import SpeakerEngine

        speaker = SpeakerEngine(model=cfg.speaker.model)
    return asr, speaker


def _load_audio(path: str, sr: int = 16000):
    """Read an audio file to float32 mono at ``sr``.  Tries librosa, then soundfile, then torchaudio."""
    try:
        import librosa

        y, _ = librosa.load(path, sr=sr, mono=True)
        return y
    except Exception:
        pass
    try:
        import soundfile as sf

        data, file_sr = sf.read(path, always_2d=False)
        import numpy as np

        y = data.mean(axis=1) if data.ndim > 1 else data
        if file_sr != sr:
            # Linear resample is good enough for a 16 kHz ASR front end.
            n = int(round(len(y) * sr / file_sr))
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        return y.astype("float32")
    except Exception:
        pass
    import torchaudio

    waveform, file_sr = torchaudio.load(path)
    if file_sr != sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)
    return waveform.mean(dim=0).numpy().astype("float32")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str))


def _load_image(path: str):
    """Load any image format to an RGB array (used by ``enroll --image``)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _human_time(ts: Optional[float]) -> str:
    return "-" if not ts else time.strftime("%m-%d %H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .monitor import Monitor

    asr, speaker = _make_engines(cfg)
    monitor = Monitor(cfg, asr=asr, speaker=speaker)

    json_out = args.json

    def emit(event) -> None:
        if json_out:
            _print_json(event.as_dict())
            return
        stamp = time.strftime("%H:%M:%S", time.localtime(event.ts))
        tag = event.kind.upper()
        if event.kind == "alert":
            print(f"[{stamp}] ALERT({event.importance:.2f}) {event.message}", flush=True)
        elif event.kind == "speech":
            print(f"[{stamp}] SPEECH {event.message}", flush=True)
        elif event.kind == "scene":
            print(f"[{stamp}] SCENE {event.message}", flush=True)
        elif event.kind == "error":
            print(f"[{stamp}] ERROR {event.message}", file=sys.stderr, flush=True)
        elif event.kind == "tick" and args.verbose:
            print(f"[{stamp}] {tag} {event.message}", flush=True)

    monitor.on_event(emit)
    ticks = monitor.run(max_ticks=args.max_ticks, duration_s=args.duration)
    if not json_out:
        print(f"\nstopped after {ticks} ticks: {json.dumps(monitor.stats, ensure_ascii=False)}")
    return 0


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> int:
    """A short end-to-end run: camera -> faces -> agent -> memory, with a report."""
    cfg = load_config(args.config, _overrides(args))
    from .monitor import Monitor

    asr, speaker = _make_engines(cfg)
    monitor = Monitor(cfg, asr=asr, speaker=speaker)
    seen: List[dict] = []
    monitor.on_event(lambda e: seen.append(e.as_dict()))

    started = time.time()
    ticks = monitor.run(max_ticks=args.ticks, duration_s=args.duration)
    report = {
        "ticks": ticks,
        "elapsed_s": round(time.time() - started, 2),
        "stats": monitor.stats,
        "events": seen[-20:],
        "memory": None,
    }
    try:
        from .memory import Memory

        with Memory(cfg) as mem:
            report["memory"] = mem.stats()
    except Exception as exc:
        report["memory"] = f"unavailable: {exc}"

    if args.json:
        _print_json(report)
    else:
        print(f"selftest: {ticks} ticks in {report['elapsed_s']}s")
        print(f"stats:   {json.dumps(monitor.stats, ensure_ascii=False)}")
        for event in seen[-10:]:
            print(f"  [{event['time']}] {event['kind']}: {event['message'][:100]}")
        if report["memory"]:
            print(f"memory:  {json.dumps(report['memory'], ensure_ascii=False)}")
    return 0 if ticks else 1


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------
def cmd_enroll(args: argparse.Namespace) -> int:
    """Enroll a person's face (and optionally voice) into person memory.

    Enrollment can come from the live camera (default) or from still images
    (``--image``), which is the practical path when the person is not standing in front of the
    lens right now -- a passport photo or a screenshot works fine.
    """
    cfg = load_config(args.config, _overrides(args))
    from .memory import Memory
    from .perception.face import FaceEngine

    face = FaceEngine(cfg.detector_model_path, cfg.recognizer_model_path,
                      detect_score=cfg.face.detect_score, min_face_px=cfg.face.min_face_px)
    if not face.available():
        print(f"face models unavailable ({cfg.detector_model_path}); "
              "run scripts/fetch_models.sh", file=sys.stderr)
        return 2

    collected: List[Any] = []

    if args.image:
        from .imaging import decode_jpeg
        from PIL import Image

        for path in args.image:
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                rgb = decode_jpeg(raw) if path.lower().endswith((".jpg", ".jpeg")) \
                    else _load_image(path)
            except Exception as exc:
                print(f"{path}: {exc}", file=sys.stderr)
                continue
            faces = face.analyze(rgb)
            if not faces:
                print(f"{path}: no face detected", file=sys.stderr)
                continue
            best = max(faces, key=lambda f: f.area)
            if best.embedding is not None:
                collected.append(best)
                print(f"{path}: face {best.box} score {best.score:.2f}")
    else:
        from .perception.video import CameraSource

        camera = CameraSource(device=cfg.camera.device, width=cfg.camera.width)
        camera.open()
        try:
            deadline = time.time() + args.seconds
            while time.time() < deadline and len(collected) < args.frames:
                frame = camera.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue
                faces = face.analyze(frame.rgb)
                if not faces:
                    continue
                # The largest face is the one most likely to be the person being enrolled.
                best = max(faces, key=lambda f: f.area)
                if best.embedding is not None:
                    collected.append(best)
        finally:
            camera.close()

    if not collected:
        print("no usable face captured; make sure the face fills a good part of the frame",
              file=sys.stderr)
        return 3

    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        person = mem.person.ensure(args.name, notes=args.notes or "")
        for det in collected:
            mem.person.add_face(person, det.embedding, quality=det.score)
        mem.person.touch(person.id)
        if args.audio_seconds > 0:
            voice = _capture_voiceprint(cfg, args.audio_seconds)
            if voice is not None:
                mem.person.add_voice(person, voice, quality=1.0)
            else:
                print("voice capture produced no usable voiceprint (skipped)", file=sys.stderr)
        counts = mem.person.embedding_counts(person.id)

    if args.json:
        _print_json({"name": args.name, "faces": len(collected), **counts})
    else:
        print(f"enrolled {args.name}: {len(collected)} face embeddings, "
              f"{counts['voices']} voiceprints")
    return 0


def _capture_voiceprint(cfg, seconds: float):
    """Record ``seconds`` of microphone audio and return a voiceprint, or ``None``."""
    from .perception.audio import AudioCapture, AudioUnavailable, pcm16_to_float
    from .perception.speaker import SpeakerEngine

    engine = SpeakerEngine(model=cfg.speaker.model)
    try:
        capture = AudioCapture(device=cfg.audio.device, rate=cfg.audio.rate,
                               channels=cfg.audio.channels, chunk_ms=cfg.audio.chunk_ms)
        capture.start()
    except AudioUnavailable as exc:
        print(f"audio unavailable: {exc}", file=sys.stderr)
        return None

    chunks = []
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            chunk = capture.read_float()
            if chunk is None:
                break
            chunks.append(chunk)
    finally:
        capture.stop()

    import numpy as np

    if not chunks:
        return None
    audio = np.concatenate(chunks)
    try:
        return engine.embed(audio, rate=cfg.audio.rate)
    except Exception as exc:
        print(f"voiceprint failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# voiceprint / transcribe
# ---------------------------------------------------------------------------
def cmd_voiceprint(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .perception.speaker import SpeakerEngine
    from .vectors import cosine

    engine = SpeakerEngine(model=cfg.speaker.model)
    if len(args.files) < 1:
        print("give at least one wav (two to compare)", file=sys.stderr)
        return 2
    embs = []
    for path in args.files[:2]:
        audio = _load_audio(path, cfg.audio.rate)
        emb = engine.embed(audio, rate=cfg.audio.rate)
        if emb is None:
            print(f"{path}: no usable voiceprint", file=sys.stderr)
            return 3
        embs.append(emb)
        print(f"{path}: {emb.size}-d voiceprint")
    if len(embs) == 2:
        score = cosine(embs[0], embs[1])
        verdict = "same speaker" if score >= cfg.speaker.threshold else "different speaker"
        if args.json:
            _print_json({"similarity": score, "threshold": cfg.speaker.threshold, "verdict": verdict})
        else:
            print(f"similarity {score:.4f} (threshold {cfg.speaker.threshold}) -> {verdict}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .perception.asr import ASREngine

    engine = ASREngine(model=cfg.asr.model, language=cfg.asr.language, use_itn=cfg.asr.use_itn)

    if args.mic:
        from .perception.audio import AudioCapture, AudioUnavailable

        import numpy as np

        try:
            capture = AudioCapture(device=cfg.audio.device, rate=cfg.audio.rate,
                                   channels=cfg.audio.channels, chunk_ms=cfg.audio.chunk_ms)
            capture.start()
        except AudioUnavailable as exc:
            print(f"audio unavailable: {exc}", file=sys.stderr)
            return 2
        chunks = []
        try:
            deadline = time.time() + args.mic
            while time.time() < deadline:
                chunk = capture.read_float()
                if chunk is None:
                    break
                chunks.append(chunk)
        finally:
            capture.stop()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
    else:
        if not args.files:
            print("give a wav path, or --mic SECONDS", file=sys.stderr)
            return 2
        audio = _load_audio(args.files[0], cfg.audio.rate)

    transcript = engine.transcribe(audio, rate=cfg.audio.rate)
    if transcript is None:
        print("(nothing recognised)")
        return 1
    if args.json:
        _print_json(transcript.as_dict())
    else:
        print(f"[{transcript.language}] {transcript.text}")
    return 0


# ---------------------------------------------------------------------------
# persons / mem / rules
# ---------------------------------------------------------------------------
def cmd_persons(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .memory import Memory

    with Memory(cfg) as mem:
        people = mem.person.all()
        if args.json:
            _print_json([{**p.as_dict(), **mem.person.embedding_counts(p.id)} for p in people])
            return 0
        if not people:
            print("no people enrolled yet (use: qwen25vl-agent enroll --name <name>)")
            return 0
        print(f"{'id':>4} {'name':<16} {'faces':>5} {'voices':>6} {'seen':>5}  last seen")
        for p in people:
            counts = mem.person.embedding_counts(p.id)
            print(f"{p.id:>4} {p.name:<16} {counts['faces']:>5} {counts['voices']:>6} "
                  f"{p.seen_count:>5}  {_human_time(p.last_seen)}")
        if args.history:
            for p in people:
                print(f"\n-- {p.name} --")
                for item in mem.person.history(p.id, limit=10):
                    print(f"  {_human_time(item['ts'])} {item['kind']}: {item['text'][:100]}")
    return 0


def cmd_mem(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .memory import Memory

    with Memory(cfg, frames_dir=cfg.frames_dir) as mem:
        what = args.what
        if what == "stats":
            _print_json(mem.stats()) if args.json else print(
                json.dumps(mem.stats(), ensure_ascii=False, indent=2))
        elif what == "episodes":
            rows = [e.as_dict() for e in mem.episodic.recent(limit=args.limit)]
            _print_json(rows) if args.json else _print_episodes(rows)
        elif what == "facts":
            rows = [f.as_dict() for f in mem.semantic.get(limit=args.limit)]
            _print_json(rows) if args.json else _print_facts(rows)
        elif what == "rules":
            rows = [r.as_dict() for r in mem.procedural.all()]
            _print_json(rows) if args.json else _print_rules(rows)
        elif what == "search":
            if not args.query:
                print("--query is required for search", file=sys.stderr)
                return 2
            eps = [e.as_dict() for e in mem.episodic.search(args.query, limit=args.limit)]
            facts = [f.as_dict() for f in mem.semantic.search(args.query, limit=args.limit)]
            payload = {"episodes": eps, "facts": facts}
            _print_json(payload) if args.json else (
                _print_episodes(eps), _print_facts(facts))
        elif what == "archive":
            report = mem.archiver.run()
            _print_json(report) if args.json else print(
                json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"unknown --what {what}", file=sys.stderr)
            return 2
    return 0


def _print_episodes(rows: List[dict]) -> None:
    if not rows:
        print("(no episodes)")
        return
    for r in rows:
        print(f"{_human_time(r['ts'])} [{r['kind']}] (imp {r['importance']:.2f}) {r['summary']}")


def _print_facts(rows: List[dict]) -> None:
    if not rows:
        print("(no facts)")
        return
    for r in rows:
        print(f"(conf {r['confidence']:.2f}, {r['source'] or '-'}) "
              f"{r['subject']} {r['predicate']} {r['object']}")


def _print_rules(rows: List[dict]) -> None:
    if not rows:
        print("(no rules)")
        return
    for r in rows:
        state = "on " if r["enabled"] else "off"
        print(f"[{state}] {r['name']} (cooldown {r['cooldown_s']:.0f}s) "
              f"{r['trigger'].get('type')} -> {r['action'].get('type')}")
        if r["description"]:
            print(f"        {r['description']}")


def cmd_rules(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides(args))
    from .memory import Memory

    with Memory(cfg) as mem:
        if args.action == "list":
            rows = [r.as_dict() for r in mem.procedural.all()]
            _print_json(rows) if args.json else _print_rules(rows)
        elif args.action == "add":
            if not args.name or not args.trigger or not args.action_json:
                print("add needs --name, --trigger JSON and --action JSON", file=sys.stderr)
                return 2
            rule = mem.procedural.add(
                args.name, json.loads(args.trigger), json.loads(args.action_json),
                description=args.description or "", cooldown_s=args.cooldown,
            )
            print(f"added rule {rule.name}")
        elif args.action in ("enable", "disable"):
            ok = mem.procedural.set_enabled(args.name, args.action == "enable")
            print(f"{args.action}d {args.name}" if ok else f"no rule named {args.name}")
            return 0 if ok else 3
        elif args.action == "remove":
            ok = mem.procedural.remove(args.name)
            print(f"removed {args.name}" if ok else f"no rule named {args.name}")
            return 0 if ok else 3
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """Check every dependency the agent needs and say which are missing."""
    cfg = load_config(args.config, _overrides(args))
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, critical: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "critical": critical})

    # 1. reasoning engine
    client = LLMClient(url=cfg.llm.url, timeout=min(cfg.llm.timeout, 10.0))
    try:
        health = client.health()
        add("llm", True, f"{cfg.llm.url} -> {health.get('model', '?')}")
    except LLMError as exc:
        add("llm", False, f"{cfg.llm.url} unreachable: {exc}")

    # 2. camera
    try:
        from .perception.video import CameraSource

        cam = CameraSource(device=cfg.camera.device, width=cfg.camera.width)
        cam.open()
        frame = cam.grab()
        cam.close()
        if frame is None:
            add("camera", False, f"/dev/video{cfg.camera.device} opened but returned no frame")
        else:
            h, w = frame.rgb.shape[:2]
            add("camera", True, f"/dev/video{cfg.camera.device} {w}x{h}")
    except Exception as exc:
        add("camera", False, f"/dev/video{cfg.camera.device}: {exc}")

    # 3. face models
    try:
        from .perception.face import FaceEngine

        engine = FaceEngine(cfg.detector_model_path, cfg.recognizer_model_path,
                            detect_score=cfg.face.detect_score, min_face_px=cfg.face.min_face_px)
        ok = engine.available()
        detail = "YuNet + SFace loaded" if ok else (
            f"missing models: {cfg.detector_model_path} / {cfg.recognizer_model_path}")
        add("face", ok, detail, critical=False)
    except Exception as exc:
        add("face", False, str(exc), critical=False)

    # 4. audio devices
    try:
        from .perception.audio import list_devices

        devices = list_devices()
        add("audio-devices", bool(devices), ", ".join(devices) or "arecord found no capture device",
            critical=False)
    except Exception as exc:
        add("audio-devices", False, str(exc), critical=False)

    # 5. ASR backend
    try:
        from .perception.asr import ASREngine

        ok = ASREngine.available()
        add("asr", ok, "funasr present" if ok else "funasr not installed "
            "(pip install 'qwen25vl-agent[asr]')", critical=False)
    except Exception as exc:
        add("asr", False, str(exc), critical=False)

    # 6. speaker backend
    try:
        from .perception.speaker import SpeakerEngine

        ok = SpeakerEngine.available()
        add("speaker", ok, "funasr present" if ok else "funasr not installed", critical=False)
    except Exception as exc:
        add("speaker", False, str(exc), critical=False)

    # 7. disk for memory + archive
    try:
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        usage = shutil.disk_usage(os.path.dirname(cfg.db_path) or ".")
        free_gb = usage.free / 1e9
        add("disk", free_gb > 1.0, f"{free_gb:.1f} GB free for {cfg.db_path}")
    except Exception as exc:
        add("disk", False, str(exc), critical=False)

    # 8. memory database
    try:
        from .memory import Memory

        with Memory(cfg) as mem:
            stats = mem.stats()
        add("memory", True, f"{stats['persons']} people, {stats['episodes']} episodes, "
                            f"{stats['facts']} facts")
    except Exception as exc:
        add("memory", False, str(exc))

    ok_all = all(c["ok"] or not c["critical"] for c in checks)
    if args.json:
        _print_json({"ok": ok_all, "checks": checks})
        return 0 if ok_all else 1
    for c in checks:
        mark = "ok  " if c["ok"] else ("FAIL" if c["critical"] else "warn")
        print(f"[{mark}] {c['check']:<14} {c['detail']}")
    print(f"\nconfig: {args.config or DEFAULT_CONFIG_PATH}")
    print(f"overall: {'ready' if ok_all else 'not ready (a critical check failed)'}")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--db", default=None, help="memory database path")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen25vl-agent",
        description="24/7 camera monitoring agent: video + speech + face perception, a local "
                    "Qwen2.5-VL brain, and a five-part multimodal memory.",
    )
    parser.add_argument("--version", action="version", version=f"qwen25vl-agent {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p = sub.add_parser("run", help="start the 24/7 monitoring loop")
    _add_common(p)
    p.add_argument("--url", default=None, help="qwen25vl-lite serve URL")
    p.add_argument("--device", type=int, default=None, help="camera /dev/video<N>")
    p.add_argument("--width", type=int, default=None, help="capture width")
    p.add_argument("--interval", type=float, default=None, help="perception tick interval (s)")
    p.add_argument("--describe-interval", type=float, default=None,
                   help="seconds between VL scene descriptions")
    p.add_argument("--max-vision-tokens", type=int, default=None,
                   help="request-level image budget (server --vision-tier does not change pixels)")
    p.add_argument("--max-tokens", type=int, default=None, help="max tokens per agent reply")
    p.add_argument("--timeout", type=float, default=None, help="HTTP timeout per agent call")
    p.add_argument("--audio-device", default=None, help="arecord device, e.g. plughw:1,0")
    p.add_argument("--no-audio", action="store_true", help="disable audio/ASR entirely")
    p.add_argument("--asr-backend", default=None, choices=["funasr", "none"])
    p.add_argument("--speaker-backend", default=None, choices=["funasr", "none"])
    p.add_argument("--no-save-frames", dest="save_frames", action="store_false", default=None)
    p.add_argument("--max-ticks", type=int, default=0, help="stop after N ticks (testing)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (testing)")
    p.add_argument("--verbose", action="store_true", help="also print tick-level events")
    p.set_defaults(func=cmd_run)

    # selftest
    p = sub.add_parser("selftest", help="short end-to-end run with a report")
    _add_common(p)
    p.add_argument("--url", default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--ticks", type=int, default=4, help="how many ticks to run")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--asr-backend", default=None, choices=["funasr", "none"])
    p.add_argument("--speaker-backend", default=None, choices=["funasr", "none"])
    p.add_argument("--audio-device", default=None)
    p.add_argument("--interval", type=float, default=None)
    p.add_argument("--describe-interval", type=float, default=None)
    p.add_argument("--max-vision-tokens", type=int, default=None)
    p.set_defaults(func=cmd_selftest)

    # enroll
    p = sub.add_parser("enroll", help="enroll a person's face (and voice)")
    _add_common(p)
    p.add_argument("--name", required=True, help="person name")
    p.add_argument("--image", nargs="+", default=None,
                   help="enroll from still image(s) instead of the camera")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--frames", type=int, default=8, help="how many face samples to collect")
    p.add_argument("--seconds", type=float, default=15.0, help="capture time budget")
    p.add_argument("--audio-seconds", type=float, default=0.0,
                   help="also record N seconds of speech for a voiceprint")
    p.add_argument("--audio-device", default=None)
    p.add_argument("--notes", default=None, help="free-form notes stored with the person")
    p.set_defaults(func=cmd_enroll)

    # persons
    p = sub.add_parser("persons", help="list enrolled people")
    _add_common(p)
    p.add_argument("--history", action="store_true", help="also show recent interactions")
    p.set_defaults(func=cmd_persons)

    # mem
    p = sub.add_parser("mem", help="inspect memory")
    _add_common(p)
    p.add_argument("--what", default="stats",
                   choices=["stats", "episodes", "facts", "rules", "search", "archive"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--query", default=None)
    p.set_defaults(func=cmd_mem)

    # rules
    p = sub.add_parser("rules", help="manage procedural rules")
    _add_common(p)
    p.add_argument("action", choices=["list", "add", "enable", "disable", "remove"])
    p.add_argument("--name", default=None)
    p.add_argument("--trigger", default=None, help="JSON, e.g. '{\"type\":\"unknown_face\"}'")
    p.add_argument("--action-json", default=None,
                   help="JSON, e.g. '{\"type\":\"alert\",\"message\":\"...\"}'")
    p.add_argument("--description", default=None)
    p.add_argument("--cooldown", type=float, default=60.0)
    p.set_defaults(func=cmd_rules)

    # transcribe
    p = sub.add_parser("transcribe", help="one-shot speech recognition")
    _add_common(p)
    p.add_argument("files", nargs="*", help="audio file(s)")
    p.add_argument("--mic", type=float, default=0.0, help="record N seconds from the mic instead")
    p.add_argument("--audio-device", default=None)
    p.set_defaults(func=cmd_transcribe)

    # voiceprint
    p = sub.add_parser("voiceprint", help="compare or inspect speaker embeddings")
    _add_common(p)
    p.add_argument("files", nargs="+", help="one wav to embed, or two to compare")
    p.set_defaults(func=cmd_voiceprint)

    # doctor
    p = sub.add_parser("doctor", help="check dependencies and readiness")
    _add_common(p)
    p.add_argument("--url", default=None)
    p.add_argument("--device", type=int, default=None)
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
