"""Energy VAD: segmenting the microphone stream into utterances."""

from __future__ import annotations

import numpy as np

from qwen25vl_agent.perception.vad import EnergyVAD

RATE = 16000
CHUNK = 512          # 512 samples at 16 kHz = 32 ms


def loud() -> np.ndarray:
    return np.full(CHUNK, 0.5, dtype=np.float32)     # about -6 dBFS


def quiet() -> np.ndarray:
    return np.zeros(CHUNK, dtype=np.float32)


def _vad(**over):
    kwargs = dict(start_db=-32.0, hangover_ms=400, min_speech_ms=300, max_segment_s=20.0, rate=RATE)
    kwargs.update(over)
    return EnergyVAD(**kwargs)


def test_silence_never_starts_speech():
    vad = _vad()
    ts = 0.0
    for _ in range(20):
        ts += 0.032
        assert vad.push(quiet(), ts) is None


def test_burst_then_hangover_yields_a_segment():
    vad = _vad()
    ts = 0.032
    assert vad.push(loud(), ts) is None      # speech starts; not ended yet

    segment = None
    for _ in range(20):
        ts += 0.032
        segment = vad.push(quiet(), ts)
        if segment is not None:
            break

    assert segment is not None
    assert segment.peak_db > -10.0           # a 0.5-amplitude tone is roughly -6 dBFS
    assert segment.audio.size >= CHUNK
    assert segment.duration_s > 0.0


def test_short_burst_is_dropped_as_noise():
    """With a short hangover, a single 32 ms click is under min_speech_ms and is discarded."""
    vad = _vad(hangover_ms=100, min_speech_ms=300)
    ts = 0.032
    assert vad.push(loud(), ts) is None
    segment = None
    for _ in range(10):
        ts += 0.032
        segment = vad.push(quiet(), ts)
        if segment is not None:
            break
    assert segment is None


def test_long_speech_is_force_cut():
    """A talkative stream must be cut at max_segment_s so transcription stays responsive."""
    vad = _vad(hangover_ms=10000, min_speech_ms=0, max_segment_s=0.2)
    ts = 0.0
    segment = None
    for _ in range(20):
        ts += 0.032
        out = vad.push(loud(), ts)
        if out is not None:
            segment = out
            break
    assert segment is not None
    assert segment.duration_s >= 0.2


def test_flush_returns_the_open_segment():
    vad = _vad()
    assert vad.push(loud(), 0.032) is None
    segment = vad.flush()
    assert segment is not None
    assert segment.audio.size == CHUNK
    # A second flush has nothing left to return.
    assert vad.flush() is None


def test_reset_discards_state():
    vad = _vad()
    vad.push(loud(), 0.032)
    vad.reset()
    assert vad.flush() is None


def test_hysteresis_keeps_a_dip_alive():
    """A chunk just below the start threshold still counts as voice (hysteresis).

    -36 dBFS is below ``start_db`` (-32) so it would not *start* speech, but it is above
    ``start_db - hysteresis_db`` (-38) so it keeps an ongoing utterance alive.
    """
    vad = _vad(hysteresis_db=6.0, hangover_ms=100, min_speech_ms=0, max_segment_s=0.2)
    dip = np.full(CHUNK, 10 ** (-36 / 20.0), dtype=np.float32)
    assert vad.push(loud(), 0.032) is None

    ts = 0.032
    segment = None
    for _ in range(10):
        ts += 0.032
        segment = vad.push(dip, ts)
        if segment is not None:
            break

    # The dip keeps `_last_voice_ts` moving, so the 100 ms hangover never expires; the segment
    # only ends because max_segment_s force-cuts it.
    assert segment is not None
    assert segment.duration_s >= 0.2
