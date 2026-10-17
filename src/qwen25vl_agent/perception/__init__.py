"""Perception backends: video sampling, face recognition, and speech/speaker analysis.

Each backend is optional and reports its own availability, so a deployment without OpenCV or
without funasr still runs -- it just degrades to the parts it can do, and says so.
"""

from __future__ import annotations

__all__ = ["video", "face", "audio", "vad", "asr", "speaker"]
