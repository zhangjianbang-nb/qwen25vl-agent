"""Speaker (voiceprint) embeddings via funasr's CAM++ model.

Voice is the second half of the person memory: face recognition needs the person to be visible
and facing the camera, while a voiceprint survives a turned back and a dark room.  Fusing them
(see ``identity.mode``) is what makes "who is here" workable over a full day rather than only
when someone looks at the lens.

CAM++ produces a 192-d embedding from a 16 kHz utterance; the same cosine machinery as faces
then matches it against the stored voiceprints.  The model runs on CPU for the same reason as
ASR -- the GPU is fully committed to the served Qwen2.5-VL.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..vectors import cosine, normalize

__all__ = ["SpeakerEngine", "SpeakerUnavailable"]


class SpeakerUnavailable(RuntimeError):
    """funasr is missing, the model could not be loaded, or the audio was unusable."""


class SpeakerEngine:
    """Lazily-loaded CAM++ wrapper producing normalized 192-d voiceprints."""

    def __init__(self, model: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                 device: str = "cpu", min_seconds: float = 0.6) -> None:
        self.model_id = model
        self.device = device
        # A voiceprint from under ~0.6 s of speech is unstable; skip rather than pollute the
        # gallery with a noisy embedding.
        self.min_seconds = min_seconds
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import funasr  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise SpeakerUnavailable(
                "voiceprints need funasr: pip install 'qwen25vl-agent[asr]'"
            ) from exc
        try:
            self._model = AutoModel(
                model=self.model_id,
                task="speaker-verification",
                device=self.device,
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:
            raise SpeakerUnavailable(f"could not load speaker model {self.model_id}: {exc}") from exc

    def embed(self, audio: np.ndarray, rate: int = 16000) -> Optional[np.ndarray]:
        """192-d L2-normalized voiceprint, or ``None`` when the clip is too short/failed."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < self.min_seconds * rate:
            return None
        self.load()
        assert self._model is not None

        embedding = None
        # `AutoModel.inference` forwards to the underlying CAMPPlus model, whose `inference`
        # returns (results, meta_data) with results[0]["spk_embedding"] of shape (1, 192).
        try:
            results, _ = self._model.inference(audio, device=self.device, fs=rate)
            embedding = results[0]["spk_embedding"]
        except Exception:
            # Older funasr builds route embedding extraction through `generate`.
            try:
                results = self._model.generate(input=audio, disable_pbar=True)
                first = results[0]
                embedding = first.get("spk_embedding") if isinstance(first, dict) else None
            except Exception as exc:
                raise SpeakerUnavailable(f"speaker embedding failed: {exc}") from exc

        if embedding is None:
            return None
        vec = np.asarray(embedding.detach().cpu().numpy() if hasattr(embedding, "detach")
                         else embedding, dtype=np.float32).reshape(-1)
        return normalize(vec)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine(a, b)

    def warmup(self) -> bool:
        try:
            self.embed(np.zeros(int(self.min_seconds * 16000) + 1600, dtype=np.float32))
            return True
        except SpeakerUnavailable:
            return False
