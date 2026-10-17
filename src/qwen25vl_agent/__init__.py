"""qwen25vl-agent -- a 24/7 camera monitoring agent.

The package has four parts, kept in separate subpackages so each can be tested without the
others:

* ``perception`` -- video sampling, face detection/recognition, and speech recognition with
  speaker (voiceprint) embeddings.
* ``memory`` -- five stores: person (face + voiceprint), working, episodic, semantic and
  procedural.
* ``agent`` -- the reasoning step: build a prompt from memory plus the current observation,
  ask a local Qwen model, and parse a structured decision back out.
* ``monitor`` -- the long-running loop that ties perception, memory and the agent together.

``llm`` is the HTTP client for the reasoning engine.  This package never loads a model
itself: on an 8 GB card a served Qwen2.5-VL already holds ~5.9 GB, so the agent talks to a
*running* ``qwen25vl-lite serve`` over HTTP instead of building a second copy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
