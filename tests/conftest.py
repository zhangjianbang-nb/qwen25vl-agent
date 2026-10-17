"""Shared fixtures.

The tests are deliberately hermetic: every config points its database, archive and alert log at
a pytest ``tmp_path``, and audio is off, so nothing touches the real ``~/.qwen25vl-agent`` state
or the machine's microphone.  Hardware-dependent paths (camera, face models, the served model)
are either skipped or replaced with fakes.
"""

from __future__ import annotations

import pytest

from qwen25vl_agent.config import load_config


@pytest.fixture
def make_cfg(tmp_path):
    """A factory building a config rooted entirely in ``tmp_path``.

    Call it as ``make_cfg(monitor={"tick_interval": 0.01})``; each named section is merged onto
    the hermetic base, so a test only states the fields it cares about.
    """

    def _make(**sections):
        base = {
            "memory": {
                "path": str(tmp_path / "memory.db"),
                "archive_dir": str(tmp_path / "archive"),
            },
            "alerts": {"log_path": str(tmp_path / "alerts.log")},
            "audio": {"enabled": False},
        }
        for name, values in sections.items():
            base[name] = {**(base.get(name) or {}), **values}
        return load_config(overrides=base)

    return _make


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()
