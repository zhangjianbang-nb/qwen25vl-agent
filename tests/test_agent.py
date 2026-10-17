"""The agent step: prompt construction and defensive parsing of the model's JSON reply."""

from __future__ import annotations

from qwen25vl_agent.agent import Decision, Observation, parse_decision


def test_plain_json_object():
    reply = (
        '{"scene":"一只猫在窗台上","people":[{"name":"小明","action":"坐着"}],'
        '"events":[{"kind":"presence","summary":"有人出现","importance":0.7}],'
        '"remember":[{"subject":"猫","predicate":"在","object":"窗台上"}],'
        '"notes":[{"name":"小明","text":"下午在看书"}]}'
    )
    decision = parse_decision(reply)
    assert decision.ok
    assert decision.scene == "一只猫在窗台上"
    assert decision.people[0]["name"] == "小明"
    assert decision.remember[0]["object"] == "窗台上"
    assert decision.notes[0]["text"] == "下午在看书"
    assert decision.importance == 0.7


def test_fenced_code_block():
    decision = parse_decision('```json\n{"scene":"fenced"}\n```')
    assert decision.ok
    assert decision.scene == "fenced"


def test_prose_wrapped_json():
    """A 3B model will sometimes wrap the object in a sentence; the parser still finds it."""
    decision = parse_decision('好的，结果如下：{"scene":"prose"} 以上。')
    assert decision.ok
    assert decision.scene == "prose"


def test_braces_inside_a_string_do_not_break_balance():
    decision = parse_decision('{"scene":"a {weird} scene","people":[]}')
    assert decision.ok
    assert decision.scene == "a {weird} scene"


def test_invalid_json_degrades_without_raising():
    decision = parse_decision('{"scene": }')
    assert not decision.ok
    assert "invalid JSON" in decision.error
    # The whole reply is kept as the scene so the tick still carries information.
    assert decision.scene == '{"scene": }'


def test_no_json_at_all_degrades():
    decision = parse_decision("just prose, no object")
    assert not decision.ok
    assert "no JSON object" in decision.error
    assert decision.scene == "just prose, no object"


def test_missing_keys_default_to_empty_lists():
    decision = parse_decision('{"scene":"only scene"}')
    assert decision.ok
    assert decision.people == []
    assert decision.events == []
    assert decision.remember == []
    assert decision.notes == []


def test_description_alias_is_accepted():
    decision = parse_decision('{"description":"aliased"}')
    assert decision.ok
    assert decision.scene == "aliased"


def test_a_dict_where_a_list_is_expected_is_wrapped():
    decision = parse_decision('{"scene":"x","people":{"name":"a"}}')
    assert decision.people == [{"name": "a"}]


def test_non_dict_entries_in_lists_are_dropped():
    decision = parse_decision('{"scene":"x","people":["nope",{"name":"a"}]}')
    assert decision.people == [{"name": "a"}]


def test_importance_is_the_max_of_events():
    decision = parse_decision('{"scene":"x","events":[{"importance":0.2},{"importance":0.9}]}')
    assert decision.importance == 0.9


def test_importance_zero_without_events():
    assert parse_decision('{"scene":"x"}').importance == 0.0


def test_empty_reply_is_not_ok():
    decision = parse_decision("")
    assert not decision.ok
    assert decision.scene == ""


def test_decision_as_dict_round_trips_keys():
    decision = Decision(scene="s", events=[{"importance": 0.4}])
    payload = decision.as_dict()
    assert payload["scene"] == "s"
    assert payload["ok"] is True
    assert payload["error"] == ""


def test_observation_text_reports_faces_speech_and_note():
    observation = Observation(
        ts=0.0,
        faces=[{"name": "小明", "known": True, "score": 0.8},
               {"name": "unknown", "known": False, "score": 0.0}],
        speech=[{"text": "你好", "language": "zh", "speaker_name": "小明"}],
        note="备注A",
    )
    text = observation.as_text()
    assert "人脸：" in text
    assert "小明(已知" in text
    assert "unknown(未知" in text
    assert "语音[zh] 小明: 你好" in text
    assert "备注：备注A" in text


def test_observation_text_handles_absent_modalities():
    text = Observation(ts=0.0).as_text()
    assert "人脸：无" in text
    assert "语音：无" in text
