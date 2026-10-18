import json

import pytest

from jev_client import (
    DEFAULT_JEV_MODEL,
    DOCUMENT_QUESTIONS,
    JevClient,
    JevConfig,
)


def _document_response():
    return {
        "model": DEFAULT_JEV_MODEL,
        "answers": {
            "policy_stance": {
                "type": "choice",
                "choice": "easing",
                "confidence": 0.8,
                "probabilities": {
                    "easing": 0.7,
                    "neutral": 0.1,
                    "tightening": 0.1,
                    "unclear": 0.1,
                },
            },
            "bond_impact": {
                "type": "choice",
                "choice": "bullish",
                "confidence": 0.7,
                "probabilities": {
                    "bullish": 0.6,
                    "neutral": 0.2,
                    "bearish": 0.1,
                    "unclear": 0.1,
                },
            },
            "impact_strength": {
                "type": "score",
                "score": 3.0,
                "confidence": 0.75,
                "probabilities": {
                    "0": 0.0,
                    "1": 0.1,
                    "2": 0.2,
                    "3": 0.5,
                    "4": 0.2,
                },
            },
            "explicit_new_information": {"type": "noul", "noul": 0.9},
            "liquidity_channel": {"type": "noul", "noul": 0.85},
            "growth_channel": {"type": "noul", "noul": 0.2},
            "inflation_channel": {"type": "noul", "noul": 0.1},
            "supply_channel": {"type": "noul", "noul": 0.05},
        },
        "usage": {"input_tokens": 500, "output_tokens": 50},
    }


def _gate_response():
    return {
        "model": DEFAULT_JEV_MODEL,
        "answers": {
            "action": {
                "type": "choice",
                "choice": "reduce",
                "confidence": 0.7,
                "probabilities": {
                    "maintain": 0.2,
                    "reduce": 0.7,
                    "abstain": 0.1,
                },
            },
            "evidence_sufficient": {"type": "noul", "noul": 0.65},
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def test_document_analysis_uses_pinned_model_and_content_cache(tmp_path):
    calls = []

    def transport(payload, api_key, timeout_seconds):
        calls.append((payload, api_key, timeout_seconds))
        return _document_response()

    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="test-key",
        transport=transport,
    )

    first = client.analyze_document("央行宣布降准，释放长期流动性。")
    second = client.analyze_document("央行宣布降准，释放长期流动性。")

    assert len(calls) == 1
    assert calls[0][0]["model"] == DEFAULT_JEV_MODEL
    assert "unclear" in calls[0][0]["questions"]["policy_stance"]["criteria"]
    assert first == second
    assert first["JEV_Policy_Direction"] == pytest.approx(0.6)
    assert first["JEV_Bond_Direction"] == pytest.approx(0.5)
    assert first["JEV_New_Information_P"] == pytest.approx(0.9)
    assert first["JEV_Impact_Level"] == pytest.approx(3.0)

    cache_files = list((tmp_path / "data" / "rag" / "jev_cache").glob("*.json"))
    assert len(cache_files) == 1
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert cached["requested_model"] == DEFAULT_JEV_MODEL
    assert "央行" not in cache_files[0].read_text(encoding="utf-8")

    cache_only = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=False, use_cache=True),
        transport=lambda *args: (_ for _ in ()).throw(
            AssertionError("cache-only replay must not call transport")
        ),
    )
    assert cache_only.analyze_document("央行宣布降准，释放长期流动性。") == first


def test_cache_key_is_bound_to_question_schema_version(tmp_path):
    calls = []

    def transport(*args):
        calls.append(args)
        return _document_response()

    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="test-key",
        transport=transport,
    )
    state = {"document": "央行公告"}

    client._query(
        state=state,
        questions=DOCUMENT_QUESTIONS,
        schema_version="macro-document-v1",
    )
    client._query(
        state=state,
        questions=DOCUMENT_QUESTIONS,
        schema_version="macro-document-v2",
    )

    assert len(calls) == 2
    assert len(list(client.cache_dir.glob("*.json"))) == 2


def test_disabled_client_is_cache_only_and_never_calls_transport(tmp_path):
    def unexpected_transport(*args, **kwargs):
        raise AssertionError("disabled Jev client must not make a network call")

    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=False),
        api_key="test-key",
        transport=unexpected_transport,
    )

    assert client.analyze_document("降准") is None


def test_invalid_response_fails_closed_without_writing_cache(tmp_path):
    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="test-key",
        transport=lambda *args: {"model": DEFAULT_JEV_MODEL, "answers": {}},
    )

    assert client.analyze_document("降准") is None
    assert "missing answers" in client.last_error
    assert not list((tmp_path / "data" / "rag" / "jev_cache").glob("*.json"))


def test_alias_model_is_rejected_for_reproducibility():
    with pytest.raises(ValueError, match="versioned Jev model"):
        JevConfig(model="jev-latest")


def test_signal_gate_returns_typed_shadow_decision(tmp_path):
    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="test-key",
        transport=lambda *args: _gate_response(),
    )

    result = client.assess_signal_gate({"existing_signal": {"direction": "long"}})

    assert result == {
        "action": "reduce",
        "confidence": 0.7,
        "probabilities": {"maintain": 0.2, "reduce": 0.7, "abstain": 0.1},
        "evidence_sufficient": 0.65,
        "model": DEFAULT_JEV_MODEL,
    }


def test_online_mode_without_api_key_fails_closed(tmp_path):
    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="",
        transport=lambda *args: _document_response(),
    )

    assert client.analyze_document("央行公告") is None
    assert client.last_error == "TYPESAFE_API_KEY is not configured"


def test_long_document_is_bounded_before_transport(tmp_path):
    payloads = []

    def transport(payload, *_):
        payloads.append(payload)
        return _document_response()

    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True, max_state_chars=1_000),
        api_key="test-key",
        transport=transport,
    )

    assert client.analyze_document("政策" * 2_000) is not None
    sent = payloads[0]["state"]["document"]
    assert "document truncated by client" in sent
    assert len(sent) < 1_100


def test_invalid_probability_distribution_is_not_cached(tmp_path):
    response = _document_response()
    response["answers"]["policy_stance"]["probabilities"]["easing"] = 0.95
    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="test-key",
        transport=lambda *args: response,
    )

    assert client.analyze_document("央行公告") is None
    assert "sum to one" in client.last_error


def test_environment_mode_and_dotenv_key_are_loaded(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=dotenv-test-key\n", encoding="utf-8"
    )
    monkeypatch.setenv("F_AGENT_JEV_MODE", "online")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    client = JevClient(base_dir=tmp_path, transport=lambda *args: _document_response())

    assert client.config.enabled is True
    assert client.config.use_cache is True
    assert client.api_key == "dotenv-test-key"


def test_builtin_http_transport_sends_bearer_json(tmp_path, monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            captured["read_limit"] = limit
            return json.dumps(_document_response()).encode("utf-8")

    class _Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response()

    def fake_build_opener(handler):
        captured["redirect_handler"] = handler
        return _Opener()

    monkeypatch.setattr("jev_client.build_opener", fake_build_opener)
    client = JevClient(
        base_dir=tmp_path,
        config=JevConfig(enabled=True),
        api_key="secret-test-key",
    )

    response = client._http_transport(
        {"model": DEFAULT_JEV_MODEL, "state": "x", "questions": {}},
        "secret-test-key",
        12.0,
    )

    assert response["model"] == DEFAULT_JEV_MODEL
    assert captured["timeout"] == 12.0
    assert captured["read_limit"] == 128001
    assert captured["redirect_handler"].redirect_request(
        None, None, 302, "", {}, "https://unapproved.example"
    ) is None
    assert captured["request"].get_header("Authorization") == "Bearer secret-test-key"
    assert captured["request"].get_header("Content-type") == "application/json"


@pytest.mark.parametrize("field", ["confidence", "noul", "probability", "score", "score_confidence", "score_probability"])
@pytest.mark.parametrize("invalid", [True, False, "0.7"])
def test_real_jev_response_rejects_coercible_non_numeric_values(tmp_path, field, invalid):
    response = _document_response() if field.startswith("score") else _gate_response()
    answers = response["answers"]
    if field == "confidence":
        answers["action"]["confidence"] = invalid
    elif field == "noul":
        answers["evidence_sufficient"]["noul"] = invalid
    elif field == "probability":
        answers["action"]["probabilities"] = {
            "maintain": invalid, "reduce": 1 - float(invalid), "abstain": 0.0,
        }
    elif field == "score":
        answers["impact_strength"]["score"] = invalid
    elif field == "score_confidence":
        answers["impact_strength"]["confidence"] = invalid
    else:
        answers["impact_strength"]["probabilities"] = {
            "0": invalid, "1": 1 - float(invalid), "2": 0.0, "3": 0.0, "4": 0.0,
        }
    client = JevClient(
        tmp_path, config=JevConfig(enabled=True), api_key="test-key",
        transport=lambda *_: response,
    )
    result = client.analyze_document("央行公告") if field.startswith("score") else client.assess_signal_gate({"existing_signal": "long"})
    assert result is None
    assert "numeric" in client.last_error
    assert not list(client.cache_dir.glob("*.json"))
