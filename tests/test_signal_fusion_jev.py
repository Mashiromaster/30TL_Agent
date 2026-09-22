from signal_fusion import SignalFusionEngine


class _FakeJevGate:
    def assess_signal_gate(self, state):
        return {
            "action": "abstain",
            "confidence": 0.95,
            "probabilities": {
                "maintain": 0.02,
                "reduce": 0.03,
                "abstain": 0.95,
            },
            "model": "jev-1.13.0",
        }


def test_jev_shadow_gate_is_audited_but_never_changes_signal(tmp_path):
    engine = SignalFusionEngine(
        base_dir=tmp_path,
        jev_client=_FakeJevGate(),
        jev_gate_mode="shadow",
    )
    base_signal = {
        "direction": 1,
        "confidence": 0.6,
        "suggested_weight": 0.6,
        "regime_name": "正常",
        "predicted_return": 0.01,
        "predicted_return_smooth": 0.008,
        "timestamp": "2025-01-02 10:00:00",
        "strategy_id": "tl_regime_breakout_v1",
        "strategy_fingerprint": "sha256:test",
        "model_release_id": "release-test",
        "decision_id": "decision:test",
        "decision_ready": True,
    }

    fused = engine.fuse(base_signal, use_llm=False)

    assert fused.adjusted_direction == 1
    assert fused.adjusted_weight == 0.6
    assert fused.jev_gate_action == "abstain"
    assert fused.jev_gate_confidence == 0.95
    assert fused.jev_model == "jev-1.13.0"
    assert fused.fusion_level == "none"
