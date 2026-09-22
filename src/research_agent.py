# -*- coding: utf-8 -*-
"""Auditable TL Research Agent orchestration.

The deterministic path is always factors -> ModelRuntime -> SignalPolicy.
Sentiment/Jev/RAG/LLM are research evidence and can fail independently; they do
not acquire authority to mutate the base decision or create an order.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from intelligence_events import redact_sensitive_text
from strategy_runtime import SignalPolicy, get_strategy_spec


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _safe_error(value):
    return redact_sensitive_text(value)[:1000]


class ResearchAgent:
    """Coordinate one TL research cycle through typed, injectable boundaries."""

    def __init__(
        self,
        base_dir,
        *,
        runtime=None,
        policy=None,
        strategy_id=None,
        monitor=None,
        fusion_runner=None,
        memory_factory=None,
        api_key=None,
    ):
        self.base_dir = os.fspath(base_dir)
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.strategy_spec = get_strategy_spec(strategy_id)
        self.policy = policy or SignalPolicy(self.strategy_spec)
        self.runtime = runtime
        self.monitor = monitor
        self.fusion_runner = fusion_runner
        self.memory_factory = memory_factory
        self.api_key = api_key

    def run_cycle(
        self,
        rebuild_factors=True,
        use_llm=True,
        *,
        monitor_sentiment=True,
        persist=True,
        allow_stale_factors=False,
    ):
        now = datetime.now(timezone.utc)
        run_id = now.strftime("research_%Y%m%dT%H%M%S_%fZ")
        report = {
            "run_id": run_id,
            "timestamp": now.isoformat(),
            "step_status": {},
            "strategy": {
                "strategy_id": self.policy.spec.strategy_id,
                "strategy_version": self.policy.spec.version,
                "strategy_fingerprint": self.policy.spec.fingerprint(),
            },
            "sentiment": None,
            "base_signal": None,
            "fused_signal": None,
            "direction_changed": False,
            "research_proposal_changed": False,
        }

        if monitor_sentiment and (persist or self.monitor is not None):
            self._run_sentiment_monitor(report)
        elif monitor_sentiment:
            report["step_status"]["sentiment_monitor"] = (
                "skipped:persist=False-no-injected-monitor"
            )
        else:
            report["step_status"]["sentiment_monitor"] = (
                "skipped:monitor_sentiment=False"
            )

        factor_file = os.path.join(self.output_dir, "df_factors.pkl")
        factor_failed = False
        if rebuild_factors:
            try:
                import factor_extraction

                ok = factor_extraction.run_process(
                    self.base_dir, tick_subdir="data/tick"
                )
                factor_failed = not bool(ok)
                report["step_status"]["factors"] = (
                    "ok" if ok else "error:run_process returned False"
                )
            except Exception as exc:
                factor_failed = True
                report["step_status"]["factors"] = f"error:{_safe_error(exc)}"
        else:
            report["step_status"]["factors"] = "skipped:rebuild_factors=False"

        if not os.path.exists(factor_file):
            report["step_status"]["factors"] = "error:df_factors.pkl 不存在"
            self._finish(report, persist=persist)
            return report
        if factor_failed and not allow_stale_factors:
            report["step_status"]["factors"] += ":stale_fallback_blocked"
            self._finish(report, persist=persist)
            return report
        if factor_failed:
            report["step_status"]["factors"] += ":using_explicit_stale_fallback"

        base_signal = None
        try:
            factors = pd.read_pickle(factor_file)
            if self.runtime is not None:
                predictions = self.runtime.predict_frame(factors)
                release_id = str(
                    self.runtime.metadata.get(
                        "model_release_id", "legacy-unversioned"
                    )
                )
                base_signal = self.policy.latest(
                    predictions, model_release_id=release_id
                ).to_dict()
            else:
                from inference import SignalGenerator

                model_path = os.path.join(
                    self.base_dir, "models", "trained_model.pkl"
                )
                generator = SignalGenerator(
                    model_path, strategy_id=self.policy.spec.strategy_id
                )
                base_signal, _ = generator.generate_signal(factors)
            report["base_signal"] = base_signal
            if not base_signal.get("decision_ready", True):
                report["step_status"]["signal"] = (
                    "skipped:" + base_signal.get(
                        "missing_reason", "decision_not_ready"
                    )
                )
                self._finish(report, persist=persist)
                return report
            report["step_status"]["signal"] = "ok"
        except Exception as exc:
            report["step_status"]["signal"] = f"error:{_safe_error(exc)}"

        if base_signal is None:
            self._finish(report, persist=persist)
            return report

        fused = None
        try:
            if self.fusion_runner is None:
                from signal_fusion import run_fusion

                fusion_runner = run_fusion
            else:
                fusion_runner = self.fusion_runner
            fusion_kwargs = dict(
                base_dir=self.base_dir,
                base_signal=base_signal,
                use_llm=use_llm,
                allow_trade_adjustment=False,
                persist=persist,
            )
            if self.api_key is not None:
                fusion_kwargs["api_key"] = self.api_key
            fused = fusion_runner(**fusion_kwargs)
            if fused is None:
                report["step_status"]["fusion"] = (
                    "skipped:run_fusion returned None"
                )
            else:
                fused_dict = _jsonable(fused)
                report["fused_signal"] = fused_dict
                report["direction_changed"] = (
                    fused.adjusted_direction != fused.raw_direction
                )
                report["research_proposal_changed"] = (
                    fused.proposed_direction != fused.raw_direction
                )
                report["step_status"]["fusion"] = (
                    f"ok:level={fused.fusion_level}:shadow={fused.shadow_only}"
                )
        except Exception as exc:
            report["step_status"]["fusion"] = f"error:{_safe_error(exc)}"

        if persist:
            self._record_memory(report, base_signal, fused)
        else:
            report["step_status"]["record"] = "skipped:persist=False"
            report["step_status"]["backfill"] = "skipped:persist=False"

        self._finish(report, persist=persist)
        return report

    def _run_sentiment_monitor(self, report):
        try:
            if self.monitor is None:
                from jev_client import JevClient
                from sentiment_monitor import SentimentMonitor

                self.monitor = SentimentMonitor(
                    self.base_dir, jev_adapter=JevClient(self.base_dir)
                )
            run = self.monitor.run_once()
            store = self.monitor.store
            assessments = store.list_assessments()[-20:]
            report["sentiment"] = {
                "run": _jsonable(run),
                "source_health": [
                    _jsonable(item) for item in store.list_source_health()
                ],
                "open_alerts": [
                    _jsonable(item)
                    for item in store.list_alerts()
                    if getattr(item, "status", "") == "open"
                ],
                "recent_assessments": [
                    _jsonable(item) for item in assessments
                ],
            }
            status = str(getattr(run, "status", "failed"))
            prefix = {
                "ok": "ok",
                "partial_failure": "warning",
                "failed": "error",
            }.get(status, "warning")
            report["step_status"]["sentiment_monitor"] = (
                f"{prefix}:status={status}"
            )
        except Exception as exc:
            report["step_status"]["sentiment_monitor"] = (
                f"error:{_safe_error(exc)}"
            )

    def _record_memory(self, report, base_signal, fused):
        try:
            if self.memory_factory is None:
                from memory import TradingMemory

                memory = TradingMemory(
                    self.base_dir, strategy_id=self.policy.spec.strategy_id
                )
            else:
                memory = self.memory_factory(self.base_dir)
            record = dict(base_signal)
            if fused is not None:
                record.update(
                    {
                        "fused_direction": fused.adjusted_direction,
                        "fused_weight": fused.adjusted_weight,
                        "research_proposed_direction": fused.proposed_direction,
                        "research_proposed_weight": fused.proposed_weight,
                        "fusion_level": fused.fusion_level,
                        "fusion_reasons": list(fused.reasons)[:10],
                    }
                )
            memory.record_signal(record)
            report["step_status"]["record"] = "ok"
            memory.update_actuals()
            report["step_status"]["backfill"] = "ok"
        except Exception as exc:
            report["step_status"]["record"] = f"error:{_safe_error(exc)}"
            report["step_status"].setdefault("backfill", "skipped:record_failed")

    def _finish(self, report, *, persist):
        if persist:
            self._save_report(report)
        self._print_summary(report)

    def _save_report(self, report):
        os.makedirs(self.output_dir, exist_ok=True)
        path = Path(self.output_dir) / f"research_cycle_{report['run_id']}.json"
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".research_cycle.", suffix=".tmp", dir=self.output_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
            os.replace(temporary_path, path)
            report["_report_path"] = str(path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _print_summary(report):
        print("\n" + "=" * 56)
        print("  Research Agent 闭环报告")
        print("=" * 56)
        for step, status in report["step_status"].items():
            mark = "OK " if status.startswith("ok") else (
                "-- " if status.startswith("skipped") else "!! "
            )
            print(f"  [{mark}] {step:18s} {status}")
        base = report.get("base_signal") or {}
        if base:
            direction = {1: "做多", -1: "做空", 0: "观望"}.get(
                base.get("direction"), "?"
            )
            print(
                f"  决策: {direction}; strategy={base.get('strategy_id')}; "
                f"release={base.get('model_release_id')}"
            )
        print("=" * 56)


def run_research(
    base_dir,
    rebuild_factors=True,
    use_llm=True,
    *,
    strategy_id=None,
    monitor_sentiment=True,
    persist=True,
    api_key=None,
):
    agent = ResearchAgent(
        base_dir, strategy_id=strategy_id, api_key=api_key
    )
    return agent.run_cycle(
        rebuild_factors=rebuild_factors,
        use_llm=use_llm,
        monitor_sentiment=monitor_sentiment,
        persist=persist,
    )
