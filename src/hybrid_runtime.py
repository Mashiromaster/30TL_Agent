"""Code-owned, bounded multi-model research workflow; deliberately no broker API.

Model probabilities describe different questions. They are never averaged into
an invented return probability. The canonical SignalPolicy result is immutable;
all reductions and abstentions below are *research proposals*, not order intents.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence_events import IntelligenceEventStore, redact_sensitive_text
from strategy_runtime import SignalPolicy, get_strategy_spec

SCHEMA_VERSION = "f_agent.hybrid.v1"
_RUN_ID = re.compile(r"hybrid_\d{8}T\d{6}_\d{6}Z_[a-f0-9]{12}")
_PRIVATE_KEYS = {"api_key", "authorization", "password", "secret", "token", "reasoning_content", "chain_of_thought", "thinking"}
_STATUSES = {"ok", "skipped", "unavailable", "error", "blocked"}
_SEVERITY = {"maintain": 0, "reduce": 1, "abstain": 2}


def _public(value):
    """JSON-safe public audit data, excluding secrets and hidden reasoning."""
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items() if str(k).lower() not in _PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", redact_sensitive_text(value))
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _public(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _public(str(value))


def _utc(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("missing timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Shanghai")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class HybridConfig:
    network_mode: str = "offline"
    asset: str = "TL"
    max_age_hours: float = 24.0
    max_model_calls: int = 4
    max_evidence: int = 20

    def __post_init__(self):
        if self.network_mode not in {"offline", "cache", "online"}:
            raise ValueError("network_mode must be offline, cache, or online")
        if self.asset != "TL":
            raise ValueError("this version supports the TL strategy only; do not reuse TL factors for other assets")
        if not math.isfinite(self.max_age_hours) or self.max_age_hours <= 0:
            raise ValueError("max_age_hours must be finite and positive")
        for name, lower, upper in (("max_model_calls", 0, 4), ("max_evidence", 1, 20)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def _reply(output=None, *, status="ok", reason="", model="", source="local"):
    return {"status": status, "reason": reason, "model": model, "source": source,
            "output": output or {}, "usage": {}}


def _snapshot(factors, as_of, now, config):
    if factors is None or not isinstance(factors, pd.DataFrame) or factors.empty:
        raise ValueError("没有可用因子数据 / no factor snapshot")
    required = {"date", "ticker", "close"}
    if not required.issubset(factors):
        raise ValueError(f"missing factor snapshot columns: {sorted(required.difference(factors))}")
    if not factors.columns.is_unique:
        raise ValueError("duplicate factor columns")
    # Row labels are not market identity. Normalizing our private copy avoids
    # duplicate-index expansion while preserving the caller's frame untouched.
    factors = factors.reset_index(drop=True)
    dates = factors["date"].map(_utc)
    cutoff = _utc(as_of) if as_of is not None else now
    if cutoff > now:
        raise ValueError("as_of cannot be in the future")
    frame = factors.loc[dates <= cutoff].copy()
    if frame.empty:
        raise ValueError("no factor rows available at requested as_of")
    # Canonical model runtime uses exchange-local naive timestamps. Normalize a
    # mixed-offset input before sorting; never re-label UTC as exchange time.
    frame["date"] = dates.loc[frame.index].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate timestamp/contract observations")
    if not frame["ticker"].astype(str).str.fullmatch(r"TL(?:\d{3,4}|0)", case=False).all():
        raise ValueError("factor snapshot contains non-TL or missing contract identities")
    prices = pd.to_numeric(frame["close"], errors="coerce")
    if not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError("snapshot prices must be finite and positive")
    market_as_of = _utc(frame["date"].iloc[-1])
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(str, frame.columns))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    age = max(0.0, (now - market_as_of).total_seconds())
    snapshot = {"snapshot_id": "snapshot:" + digest.hexdigest(), "as_of": market_as_of.isoformat(),
                "requested_as_of": cutoff.isoformat(), "rows": len(frame), "source_rows": len(factors),
                "excluded_future_rows": len(factors) - len(frame), "age_seconds": age,
                "stale": age > config.max_age_hours * 3600, "max_age_hours": config.max_age_hours,
                "first_bar": _utc(frame["date"].iloc[0]).isoformat(), "contract": str(frame["ticker"].iloc[-1]),
                "feature_count": len(frame.columns), "historical_research": as_of is not None or age > config.max_age_hours * 3600}
    return frame, snapshot


def _load_evidence(base_dir, *, as_of, asset, limit):
    path = Path(base_dir) / "data" / "intelligence" / "events.sqlite"
    if not path.exists():
        return []
    store = IntelligenceEventStore(base_dir, read_only=True)
    records = []
    for assessment in store.query_assessments(as_of=as_of, assets=[asset], limit=limit):
        document = store.get_document(assessment.document_id)
        if document is None:
            continue
        records.append({
            "id": assessment.assessment_id, "document_id": document.document_id,
            "title": document.title, "source": document.source_name, "url": document.url,
            "published_at": document.published_at, "available_at": document.available_at,
            "assessed_at": assessment.assessed_at, "expires_at": assessment.expires_at,
            "affected_assets": assessment.affected_assets, "bond_impact": assessment.bond_impact,
            "severity": assessment.severity, "confidence": assessment.confidence,
            "summary": "；".join(assessment.evidence)[:1800], "evidence_quotes": assessment.evidence[:5],
            "prompt_injection_flag": assessment.prompt_injection_flag,
            "assessment_model": assessment.model_version,
        })
    return records


def _filter_evidence(records, *, as_of, config):
    selected, excluded, seen = [], Counter(), set()
    cutoff = _utc(as_of)
    for record in records:
        reason = ""
        if not isinstance(record, dict) or not record.get("id"):
            excluded["invalid_record"] += 1
            continue
        if record.get("prompt_injection_flag"):
            reason = "quarantined"
        elif config.asset not in record.get("affected_assets", []):
            reason = "other_asset"
        else:
            try:
                for key in ("published_at", "available_at", "assessed_at"):
                    if _utc(record.get(key)) > cutoff:
                        reason = "future_evidence"
                if record.get("expires_at") and _utc(record["expires_at"]) <= cutoff:
                    reason = "expired"
            except (ValueError, TypeError, OverflowError):
                reason = "unknown_time"
        if not reason and record["id"] in seen:
            reason = "duplicate"
        if reason:
            excluded[reason] += 1
            continue
        seen.add(record["id"])
        if len(selected) >= config.max_evidence:
            excluded["limit"] += 1
            continue
        # Keep only known public fields; raw_response or full content may contain
        # unbounded external instructions. Model adapters additionally validate.
        allowed = {"id", "document_id", "title", "source", "url", "published_at", "available_at", "assessed_at",
                   "expires_at", "affected_assets", "bond_impact", "severity", "confidence", "summary",
                   "evidence_quotes", "assessment_model"}
        item = {k: v for k, v in record.items() if k in allowed}
        for key in ("summary", "title", "source"):
            item[key] = str(item.get(key, ""))[:1800]
        selected.append(_public(item))
    return selected, dict(excluded)


def _quant(base_dir, frame, runtime, policy):
    if runtime is None:
        from research_readiness import resolve_managed_runtime
        runtime, signature = resolve_managed_runtime(base_dir, strategy_id=getattr(getattr(policy, "spec", None), "strategy_id", None))
        release_id = signature.release_id
    else:
        release_id = str(runtime.metadata.get("model_release_id", "unversioned"))
    predictions = runtime.predict_frame(frame)
    baseline = policy.latest(predictions, model_release_id=release_id).to_dict()
    if baseline.get("direction") not in {-1, 0, 1}:
        raise ValueError("invalid canonical direction")
    weight = float(baseline.get("suggested_weight", float("nan")))
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("invalid canonical weight")
    if _utc(baseline["timestamp"]) != _utc(frame["date"].iloc[-1]):
        raise ValueError("canonical decision timestamp does not match snapshot")
    metadata = dict(runtime.metadata)
    components = {}
    # Flat models have no such decomposition; never synthesize one for the UI.
    if metadata.get("selected_model_kind") == "hierarchical":
        components = {key: float(predictions.iloc[-1][key]) for key in
                      ("micro_alpha", "macro_state", "gate_weight", "event_alpha")}
        components["formula"] = "prediction = micro_alpha * gate_weight + macro_state; event_alpha is diagnostic only"
    return _reply({"baseline": baseline, "metadata": metadata, "components": components},
                  model=release_id, source="local")


def _validate_expert(reply, stage_id, evidence_ids):
    if not isinstance(reply, dict) or reply.get("status") not in _STATUSES:
        raise ValueError("invalid expert envelope")
    if reply["status"] != "ok":
        # A failed role may supply only its status. Keep a complete envelope so
        # degradation itself cannot crash orchestration or leave a partial run.
        return _public(_reply(status=reply["status"], reason=str(reply.get("reason") or "expert_unavailable"),
                              model=str(reply.get("model") or ""), source=str(reply.get("source") or "unknown")))
    output = reply.get("output")
    if not isinstance(output, dict):
        raise ValueError("expert output must be an object")
    confidence = output.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid expert confidence")
    if stage_id == "router":
        if output.get("route") not in {"quant_only", "event_review", "deep_review", "abstain"}:
            raise ValueError("unknown route")
    elif stage_id == "jev":
        if output.get("action") not in _SEVERITY:
            raise ValueError("unknown Jev action")
        sufficient = output.get("evidence_sufficient")
        if isinstance(sufficient, bool) or not isinstance(sufficient, (int, float)) or not math.isfinite(sufficient) or not 0 <= sufficient <= 1:
            raise ValueError("invalid evidence sufficiency")
    else:
        allowed = {"recommendation", "confidence", "summary", "evidence_ids", "counter_evidence_ids", "risks", "limitations"}
        if set(output).difference(allowed) or output.get("recommendation") not in _SEVERITY:
            raise ValueError("unexpected LLM action or field")
        refs = output.get("evidence_ids", [])
        counters = output.get("counter_evidence_ids", [])
        if not isinstance(refs, list) or not isinstance(counters, list) or not set(refs + counters).issubset(evidence_ids):
            raise ValueError("unknown evidence citation")
        if not refs and not counters:
            raise ValueError("review must cite contemporaneous evidence")
    return _public(reply)


class _Run:
    def __init__(self, base_dir, config, on_event):
        self.base_dir, self.config, self.on_event = base_dir, config, on_event
        self.stages = []
        self.calls = 0

    @staticmethod
    def execute(stage_id, label, dependencies, operation, *, kind="deterministic"):
        started = datetime.now(timezone.utc).isoformat()
        timer = time.perf_counter()
        try:
            result = operation()
        except Exception as exc:
            result = _reply(status="error", reason=str(exc))
        result = _public(result)
        stage = {"stage_id": stage_id, "label": label, "kind": kind, "dependencies": dependencies,
                 "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_ms": round((time.perf_counter() - timer) * 1000, 3),
                 "input_refs": dependencies, **result}
        return stage, result

    def collect(self, executed):
        stage, result = executed
        self.stages.append(stage)
        if self.on_event is not None:
            try:
                self.on_event(copy.deepcopy(stage))
            except Exception:
                # Rendering or notification observers have no decision authority.
                pass
        return result

    def stage(self, *args, **kwargs):
        return self.collect(self.execute(*args, **kwargs))

    def expert(self, stage_id, label, dependencies, operation, state, *, skip=""):
        def invoke():
            if skip:
                return _reply(status="skipped", reason=skip)
            if self.config.network_mode == "offline":
                return _reply(status="skipped", reason="offline: external model calls disabled", source="offline")
            if self.calls >= self.config.max_model_calls:
                return _reply(status="skipped", reason="model_call_budget_exhausted")
            self.calls += 1
            return _validate_expert(operation(copy.deepcopy(state)), stage_id, {item["id"] for item in state["evidence"]})
        return self.stage(stage_id, label, dependencies, invoke, kind="model")


def _arbitrate(baseline, snapshot, opinions, evidence, route):
    blocked, reasons, conflicts = [], [], []
    if baseline is None:
        blocked.append("canonical_model_unavailable")
    elif not baseline.get("decision_ready", False):
        blocked.append("canonical_not_ready:" + str(baseline.get("missing_reason", "unknown")))
    if not snapshot:
        blocked.append("invalid_snapshot")
    elif snapshot["stale"]:
        blocked.append("stale_market_snapshot")
    if snapshot.get("factor_provenance", {}).get("verified") is False:
        blocked.append("unverified_factor_provenance")
    actions = {}
    for name in ("jev", "llm", "advisor"):
        opinion = opinions.get(name, {})
        if opinion.get("status") != "ok":
            reasons.append(f"{name}: {opinion.get('status', 'unavailable')} ({opinion.get('reason', '')})")
            continue
        output = opinion["output"]
        action = output.get("action", output.get("recommendation"))
        if output["confidence"] < .6 or (name == "jev" and output["evidence_sufficient"] < .6):
            action = "abstain"
            reasons.append(f"{name}: uncertain evidence; human review required")
        actions[name] = action
    if len(set(actions.values())) > 1:
        conflicts.append("experts_disagree:" + ",".join(f"{k}={v}" for k, v in actions.items()))
    if any(item.get("bond_impact") == "bullish" for item in evidence) and any(item.get("bond_impact") == "bearish" for item in evidence):
        conflicts.append("contemporaneous_bond_evidence_conflicts")
        actions["evidence_guard"] = "abstain"
    if route.get("route") == "abstain":
        actions["router"] = "abstain"
    action = max(actions.values(), key=_SEVERITY.get) if actions else "maintain"
    if not evidence:
        reasons.append("no_contemporaneous_evidence")
        action = "abstain"
    if blocked:
        action = "abstain"
    direction = (baseline or {}).get("direction", 0)
    weight = float((baseline or {}).get("suggested_weight", 0))
    if action == "abstain" or direction == 0:
        direction, weight = 0, 0.0
    elif action == "reduce":
        weight *= .5
    if baseline and (direction != baseline["direction"] or weight != baseline["suggested_weight"]):
        conflicts.append("research_proposal_differs_from_canonical_strategy")
    incomplete = any(opinions.get(name, {}).get("status") != "ok" for name in ("jev", "llm", "advisor"))
    status = "blocked" if blocked else "review_required" if incomplete or action == "abstain" else "research_only"
    return {"status": status, "recommendation": action, "proposed_direction": direction, "proposed_weight": weight,
            "execution_authorized": False, "baseline_unchanged": True, "blocked_reasons": blocked,
            "conflicts": conflicts, "reasons": reasons,
            "arbitration_rule": "most_conservative_valid_opinion_v1; reduce=0.5; no reversal or increase",
            "disclaimer": "研究建议非订单；各模型置信度不等于收益概率，未进行概率加权。"}


def run_hybrid_research(base_dir, *, factors=None, question="宏观与微观证据是否支持当前TL策略？",
                        as_of=None, config=None, api_key=None, runtime=None, policy=None, adapters=None,
                        evidence_loader=None, persist=True, on_event=None, now=None, factor_path=None):
    """Run the audited research entry, with trusted Python integration seams.

    UI/CLI pass only disk paths and built-in adapters. Injected frames, models,
    adapters, evidence and clocks are for trusted tests/integrations, not a
    sandbox for arbitrary Python code or proof of production readiness. In
    particular, caller-supplied cache adapters must honor their own I/O contract.
    Every override is recorded so fixtures cannot masquerade as standard runs.
    """
    # ResearchAgent normally supplies its canonical SignalPolicy to select a
    # supported strategy. Only replacement implementations are test overrides.
    injected = [name for name, value in (("factors", factors), ("runtime", runtime),
                ("policy", policy if type(policy) is not SignalPolicy else None), ("adapters", adapters), ("evidence_loader", evidence_loader),
                ("clock", now)) if value is not None]
    config = config or HybridConfig()
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    question = _public(str(question).strip()[:2000])
    policy = policy or SignalPolicy(get_strategy_spec())
    run = _Run(base_dir, config, on_event)
    trace = {"schema_version": SCHEMA_VERSION, "run_id": now.strftime("hybrid_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:12],
             "generated_at": now.isoformat(), "asset": config.asset, "question": question,
             "integration": {"production_inputs": not injected, "injected_components": injected},
             "network_mode": config.network_mode, "config": asdict(config), "snapshot": {}, "baseline": None,
             "evidence": [], "evidence_exclusions": {}, "opinions": {}, "route": {},
             "persistence": {"status": "not_requested", "path": "", "reason": ""}}
    prepared = []

    def prepare():
        frame = factors
        source = {"source": "injected"}
        if frame is not None and factor_path is not None:
            raise ValueError("factors and factor_path are mutually exclusive")
        if frame is None:
            from research_readiness import load_factor_input, verify_factor_provenance
            frame, path, digest = load_factor_input(base_dir, factor_path)
            source = {"source": "local_artifact", "factor_path": str(path), "factor_sha256": digest,
                      "factor_provenance": verify_factor_provenance(base_dir, path, frame, expected_sha256=digest)}
        selected, snapshot = _snapshot(frame, as_of, now, config)
        snapshot.update(source)
        prepared.append(selected)
        trace["snapshot"] = snapshot
        if source.get("factor_provenance", {}).get("verified") is False:
            return _reply(snapshot, status="blocked", reason="unverified_factor_provenance")
        return _reply(snapshot)

    snapshot_stage = run.stage("snapshot", "时点快照 / 数据新鲜度", [], prepare)
    run.stage("advisor_plan", "Advisor · 计划约束检查（规则）", ["snapshot"], lambda: _reply({
        "network_mode": config.network_mode, "call_limit": config.max_model_calls,
        "asset": config.asset, "constraints": ["no_orders", "canonical_immutable", "as_of_evidence_only", "no_hidden_retries"],
        "model_roles": ["quantitative_worker", "evidence_explorer", "jev_router", "llm_researcher", "advisor"]}))

    def explorer():
        loader = evidence_loader or (lambda **kwargs: _load_evidence(base_dir, **kwargs))
        records = loader(as_of=trace["snapshot"]["as_of"], asset=config.asset, limit=config.max_evidence)
        selected, excluded = _filter_evidence(records, as_of=trace["snapshot"]["as_of"], config=config)
        return _reply({"evidence": selected, "excluded": excluded},
                      status="ok" if selected else "unavailable", reason="" if selected else "no_contemporaneous_evidence")

    if snapshot_stage["status"] == "ok":
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-local") as pool:
            quant_future = pool.submit(run.execute, "quant", "Worker · 已发布量化模型 / SignalPolicy", ["snapshot", "advisor_plan"],
                                       lambda: _quant(base_dir, prepared[0], runtime, policy), kind="local_model")
            evidence_future = pool.submit(run.execute, "evidence", "Explorer · 可用时点证据检索", ["snapshot", "advisor_plan"], explorer)
            quant = run.collect(quant_future.result())
            evidence_result = run.collect(evidence_future.result())
    else:
        quant = run.stage("quant", "Worker · 量化模型", ["snapshot"], lambda: _reply(status="skipped", reason="invalid_snapshot"))
        evidence_result = run.stage("evidence", "Explorer · 证据", ["snapshot"], lambda: _reply(status="skipped", reason="invalid_snapshot"))
    trace["opinions"]["quant"] = quant
    trace["baseline"] = copy.deepcopy(quant["output"].get("baseline"))
    trace["evidence"] = evidence_result["output"].get("evidence", [])
    trace["evidence_exclusions"] = evidence_result["output"].get("excluded", {})
    if adapters is None:
        from hybrid_models import HybridModelAdapters
        adapters = HybridModelAdapters(base_dir, network_mode=config.network_mode, api_key=api_key)
    state = _public({"question": question, "snapshot": trace["snapshot"], "baseline": trace["baseline"],
                     "evidence": trace["evidence"], "quant_components": quant["output"].get("components", {})})
    skip = "canonical_model_unavailable" if trace["baseline"] is None else "no_contemporaneous_evidence" if not trace["evidence"] else ""
    router = run.expert("router", "Jev · 研究深度路由", ["quant", "evidence"], adapters.route, state, skip=skip)
    route = router["output"] if router["status"] == "ok" else {"route": "event_review" if trace["evidence"] else "quant_only", "fallback": True, "reason": router["reason"]}
    if router["status"] == "ok" and route.get("confidence", 0) < .65:
        route = {**route, "route": "deep_review", "fallback": True, "reason": "low_confidence_route_escalated"}
    trace["route"] = route
    trace["opinions"]["jev"] = run.expert("jev", "Jev · 语义风险判断", ["router", "quant", "evidence"], adapters.semantic, state, skip=skip)
    state["jev"] = trace["opinions"]["jev"]
    llm_skip = skip or ("router_selected_quant_only" if route["route"] == "quant_only" else "")
    trace["opinions"]["llm"] = run.expert("llm", "Researcher · LLM 证据研究", ["jev", "evidence", "quant"],
                                             lambda s: adapters.review(s, role="researcher"), state, skip=llm_skip)
    state["researcher"] = trace["opinions"]["llm"]
    state["proposal"] = _arbitrate(trace["baseline"], trace["snapshot"], trace["opinions"], trace["evidence"], route)
    state["review_triggers"] = ["pre_delivery"] + (["expert_failure"] if any(x["status"] == "error" for x in run.stages) else [])
    trace["opinions"]["advisor"] = run.expert("advisor", "Advisor · 交付 / 失败复核", ["llm", "jev", "quant"],
                                                 lambda s: adapters.review(s, role="advisor"), state, skip=llm_skip)
    decision = _arbitrate(trace["baseline"], trace["snapshot"], trace["opinions"], trace["evidence"], route)
    run.stage("risk", "确定性边界 / 风险检查", ["snapshot", "quant", "advisor"], lambda: _reply({
        "blocked_reasons": decision["blocked_reasons"], "execution_authorized": False,
        "baseline_unchanged": True, "limits": ["no_reversal", "no_weight_increase", "no_order_api"]},
        status="blocked" if decision["blocked_reasons"] else "ok"))
    run.stage("arbitration", "保守仲裁 · 影子研究结论", ["risk", "jev", "llm", "advisor"], lambda: _reply(decision))
    trace.update(decision=decision, stages=run.stages, budget={"call_limit": config.max_model_calls, "attempted_calls": run.calls})
    trace = _public(trace)
    if persist:
        try:
            path = _run_path(base_dir, trace["run_id"])
            trace["persistence"] = {"status": "saved", "path": str(path), "reason": ""}
            save_hybrid_run(trace, base_dir)
        except Exception as exc:
            trace["persistence"] = {"status": "error", "path": "", "reason": _public(str(exc))}
    return trace


def _run_path(base_dir, run_id):
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("invalid hybrid run_id")
    base = Path(base_dir).resolve()
    path = base / "outputs" / "hybrid_runs" / (run_id + ".json")
    if not path.resolve().is_relative_to(base):
        raise ValueError("hybrid run path resolves outside project")
    return path


def save_hybrid_run(trace, base_dir):
    destination = _run_path(base_dir, trace["run_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".hybrid-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_public(trace), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic no-clobber publish: concurrent runs cannot replace each other.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def load_hybrid_run(base_dir, run_id):
    path = _run_path(base_dir, run_id)
    if path.stat().st_size > 5_000_000:
        raise ValueError("hybrid audit file exceeds size limit")
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace.get("schema_version") != SCHEMA_VERSION or trace.get("run_id") != run_id:
        raise ValueError("invalid hybrid trace identity")
    if trace.get("decision", {}).get("execution_authorized") is not False:
        raise ValueError("audit file must never authorize execution")
    return _public(trace)


def list_hybrid_runs(base_dir, limit=30):
    root = Path(base_dir).resolve() / "outputs" / "hybrid_runs"
    if not root.exists():
        return []
    records = []
    for path in sorted(root.glob("hybrid_*.json"), reverse=True):
        try:
            trace = load_hybrid_run(base_dir, path.stem)
            records.append({"run_id": trace["run_id"], "generated_at": trace["generated_at"],
                            "market_as_of": trace["snapshot"].get("as_of"), "status": trace["decision"]["status"]})
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(records) >= max(1, min(int(limit), 100)):
            break
    return records
