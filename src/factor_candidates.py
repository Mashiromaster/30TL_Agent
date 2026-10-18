"""Build quarantined local factor candidates; never update live data or models.

The provenance contract certifies the algorithm used here, not the historical
vendor identities, publication vintages, trading calendar, or trading fitness.
Pickles are trusted local project inputs, not an untrusted interchange format.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from factor_extraction import calculate_enhanced_factors
from macro_timing import merge_point_in_time


SCHEMA_VERSION = 1
PIT_CONTRACT = "previous_completed_observed_day_oi_v1"
_MARKET_COLUMNS = ["date", "trade_dt", "ticker", "open", "high", "low", "close", "volume", "oi"]
_NUMERIC = ["open", "high", "low", "close", "volume", "oi"]
# Matches src/check_timestamp_format.py's session hours and the observed local
# end-labelled raw bars. This is not an independently verified exchange calendar.
_SESSION_MINUTES = tuple(range(571, 691)) + tuple(range(781, 916))
_LIMITATIONS = [
    "Research-only historical candidate; no freshness, tradability or full PIT certification.",
    "Source data lacks independently verified vendor identity and historical revision vintages.",
    "Completion uses a 255-bar TL end-labelled minute template, not an exchange calendar.",
    "Absent observed dates cannot be classified as holidays versus missing sessions.",
    "Previous observed day closing OI may omit contracts absent from that source snapshot.",
    "Roll adjustment is a forward synthetic return chain with zero switch-bar return; not executable PnL.",
    "Core legacy factor windows count rows (240 bars labelled 1D), not elapsed trading days.",
    "Core calculator drops 500 warm-up rows and fills unavailable market features with zero; long windows need up to 10000 bars.",
    "Tick, basis, text, CNN and GP sources are excluded; legacy model feature compatibility is not asserted.",
    "money is deliberately excluded: mixed historical estimation semantics are not certified turnover.",
]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_snapshot(path):
    payload = path.read_bytes()
    return payload, _sha(payload)


def _wall_time(values):
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError("mixed or invalid timestamp timezones")
    return parsed


def _asof(now):
    stamp = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    if pd.isna(stamp):
        raise ValueError("invalid as-of timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    return stamp.tz_convert("Asia/Shanghai")


def _atomic_json(path, value):
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _check_candidate_output(base, directory):
    """Reject redirected output ancestors before mkdir or any artifact write."""
    if not directory.resolve().is_relative_to(base):
        raise ValueError("candidate output path resolves outside the project")
    # Even an in-project alias can redirect candidates into production outputs
    # and disagrees with the canonical candidate path used by provenance checks.
    for component in (base / "artifacts", base / "artifacts/factor_candidates", directory):
        if component.is_symlink():
            raise ValueError("candidate output path must not contain symbolic links")


def _validate_market(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("market source is not a non-empty DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("duplicate market columns")
    missing = set(_MARKET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing market columns: {sorted(missing)}")
    # Do not propagate unexplained old features, estimated money, or CNN columns.
    data = frame[_MARKET_COLUMNS].copy()
    for name in ("date", "trade_dt"):
        if data[name].map(lambda value: isinstance(value, (bool, np.bool_, int, float, np.number))).any():
            raise ValueError(f"numeric timestamp units are not declared for {name}")
        data[name] = _wall_time(data[name])
        if data[name].isna().any():
            raise ValueError(f"invalid timestamp in {name}")
    if not data.date.dt.normalize().eq(data.trade_dt.dt.normalize()).all():
        raise ValueError("trade_dt disagrees with date; no trade-day shifting is allowed")
    data["trade_dt"] = data.date.dt.normalize()
    if not data.date.eq(data.date.dt.floor("min")).all():
        raise ValueError("market source contains sub-minute timestamps")
    if not data.ticker.astype(str).str.fullmatch(r"TL\d{2}(03|06|09|12)").all():
        raise ValueError("only explicit TL quarterly contract symbols are permitted")
    if data.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date+ticker in market source")
    for name in _NUMERIC:
        values = data[name]
        if (pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype) or
                (not pd.api.types.is_numeric_dtype(values.dtype) and values.map(
                    lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
                ).any())):
            raise ValueError(f"market numeric columns require real non-boolean values: {name}")
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if not np.isfinite(data[_NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("non-finite market price, volume or OI")
    prices = data[["open", "high", "low", "close"]]
    if ((prices <= 0).any(axis=1) | (data.high < prices.max(axis=1)) |
            (data.low > prices.min(axis=1))).any():
        raise ValueError("invalid OHLC bounds or non-positive price")
    if (data[["volume", "oi"]] < 0).any().any():
        raise ValueError("negative volume or OI")
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def _minute_set(frame):
    return set((frame.date.dt.hour * 60 + frame.date.dt.minute).tolist())


def _causal_main(data, asof, audit):
    # A candidate only contains fully ended observed days. Current-day bars are
    # withheld even after the nominal close: there is no exchange calendar here.
    completed = data.loc[data.trade_dt < asof.tz_localize(None).normalize()].copy()
    audit["excluded_not_completed_rows"] = int(len(data) - len(completed))
    audit["quarantined_days"] = []
    audit["selections"] = []
    audit["oi_observation_exclusions"] = []
    observed_days = sorted(completed.trade_dt.unique())
    audit["observed_day_gaps"] = [
        {"after": pd.Timestamp(left).date().isoformat(),
         "before": pd.Timestamp(right).date().isoformat(),
         "unobserved_calendar_days": (pd.Timestamp(right) - pd.Timestamp(left)).days - 1}
        for left, right in zip(observed_days, observed_days[1:])
        if (pd.Timestamp(right) - pd.Timestamp(left)).days > 1
    ]
    prior_day = None
    target = None
    selected = []
    expected = set(_SESSION_MINUTES)
    for day, daily in completed.groupby("trade_dt", sort=True):
        reason = None
        chosen = daily.loc[daily.ticker.eq(target)].copy()
        if prior_day is None:
            reason = "first_observed_day"
        elif target is None:
            reason = "prior_observed_day_incomplete"
        elif chosen.empty:
            reason = "selected_contract_missing"
        elif len(chosen) != 255 or _minute_set(chosen) != expected:
            reason = "selected_contract_incomplete"
        if reason:
            audit["quarantined_days"].append({
                "date": day.date().isoformat(), "reason": reason,
                "selected_contract": target, "observed_selected_rows": len(chosen),
            })
        else:
            decision_ts = prior_day + pd.Timedelta(hours=15, minutes=15)
            chosen["oi_selection_asof"] = decision_ts
            selected.append(chosen)
            audit["selections"].append({
                "date": day.date().isoformat(), "ticker": target,
                "oi_selection_asof": decision_ts.isoformat(), "rows": len(chosen),
            })
        # Read end-of-day OI only after making this day's selection. If the
        # previous observed day is incomplete, do not silently reach farther back.
        closing = daily.loc[(daily.date.dt.hour == 15) & (daily.date.dt.minute == 15)]
        all_contracts = set(daily.ticker)
        excluded = sorted(all_contracts - set(closing.ticker))
        if excluded:
            audit["oi_observation_exclusions"].append({
                "date": day.date().isoformat(), "contracts_without_closing_oi": excluded,
            })
        if _minute_set(daily) != expected or closing.empty:
            target = None
        else:
            target = closing.sort_values(["oi", "ticker"], ascending=[False, True]).iloc[0].ticker
        prior_day = day
    if not selected:
        raise ValueError("no complete causal contract days remain after quarantine")
    main = pd.concat(selected, ignore_index=True).sort_values("date").reset_index(drop=True)
    switched = main.ticker.ne(main.ticker.shift())
    main["is_switch"] = switched.astype(int)
    main["raw_ret"] = main.close.pct_change(fill_method=None)
    main.loc[switched, "raw_ret"] = 0.0
    main["adj_close"] = 100.0 * (1.0 + main.raw_ret).cumprod()
    return main


def _local_macro(base_dir, asof, inputs):
    cache = base_dir / "outputs/macro_factors.pkl"
    meta_path = base_dir / "outputs/macro_factors.meta.json"
    source = {"status": "excluded", "reason": "local verified macro cache unavailable"}
    if not cache.is_file() or not meta_path.is_file():
        return None, {}, source
    try:
        content, digest = _read_snapshot(cache)
        meta_bytes, meta_digest = _read_snapshot(meta_path)
        source.update({"file": str(cache), "sha256": digest, "metadata_sha256": meta_digest})
        metadata = json.loads(meta_bytes)
        if metadata.get("schema_version") != 2:
            raise ValueError("macro cache requires schema_version=2")
        if metadata.get("availability_verified") is not True or metadata.get("vintage_verified") is not True:
            raise ValueError("publication availability and historical vintages are not verified")
        if metadata.get("cache_sha256") != digest:
            raise ValueError("macro cache hash is not bound to verified metadata")
        events = pd.read_pickle(io.BytesIO(content))
        if not isinstance(events, pd.DataFrame) or events.empty or "available_ts" not in events:
            raise ValueError("macro events require non-empty available_ts")
        ttls = metadata.get("factor_ttls", {})
        if not isinstance(ttls, dict) or not ttls:
            raise ValueError("macro factor TTLs are missing")
        columns = ["available_ts"]
        for name, ttl in ttls.items():
            asof_col = f"_asof__{name}"
            if not isinstance(name, str) or not name or name in _MARKET_COLUMNS or name.startswith(("_", "CNN_", "GP_")):
                raise ValueError("invalid macro factor name")
            if not np.isfinite(float(ttl)) or float(ttl) <= 0:
                raise ValueError("macro TTL must be finite and positive")
            if name not in events or asof_col not in events:
                raise ValueError("macro factor or its per-factor availability is missing")
            columns.extend([name, asof_col])
        events = events[columns].copy()
        for column in ["available_ts"] + [f"_asof__{name}" for name in ttls]:
            events[column] = _wall_time(events[column])
            if events[column].isna().any():
                raise ValueError("invalid macro availability timestamp")
        if events.available_ts.duplicated().any():
            raise ValueError("duplicate macro event availability")
        for name in ttls:
            values = pd.to_numeric(events[name], errors="coerce")
            if (events[name].notna() & values.isna()).any() or np.isinf(values).any():
                raise ValueError("invalid macro factor values")
            events[name] = values
            if events[f"_asof__{name}"].gt(events.available_ts).any():
                raise ValueError("macro per-factor availability exceeds event availability")
        events = events.loc[events.available_ts <= asof.tz_localize(None)].copy()
        if events.empty:
            raise ValueError("macro cache contains no available events at as-of")
        inputs[str(cache)] = digest
        inputs[str(meta_path)] = meta_digest
        source.update({"status": "included", "reason": "verified local metadata assertion; no external re-verification", "rows": len(events)})
        return events, {name: pd.Timedelta(days=float(ttl)) for name, ttl in ttls.items()}, source
    except Exception as exc:
        source["reason"] = f"{type(exc).__name__}: {exc}"
        return None, {}, source


def build_local_factor_candidate(base_dir, *, now=None, minute_snapshot_id=None) -> dict:
    """Build a separate candidate, returning ``built`` or durable ``failed`` audit.

    This API has no fetching/training/promotion path. ``now`` is the decision
    as-of in Shanghai time; naive timestamps are interpreted in that timezone.
    Existing output/model/cache files are never passed to the legacy pipeline.
    An explicit minute_snapshot_id selects an isolated imported data bundle;
    there is no automatic fallback to the old raw source if it is invalid.
    """
    base = Path(base_dir).expanduser().resolve()
    candidate_id = f"local-{uuid.uuid4().hex}"
    directory = base / "artifacts/factor_candidates" / candidate_id
    source_file = base / "data/TL分钟级量价数据.pkl"
    result = {"status": "failed", "candidate_id": candidate_id, "factor_path": None,
              "manifest_path": None, "coverage": {}, "warnings": list(_LIMITATIONS)}
    manifest = {"schema_version": SCHEMA_VERSION, "status": "failed", "candidate_id": candidate_id,
                "source_file": str(source_file), "source_sha256": None,
                "factor_file": None, "factor_sha256": None, "coverage": {},
                "limitations": list(_LIMITATIONS), "network_mode": "offline", "production_modified": False,
                "sources": {name: {"status": "excluded", "reason": reason} for name, reason in {
                    "tick": "unverified bar availability and cross-contract rolling semantics",
                    "basis": "unverified publication availability and stale carry-forward semantics",
                    "text": "not admitted to this market-only candidate",
                    "CNN": "requires fold-local training and a new feature contract",
                    "GP": "requires fold-local training and a new feature contract",
                }.items()}}
    audit = {"schema_version": SCHEMA_VERSION, "candidate_id": candidate_id, "status": "failed"}
    try:
        _check_candidate_output(base, directory)
        directory.mkdir(parents=True, exist_ok=False)
        result["manifest_path"] = str(directory / "manifest.json")
        asof = _asof(now)
        manifest["as_of"] = asof.isoformat()
        snapshot_manifest = None
        if minute_snapshot_id is not None:
            from minute_store import load_minute_snapshot
            source_frame, source_file, digest, snapshot_manifest = load_minute_snapshot(base, minute_snapshot_id)
            metadata_file = source_file.parent / "manifest.json"
            metadata_bytes, metadata_digest = _read_snapshot(metadata_file)
            if json.loads(metadata_bytes) != snapshot_manifest:
                raise ValueError("minute snapshot manifest changed while loading")
            inputs = {str(source_file): digest, str(metadata_file): metadata_digest}
            manifest["source_file"] = str(source_file)
            manifest["minute_snapshot_id"] = minute_snapshot_id
        else:
            content, digest = _read_snapshot(source_file)
            source_frame = pd.read_pickle(io.BytesIO(content))
            inputs = {str(source_file): digest}
        manifest["source_sha256"] = digest
        raw = _validate_market(source_frame)
        manifest["source_rows"] = len(raw)
        manifest["source_start"] = raw.date.min().isoformat()
        manifest["source_end"] = raw.date.max().isoformat()
        manifest["sources"]["market"] = {"status": "included", "sha256": digest, "file": str(source_file),
            "provider_verified": False, "source_kind": "isolated_minute_snapshot" if snapshot_manifest else "legacy_unverified"}
        if snapshot_manifest is not None:
            manifest["sources"]["market"].update(snapshot_id=minute_snapshot_id,
                manifest_sha256=metadata_digest, inputs=snapshot_manifest.get("inputs", []),
                acquired_at=snapshot_manifest.get("acquired_at"), source_trust=snapshot_manifest.get("source_trust", "declared_unverified"))
        main = _causal_main(raw, asof, audit)
        if len(main) <= 500:
            raise ValueError("insufficient selected bars for the market calculator's 500-row warm-up")
        events, ttls, macro_source = _local_macro(base, asof, inputs)
        manifest["sources"]["macro"] = macro_source
        # Only this pure in-memory calculator is reused. No loader, sentiment
        # builder, old splice cache, or run_process call is reachable here.
        factors = calculate_enhanced_factors(main, df_tick=None, df_basis=None, df_macro=None, bar_format="end")
        if factors.empty:
            raise ValueError("no factors after the market calculator's 500-row warm-up")
        if events is not None:
            if set(ttls) & set(factors.columns):
                raise ValueError("macro factor names collide with market columns")
            # Merge after market cleaning so absent/stale macro evidence stays
            # missing, rather than becoming an apparently valid zero signal.
            factors = merge_point_in_time(factors, events, factor_ttls=ttls,
                                         fill_value=np.nan, keep_asof_columns=True)
            factors.attrs["macro_features"] = list(ttls)
        if any(column.startswith(("CNN_", "GP_")) for column in factors):
            raise ValueError("unexpected trained feature in local market candidate")
        provenance = {"schema_version": 1, "kind": "local_causal_rebuild_v1",
                      "pit_contract": PIT_CONTRACT, "source_sha256": digest,
                      "limitations": list(_LIMITATIONS), "candidate_id": candidate_id}
        if snapshot_manifest is not None:
            provenance.update(minute_snapshot_id=minute_snapshot_id, minute_manifest_sha256=metadata_digest)
        factors.attrs["factor_provenance"] = provenance
        dates = factors.date.sort_values()
        selected_dates = pd.to_datetime(main.trade_dt.drop_duplicates()).sort_values()
        coverage = {"start": dates.min().isoformat(), "end": dates.max().isoformat(),
                    "rows": len(factors), "columns": len(factors.columns),
                    "source_end": raw.date.max().isoformat(), "selected_rows": len(main),
                    "quarantined_days": len(audit["quarantined_days"]),
                    "excluded_not_completed_rows": audit["excluded_not_completed_rows"],
                    "age_hours": float((asof.tz_localize(None) - dates.max()).total_seconds() / 3600),
                    "largest_observed_day_gap_days": int(selected_dates.diff().dt.days.max()) if len(selected_dates) > 1 else 0,
                    "calendar_verified": False, "freshness_certified": False,
                    "template_source": "src/check_timestamp_format.py; observed local end-labelled bars",
                    "minute_template": "09:31-11:30,13:01-15:15 Asia/Shanghai; end-labelled; 255 bars",
                    "long_window_warmup_rows_below_10001": max(0, min(len(main), 10000) - (len(main) - len(factors)))}
        # Detect input replacement during a build; output hashes must bind the
        # exact source snapshot, not a source file that changed in the meantime.
        for filename, expected in inputs.items():
            if _sha(Path(filename).read_bytes()) != expected:
                raise ValueError(f"source changed during candidate construction: {filename}")
        factor_path = directory / "factors.pkl"
        _check_candidate_output(base, directory)
        temporary = directory / "factors.tmp"
        try:
            with temporary.open("xb") as handle:
                factors.to_pickle(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, factor_path)
        finally:
            temporary.unlink(missing_ok=True)
        factor_digest = _sha(factor_path.read_bytes())
        manifest.update({"status": "built", "provenance": provenance, "factor_file": "factors.pkl",
                         "factor_sha256": factor_digest, "coverage": coverage,
                         "factor_columns": list(factors.columns), "input_sha256": inputs})
        audit.update({"status": "built", "coverage": coverage,
                      "missing_values_by_column": {name: int(n) for name, n in factors.isna().sum().items() if n}})
        result.update({"status": "built", "factor_path": str(factor_path), "coverage": coverage})
        result["warnings"].extend(f"{name} excluded: {entry['reason']}" for name, entry in manifest["sources"].items() if entry["status"] == "excluded")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["warnings"].append(error)
        manifest["error"] = audit["error"] = error
    try:
        if result["manifest_path"]:
            _check_candidate_output(base, directory)
            manifest["warnings"] = result["warnings"]
            _atomic_json(directory / "audit.json", audit)
            manifest["audit_sha256"] = _sha((directory / "audit.json").read_bytes())
            _atomic_json(directory / "manifest.json", manifest)
    except Exception as exc:
        result.update({"status": "failed", "factor_path": None, "manifest_path": None})
        result["warnings"].append(f"audit persistence failed: {type(exc).__name__}: {exc}")
    return result
