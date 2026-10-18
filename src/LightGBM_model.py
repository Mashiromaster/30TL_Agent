# -*- coding: utf-8 -*-
"""Leakage-safe LightGBM training with macro/micro ablations."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from scipy import stats
from threadpoolctl import threadpool_limits

from artifact_audit import extract_text_factor_metadata
from feature_registry import FeatureGroups, discover_feature_groups
from hierarchical_model import HierarchicalMacroMicroModel
from jev_evaluation import load_semantic_gate_status
from intelligence_events import redact_sensitive_text
from model_release import (
    BASELINE_MAX_DRAWDOWN,
    BASELINE_MIN_OBSERVATION_DAYS,
    BASELINE_MIN_SHARPE,
    BASELINE_MIN_TOTAL_RETURN,
    ReleaseManager,
    sha256_file,
    write_candidate_manifest,
)
from strategy_runtime import get_strategy_spec
from training_data import (
    LabelSpec,
    SplitSpec,
    build_forward_labels,
    split_train_validation_holdout,
)


class IdentityScaler:
    """Backward-compatible no-op scaler; LightGBM consumes raw numeric values."""

    def fit(self, values):
        return self

    def transform(self, values):
        return values

    def fit_transform(self, values):
        return self.transform(values)


def _ic(target, prediction):
    pair = pd.DataFrame({"y": target, "p": prediction}).dropna()
    if len(pair) < 10 or pair["y"].nunique() < 2 or pair["p"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(pair["y"], pair["p"]).statistic)


def _json_number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _date_block_stability(frame, prediction, blocks=5):
    days = pd.to_datetime(frame["date"]).dt.normalize()
    unique_days = np.asarray(sorted(days.dropna().unique()))
    predictions = pd.Series(prediction, index=frame.index)
    values = []
    for block_days in np.array_split(unique_days, blocks):
        if len(block_days) == 0:
            continue
        mask = days.isin(block_days)
        value = _ic(frame.loc[mask, "Target_Ret"], predictions.loc[mask])
        if np.isfinite(value):
            values.append(value)
    median = float(np.median(values)) if values else np.nan
    same_sign = (
        float(np.mean(np.sign(values) == np.sign(median)))
        if values and median != 0 else 0.0
    )
    return {
        "fold_ics": [_json_number(value) for value in values],
        "median_ic": _json_number(median),
        "same_sign_ratio": same_sign,
        "positive_fold_count": int(np.sum(np.asarray(values) > 0)),
        "fold_count": len(values),
    }


def _validation_promotion_eligible(stability):
    """Validation-only candidate eligibility; never reads sealed holdout data."""
    return bool(
        stability["fold_count"] >= 5
        and stability["positive_fold_count"] >= 3
        and (stability["median_ic"] or 0.0) > 0
    )


def _release_gate_passed(promotion_eligible, sealed_holdout_ic):
    """One-time deployment veto after all candidate choices have been frozen."""
    return bool(
        promotion_eligible
        and np.isfinite(sealed_holdout_ic)
        and sealed_holdout_ic > 0
    )


def _validate_release_policy(
    *,
    max_drawdown_limit=BASELINE_MAX_DRAWDOWN,
    min_total_return=BASELINE_MIN_TOTAL_RETURN,
    min_sharpe=BASELINE_MIN_SHARPE,
):
    """Accept only finite numeric thresholds that preserve the baseline."""
    for name, value in (("max_drawdown_limit", max_drawdown_limit),
                        ("min_total_return", min_total_return), ("min_sharpe", min_sharpe)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real number")
    if not 0 < float(max_drawdown_limit) <= BASELINE_MAX_DRAWDOWN:
        raise ValueError(
            "max_drawdown_limit must be in (0, 0.35]; the baseline can only "
            "be tightened"
        )
    if float(min_total_return) < BASELINE_MIN_TOTAL_RETURN:
        raise ValueError("min_total_return cannot weaken the zero baseline")
    if float(min_sharpe) < BASELINE_MIN_SHARPE:
        raise ValueError("min_sharpe cannot weaken the zero baseline")


def _strategy_release_gate(
    metrics,
    *,
    max_drawdown_limit=BASELINE_MAX_DRAWDOWN,
    min_total_return=BASELINE_MIN_TOTAL_RETURN,
    min_sharpe=BASELINE_MIN_SHARPE,
):
    """Cost-adjusted operational and performance gate for production release."""
    _validate_release_policy(max_drawdown_limit=max_drawdown_limit,
                             min_total_return=min_total_return, min_sharpe=min_sharpe)
    required = (
        "total_return",
        "sharpe",
        "annual_volatility",
        "max_drawdown",
        "active_ratio",
        "observation_days",
    )
    if any(name not in metrics for name in required):
        return False, "missing_metrics"
    values = [metrics[name] for name in required]
    if not all(np.isfinite(value) for value in values):
        return False, "non_finite_metrics"
    if int(metrics["observation_days"]) < BASELINE_MIN_OBSERVATION_DAYS:
        return False, "insufficient_observation_days"
    active_ratio = float(metrics["active_ratio"])
    if active_ratio <= 0:
        return False, "strategy_never_active"
    if active_ratio > 1:
        return False, "invalid_active_ratio"
    if float(metrics["annual_volatility"]) < 0:
        return False, "invalid_annual_volatility"
    if not -1.0 <= float(metrics["max_drawdown"]) <= 0.0:
        return False, "invalid_max_drawdown"
    if float(metrics["total_return"]) <= float(min_total_return):
        return False, "total_return_below_threshold"
    if float(metrics["sharpe"]) <= float(min_sharpe):
        return False, "sharpe_below_threshold"
    if float(metrics["max_drawdown"]) < -float(max_drawdown_limit):
        return False, "max_drawdown_breach"
    return True, "passed"


def _release_identifiers(strategy_fingerprint):
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    suffix = str(strategy_fingerprint).split(":")[-1][:12]
    return f"train-{timestamp}", f"tl-{timestamp}-{suffix}"


def prepare_training_data(df, prediction_horizon=30, label_spec=None):
    """Build one label definition and discover a validated feature schema."""
    spec = label_spec or LabelSpec(horizon_bars=prediction_horizon)
    labelled = build_forward_labels(df, spec)
    groups = discover_feature_groups(labelled)
    features = list(groups.all_features)
    model = labelled.loc[labelled["label_valid"]].copy()
    model.attrs["feature_groups"] = groups.as_dict()
    model.attrs["label_spec"] = spec.__dict__
    text_factor_metadata = extract_text_factor_metadata(df)
    if text_factor_metadata:
        model.attrs["text_factor_metadata"] = text_factor_metadata
    print(
        f"[Model] features={len(features)} "
        f"micro={len(groups.micro)} market={len(groups.market)} "
        f"macro={len(groups.macro)} regime={len(groups.regime)} gp={len(groups.gp)}"
    )
    return model, features


def _apply_jev_semantic_gate(df, base_dir):
    """Remove Jev candidate features until their sealed semantic gate passes."""
    metadata = extract_text_factor_metadata(df)
    jev_features = sorted(
        name for name in df.columns if str(name).startswith("JEV_")
    )
    provenance = dict(metadata)
    if not jev_features:
        provenance.update(
            {
                "semantic_gate_passed": None,
                "semantic_gate_reason": "no_jev_features",
                "semantic_gate_dataset_id": "",
                "training_jev_features": [],
            }
        )
        return df, provenance

    passed, reason, gate_report = load_semantic_gate_status(base_dir, metadata)
    provenance.update(
        {
            "semantic_gate_passed": passed,
            "semantic_gate_reason": reason,
            "semantic_gate_dataset_id": str(gate_report.get("dataset_id", "")),
            "training_jev_features": jev_features if passed else [],
        }
    )
    if passed:
        return df, provenance

    filtered = df.drop(columns=jev_features).copy()
    filtered.attrs = dict(df.attrs)
    print(
        "[Model] Jev semantic gate not passed "
        f"({reason}); excluded {len(jev_features)} Jev features"
    )
    return filtered, provenance


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_type="base",
    sample_weight=None,
    *,
    thread_limit=None,
):
    """Fit a regularized LightGBM model using validation-only early stopping."""
    high_capacity = model_type == "highvol"
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.01,
        num_leaves=12 if high_capacity else 8,
        max_depth=4 if high_capacity else 3,
        lambda_l1=10.0 if high_capacity else 15.0,
        lambda_l2=10.0 if high_capacity else 15.0,
        feature_fraction=0.5 if high_capacity else 0.4,
        bagging_fraction=0.6 if high_capacity else 0.5,
        bagging_freq=5,
        min_child_samples=250 if high_capacity else 350,
        min_split_gain=0.01,
        objective="regression_l1",
        importance_type="gain",
        random_state=42,
        n_jobs=thread_limit if thread_limit is not None else -1,
        verbosity=-1,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=75, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    return model


def _time_decay_weights(frame, half_life_days):
    if not half_life_days:
        return None
    dates = pd.to_datetime(frame["date"])
    age = (dates.max() - dates).dt.total_seconds().div(86400)
    return np.exp(-np.log(2) * age / half_life_days).to_numpy()


def _fit_flat(train, validation, features, sample_weight=None, *, thread_limit=None):
    if not features:
        return None
    return train_model(
        train[features],
        train["Target_Ret"].to_numpy(),
        validation[features],
        validation["Target_Ret"].to_numpy(),
        sample_weight=sample_weight,
        thread_limit=thread_limit,
    )


def _feature_importance(model, features):
    booster = model.booster_
    split = booster.feature_importance(importance_type="split")
    gain = booster.feature_importance(importance_type="gain")
    gain_total = gain.sum()
    return pd.DataFrame(
        {
            "feature": features,
            "importance": gain,
            "importance_split": split,
            "importance_gain": gain,
            "importance_gain_pct": gain / gain_total if gain_total else 0.0,
        }
    ).sort_values("importance_gain", ascending=False)


def _macro_block_permutation(
    predictor,
    frame,
    macro_features,
    *,
    repeats=3,
    random_state=42,
):
    """Measure OOS delta-IC after permuting macro states by whole trade day."""
    if not macro_features:
        return {}
    baseline = _ic(frame["Target_Ret"], predictor(frame))
    days = pd.to_datetime(frame["date"]).dt.normalize()
    unique_days = pd.Index(days.unique())
    rng = np.random.default_rng(random_state)
    result = {}
    for feature in macro_features:
        daily_value = frame.assign(_day=days).groupby("_day")[feature].last()
        deltas = []
        for _ in range(repeats):
            shuffled = daily_value.reindex(rng.permutation(unique_days)).to_numpy()
            mapping = dict(zip(unique_days, shuffled))
            permuted = frame.copy()
            permuted[feature] = days.map(mapping).to_numpy()
            deltas.append(baseline - _ic(frame["Target_Ret"], predictor(permuted)))
        finite = [value for value in deltas if np.isfinite(value)]
        result[feature] = float(np.mean(finite)) if finite else np.nan
    return dict(sorted(result.items(), key=lambda item: abs(item[1]), reverse=True))


def _candidate_result():
    return {
        "status": "running", "published": False, "candidate_manifest": None,
        "run_id": None, "release_id": None, "gates": {},
        "input_snapshot": {}, "partitions": {}, "errors": [],
    }


def _check_candidate_output(base, directory):
    """Reject redirected candidate ancestors and artifact paths before writes."""
    for component in (base / "artifacts", base / "artifacts/experiments", directory):
        if component.is_symlink():
            raise ValueError("candidate output path must not contain symbolic links")
    if directory.parent != base / "artifacts/experiments" or not directory.resolve().is_relative_to(base):
        raise ValueError("candidate output path must stay in the project experiments directory")
    for name in ("input_snapshot.pkl", "df_predictions.pkl", "feature_importance.csv",
                 "trained_model.pkl", "evaluation.json", "evaluation.json.tmp", "manifest.json"):
        if (directory / name).is_symlink():
            raise ValueError("candidate artifact path must not contain symbolic links")


def _frame_summary(frame):
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return {
        "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "observed_days": int(dates.dt.normalize().nunique()),
        "start": dates.min().isoformat() if dates.notna().any() else None,
        "end": dates.max().isoformat() if dates.notna().any() else None,
    }


def _input_unchanged(source_path, expected_hash):
    try:
        return sha256_file(source_path) == expected_hash
    except OSError:
        return False


def _verify_factor_provenance(base_dir, source_path, frame, *, expected_sha256=None):
    """Fail closed until a producer's PIT contract and input hash are verified.

    Presence of numeric features or good model metrics cannot attest that an
    old CNN embedding or macro feature was produced without future data.
    """
    try:
        from research_readiness import verify_factor_provenance
        return verify_factor_provenance(
            base_dir, source_path, frame, expected_sha256=expected_sha256,
        )
    except Exception as exc:
        return {"verified": False, "reason": "unverified_factor_provenance",
                "detail": redact_sensitive_text(exc)[:1000]}


def _write_training_bundle(candidate_dir, artifacts, report, base_dir):
    """Write a candidate bundle, never an active release or compatibility view."""
    _check_candidate_output(base_dir, candidate_dir)
    candidate_model_file = candidate_dir / "trained_model.pkl"
    with candidate_model_file.open("wb") as handle:
        pickle.dump(artifacts, handle)
    report["gate_attestation"] = {
        "schema_version": 1,
        "release_id": report["release_id"], "run_id": report["run_id"],
        "strategy_id": report["strategy_id"],
        "strategy_fingerprint": report["strategy_fingerprint"],
        "model_sha256": sha256_file(candidate_model_file),
        "gate_passed": report["release_gate_passed"],
        "policy": {
            "min_observation_days": BASELINE_MIN_OBSERVATION_DAYS,
            "min_total_return": report["min_total_return"],
            "min_sharpe": report["min_sharpe"],
            "max_drawdown_limit": report["max_drawdown_limit"],
        },
    }
    report_file = candidate_dir / "evaluation.json"
    temporary_report = report_file.with_suffix(".json.tmp")
    try:
        _check_candidate_output(base_dir, candidate_dir)
        with temporary_report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        _check_candidate_output(base_dir, candidate_dir)
        os.replace(temporary_report, report_file)
    finally:
        # A failed guard must not follow a newly redirected directory for cleanup.
        _check_candidate_output(base_dir, candidate_dir)
        temporary_report.unlink(missing_ok=True)
    _check_candidate_output(base_dir, candidate_dir)
    return write_candidate_manifest(
        candidate_dir, release_id=report["release_id"], run_id=report["run_id"],
        artifacts={
            "model": "trained_model.pkl", "predictions": "df_predictions.pkl",
            "feature_importance": "feature_importance.csv", "evaluation": "evaluation.json",
            "training_input": "input_snapshot.pkl",
        },
        metadata={
            "strategy_id": report["strategy_id"], "strategy_version": report["strategy_version"],
            "strategy_fingerprint": report["strategy_fingerprint"],
            "release_gate_passed": report["release_gate_passed"],
            "strategy_gate_reason": report["strategy_gate_reason"],
            "input_sha256": report["input_snapshot"]["sha256"],
            "factor_provenance_verified": report["factor_provenance"]["verified"] is True,
        },
    )


def train_candidate(
    base_dir, *, factor_path=None, max_lookback_months=9,
    time_decay_half_life=60, strategy_id=None, thread_limit=2,
):
    """Train/evaluate one isolated candidate; never activate or project it.

    The optional input is read only. A byte-for-byte frozen copy is kept in the
    candidate directory, and source or snapshot changes veto eligibility.
    Native/estimator thread limits default to two only on this candidate path.
    Failed gates retain an inspectable candidate; invalid input returns an error.
    """
    result = _candidate_result()
    try:
        if isinstance(thread_limit, bool) or not isinstance(thread_limit, int) or thread_limit < 1:
            raise ValueError("thread_limit must be a positive integer")
        result["thread_limit"] = thread_limit
        with threadpool_limits(limits=thread_limit):
            return _run_training_core(
                base_dir, max_lookback_months, time_decay_half_life,
                strategy_id=strategy_id, candidate_only=True,
                factor_path=factor_path, result=result, thread_limit=thread_limit,
            )
    except Exception as exc:
        result.update(status="failed", published=False)
        result["errors"].append(redact_sensitive_text(exc)[:1000])
        return result


def run_process(
    base_dir,
    max_lookback_months=9,
    time_decay_half_life=60,
    *,
    strategy_id=None,
    candidate_only=False,
    factor_path=None,
    thread_limit=None,
):
    """Compatibility training entry; explicit candidate_only disables release."""
    if candidate_only:
        return train_candidate(
            base_dir, factor_path=factor_path, max_lookback_months=max_lookback_months,
            time_decay_half_life=time_decay_half_life, strategy_id=strategy_id,
            thread_limit=2 if thread_limit is None else thread_limit,
        )
    if thread_limit is not None:
        raise ValueError("thread_limit is supported only for candidate_only training")
    result = _run_training_core(
        base_dir, max_lookback_months, time_decay_half_life,
        strategy_id=strategy_id, factor_path=factor_path,
    )
    return result["published"]


def _run_training_core(
    base_dir,
    max_lookback_months=9,
    time_decay_half_life=60,
    *,
    strategy_id=None,
    candidate_only=False,
    factor_path=None,
    result=None,
    thread_limit=None,
):
    result = result if result is not None else _candidate_result()
    base_dir = Path(base_dir).resolve()
    # Freeze and validate policy before any input copying or expensive fitting.
    max_drawdown_limit = float(os.environ.get("F_AGENT_RELEASE_MAX_DRAWDOWN", "0.35"))
    min_total_return = float(os.environ.get("F_AGENT_RELEASE_MIN_TOTAL_RETURN", "0.0"))
    min_sharpe = float(os.environ.get("F_AGENT_RELEASE_MIN_SHARPE", "0.0"))
    _validate_release_policy(max_drawdown_limit=max_drawdown_limit,
                             min_total_return=min_total_return, min_sharpe=min_sharpe)
    print("\n" + "=" * 50)
    print("STEP 3: point-in-time model training")
    print("=" * 50)
    strategy_spec = get_strategy_spec(strategy_id)
    run_id, release_id = _release_identifiers(strategy_spec.fingerprint())
    candidate_dir = base_dir / "artifacts" / "experiments" / run_id
    result.update(run_id=run_id, release_id=release_id)

    from research_readiness import resolve_factor_path
    factor_file = resolve_factor_path(base_dir, factor_path)
    prediction_file = os.path.join(base_dir, "outputs", "df_predictions.pkl")
    model_file = os.path.join(base_dir, "models", "trained_model.pkl")
    importance_file = os.path.join(base_dir, "outputs", "feature_importance.csv")
    report_projection = os.path.join(
        base_dir, "outputs", "model_ablation_report.json"
    )
    candidate_prediction_file = candidate_dir / "df_predictions.pkl"
    candidate_importance_file = candidate_dir / "feature_importance.csv"
    frozen_input_file = candidate_dir / "input_snapshot.pkl"
    if not os.path.exists(factor_file):
        print(f"[Model ERROR] missing factor file: {factor_file}")
        result.update(status="failed", errors=["missing factor file"])
        return result
    _check_candidate_output(base_dir, candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    if not candidate_only:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)

    initial_hash = sha256_file(factor_file)
    _check_candidate_output(base_dir, candidate_dir)
    shutil.copyfile(factor_file, frozen_input_file)
    frozen_bytes = frozen_input_file.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    result["input_snapshot"] = {
        "source_path": str(factor_file), "path": str(frozen_input_file.resolve()),
        "sha256": frozen_hash, "size_bytes": frozen_input_file.stat().st_size,
        "source_unchanged": initial_hash == frozen_hash and _input_unchanged(factor_file, frozen_hash),
    }
    if not result["input_snapshot"]["source_unchanged"]:
        raise ValueError("factor source changed while freezing input")
    raw = pd.read_pickle(io.BytesIO(frozen_bytes))
    del frozen_bytes
    result["input_snapshot"].update(_frame_summary(raw))
    factor_provenance = _verify_factor_provenance(
        base_dir, factor_file, raw, expected_sha256=frozen_hash,
    )
    raw, text_factor_metadata = _apply_jev_semantic_gate(raw, base_dir)
    label_spec = LabelSpec(horizon_bars=30, entry_lag_bars=1, price_col="adj_close")
    model_frame, _ = prepare_training_data(raw, label_spec=label_spec)
    split_spec = SplitSpec(validation_days=60, sealed_holdout_days=60, min_train_days=120)
    train, validation, holdout = split_train_validation_holdout(model_frame, split_spec)
    if max_lookback_months:
        cutoff = validation["date"].min() - pd.Timedelta(days=max_lookback_months * 30)
        train = train.loc[train["date"] >= cutoff].copy()
    if train.empty:
        raise ValueError("lookback produced an empty training partition")
    result["partitions"] = {
        name: _frame_summary(frame) for name, frame in
        (("train", train), ("validation", validation), ("sealed_holdout", holdout))
    }
    result["input_snapshot"]["valid_label_rows"] = int(len(model_frame))
    result["input_snapshot"]["valid_label_days"] = int(model_frame["date"].dt.normalize().nunique())
    result["input_snapshot"]["unlabelled_days"] = result["input_snapshot"]["observed_days"] - result["input_snapshot"]["valid_label_days"]

    gp_miner = None
    gp_report = None
    gp_enabled = os.environ.get("F_AGENT_GENETIC_MINING", "0").strip().lower() in {
        "1", "true", "yes"
    }
    if gp_enabled:
        from genetic_factor_miner import mine_genetic_factors

        gp_options = {"n_jobs": thread_limit} if thread_limit is not None else {}
        gp_miner, gp_report = mine_genetic_factors(train, validation, **gp_options)
        if gp_miner.selected_programs_:
            gp_inputs = gp_miner.feature_names_in_
            for partition in (train, validation, holdout):
                generated = gp_miner.transform(partition[gp_inputs])
                for name in generated:
                    partition[name] = generated[name]
            print(f"[Model] GP validation-selected factors: {len(gp_miner.selected_programs_)}")
        else:
            print("[Model] GP produced no validation-qualified factors")

    groups = discover_feature_groups(train)
    all_features = list(groups.all_features)
    micro_features = list(dict.fromkeys(
        groups.micro + groups.market + groups.regime + groups.gp
    ))
    macro_features = list(groups.macro)
    if not all_features:
        raise ValueError("no valid model features")
    weights = _time_decay_weights(train, time_decay_half_life)

    fit_options = {"thread_limit": thread_limit} if thread_limit is not None else {}
    flat = _fit_flat(train, validation, all_features, weights, **fit_options)
    micro = _fit_flat(train, validation, micro_features, weights, **fit_options)
    macro = _fit_flat(train, validation, macro_features, weights, **fit_options) if macro_features else None
    hierarchy = HierarchicalMacroMicroModel()
    if thread_limit is not None:
        hierarchy.micro_model.set_params(n_jobs=thread_limit)
    hierarchy.fit(train, validation, feature_groups=groups)

    validation_predictions = {
        "flat_combined": flat.predict(validation[all_features]),
        "micro_only": (
            micro.predict(validation[micro_features]) if micro is not None else None
        ),
        "macro_only": (
            macro.predict(validation[macro_features]) if macro is not None else None
        ),
        "hierarchical": hierarchy.predict(validation),
    }
    validation_metrics = {
        "flat_combined": _ic(
            validation["Target_Ret"], validation_predictions["flat_combined"]
        ),
        "micro_only": (
            _ic(validation["Target_Ret"], validation_predictions["micro_only"])
            if micro is not None else np.nan
        ),
        "macro_only": (
            _ic(validation["Target_Ret"], validation_predictions["macro_only"])
            if macro is not None else np.nan
        ),
        "hierarchical": _ic(
            validation["Target_Ret"], validation_predictions["hierarchical"]
        ),
    }
    validation_stability = {
        name: _date_block_stability(validation, prediction)
        for name, prediction in validation_predictions.items()
        if prediction is not None
    }
    deploy_kind = max(
        ("flat_combined", "hierarchical"),
        key=lambda name: np.nan_to_num(validation_metrics[name], nan=-np.inf),
    )
    if deploy_kind == "hierarchical":
        holdout_prediction = hierarchy.predict(holdout)
        predictor = hierarchy.predict
    else:
        holdout_prediction = flat.predict(holdout[all_features])
        predictor = lambda frame: flat.predict(frame[all_features])
    holdout_ic = _ic(holdout["Target_Ret"], holdout_prediction)
    selected_stability = validation_stability[deploy_kind]
    promotion_eligible = _validation_promotion_eligible(selected_stability)
    release_gate_passed = _release_gate_passed(promotion_eligible, holdout_ic)

    permutation = _macro_block_permutation(
        predictor, holdout, macro_features, repeats=3
    )
    report = {
        "selection_basis": "validation_only",
        "sealed_holdout_used_for_selection": False,
        "selected_model": deploy_kind,
        "validation_ic": {
            name: _json_number(value) for name, value in validation_metrics.items()
        },
        "validation_stability": validation_stability,
        "promotion_eligible": promotion_eligible,
        "release_gate_passed": release_gate_passed,
        "sealed_holdout_used_for_release_veto": True,
        "sealed_holdout_ic": _json_number(holdout_ic),
        "macro_block_permutation_delta_ic": {
            name: _json_number(value) for name, value in permutation.items()
        },
        "train_rows": len(train),
        "validation_rows": len(validation),
        "sealed_holdout_rows": len(holdout),
        "feature_groups": groups.as_dict(),
        "text_factor_metadata": text_factor_metadata,
        "gp": gp_report,
        "run_id": run_id,
        "release_id": release_id,
        "strategy_id": strategy_spec.strategy_id,
        "strategy_version": strategy_spec.version,
        "strategy_fingerprint": strategy_spec.fingerprint(),
    }

    holdout_out = holdout.copy()
    holdout_out["Pred_Ret"] = holdout_prediction
    holdout_out["Model_Used"] = deploy_kind
    holdout_out["Model_Release_ID"] = release_id
    if deploy_kind == "hierarchical":
        components = hierarchy.components(holdout)
        for name in components:
            holdout_out[name] = components[name]
    output_cols = [
        "date", "trade_dt", "ticker", "close", "adj_close", "Target_Ret",
        "Pred_Ret", "Market_Regime", "Model_Used", "micro_alpha", "macro_state",
        "gate_weight", "event_alpha",
        "Model_Release_ID",
    ]
    holdout_out = holdout_out[
        [name for name in output_cols if name in holdout_out]
    ].copy()

    from backtest import evaluate_predictions

    _, strategy_metrics, _ = evaluate_predictions(
        holdout_out,
        strategy_spec=strategy_spec,
        model_release_id=release_id,
    )
    strategy_gate_passed, strategy_gate_reason = _strategy_release_gate(
        strategy_metrics,
        max_drawdown_limit=max_drawdown_limit,
        min_total_return=min_total_return,
        min_sharpe=min_sharpe,
    )
    release_gate_passed = bool(release_gate_passed and strategy_gate_passed)
    input_unchanged = (_input_unchanged(factor_file, frozen_hash)
                       and _input_unchanged(frozen_input_file, frozen_hash))
    factor_provenance_verified = factor_provenance.get("verified") is True
    release_gate_passed = bool(release_gate_passed and input_unchanged and factor_provenance_verified)
    result["input_snapshot"]["source_unchanged"] = input_unchanged
    result["gates"] = {
        "validation_passed": bool(promotion_eligible),
        "holdout_passed": bool(np.isfinite(holdout_ic) and holdout_ic > 0),
        "strategy_passed": bool(strategy_gate_passed),
        "input_unchanged": input_unchanged,
        "factor_provenance_verified": factor_provenance_verified,
        "factor_provenance_reason": factor_provenance.get("reason", "unverified_factor_provenance"),
        "release_gate_passed": release_gate_passed,
    }
    report.update(
        {
            "release_gate_passed": release_gate_passed,
            "strategy_backtest": strategy_metrics,
            "strategy_gate_passed": strategy_gate_passed,
            "strategy_gate_reason": strategy_gate_reason,
            "max_drawdown_limit": max_drawdown_limit,
            "min_total_return": min_total_return,
            "min_sharpe": min_sharpe,
            "input_snapshot": dict(result["input_snapshot"]),
            "partitions": result["partitions"],
            "label_spec": dict(label_spec.__dict__),
            "split_spec": dict(split_spec.__dict__),
            "factor_provenance": factor_provenance,
            "gates": dict(result["gates"]),
            "thread_limit": thread_limit,
        }
    )

    _check_candidate_output(base_dir, candidate_dir)
    holdout_out.to_pickle(candidate_prediction_file)

    importance = _feature_importance(flat, all_features)
    _check_candidate_output(base_dir, candidate_dir)
    importance.to_csv(
        candidate_importance_file, index=False, encoding="utf-8-sig"
    )
    artifacts = {
        "artifact_schema_version": 3,
        "model_release_id": release_id,
        "training_run_id": run_id,
        "strategy_id": strategy_spec.strategy_id,
        "strategy_version": strategy_spec.version,
        "strategy_fingerprint": strategy_spec.fingerprint(),
        "model_base": flat,
        "model_active": None,
        "hierarchical_model": hierarchy,
        "selected_model_kind": deploy_kind,
        "scaler": IdentityScaler(),
        "features": all_features,
        "feature_groups": groups.as_dict(),
        "text_factor_metadata": text_factor_metadata,
        "gp_miner": gp_miner,
        "label_spec": label_spec.__dict__,
        "split_spec": split_spec.__dict__,
        "validation_metrics": validation_metrics,
        "validation_stability": validation_stability,
        "promotion_eligible": promotion_eligible,
        "release_gate_passed": release_gate_passed,
        "sealed_holdout_used_for_release_veto": True,
        "sealed_holdout_metrics": {"ic": holdout_ic},
        "input_snapshot": dict(result["input_snapshot"]),
        "partitions": result["partitions"],
        "factor_provenance": factor_provenance,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    manifest_path = _write_training_bundle(candidate_dir, artifacts, report, base_dir)
    # Catch edits occurring while model/evaluation artifacts were serialized as
    # well as edits during fitting. Keep a rejected, internally consistent
    # candidate instead of leaving a passing manifest behind.
    if not (_input_unchanged(factor_file, frozen_hash)
            and _input_unchanged(frozen_input_file, frozen_hash)):
        input_unchanged = False
        release_gate_passed = False
        result["input_snapshot"]["source_unchanged"] = False
        result["gates"].update(input_unchanged=False, release_gate_passed=False)
        report.update(release_gate_passed=False, input_snapshot=dict(result["input_snapshot"]),
                      gates=dict(result["gates"]))
        artifacts.update(release_gate_passed=False, input_snapshot=dict(result["input_snapshot"]))
        manifest_path = _write_training_bundle(candidate_dir, artifacts, report, base_dir)
    _check_candidate_output(base_dir, candidate_dir)
    result.update(
        status="candidate_ready" if release_gate_passed else "candidate_rejected",
        candidate_manifest=str(manifest_path.resolve()),
        factor_provenance=factor_provenance,
    )

    print(f"[Model] validation IC: {validation_metrics}")
    print(f"[Model] selected on validation: {deploy_kind}")
    print(f"[Model] promotion eligible (>=3/5 positive blocks): {promotion_eligible}")
    print(f"[Model] sealed holdout IC (reported once): {holdout_ic:.4f}")
    print(importance.head(15).to_string(index=False))
    if not release_gate_passed:
        print(
            "[Model] RELEASE BLOCKED: candidate kept separately; "
            "production artifacts were not overwritten"
        )
        print(f"[Model] candidate manifest: {manifest_path}")
        return result

    if candidate_only:
        print(f"[Model] candidate ready for separate review; not activated: {manifest_path}")
        return result

    manager = ReleaseManager(
        Path(base_dir) / "artifacts" / "models",
        legacy_model_path=model_file,
    )
    signature = manager.promote(
        manifest_path,
        project_legacy=True,
        legacy_artifact_name="model",
    )

    # These are compatibility views only. Runtime and backtest consumers read
    # the verified managed release, so an old view can never become authority.
    projections = {
        "predictions": prediction_file,
        "feature_importance": importance_file,
        "evaluation": report_projection,
    }
    for artifact_name, destination in projections.items():
        try:
            manager.project_active_artifact(artifact_name, destination)
        except Exception as exc:
            print(
                f"[Model WARNING] compatibility projection failed "
                f"({artifact_name}): {exc}"
            )
    print(
        "[Model] release gate passed; active release switched atomically: "
        f"{signature.release_id}"
    )
    result.update(status="published", published=True)
    return result
