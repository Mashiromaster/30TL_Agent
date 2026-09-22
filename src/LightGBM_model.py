# -*- coding: utf-8 -*-
"""Leakage-safe LightGBM training with macro/micro ablations."""

from __future__ import annotations

import json
import os
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from scipy import stats

from artifact_audit import extract_text_factor_metadata
from feature_registry import FeatureGroups, discover_feature_groups
from hierarchical_model import HierarchicalMacroMicroModel
from jev_evaluation import load_semantic_gate_status
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


def _strategy_release_gate(
    metrics,
    *,
    max_drawdown_limit=BASELINE_MAX_DRAWDOWN,
    min_total_return=BASELINE_MIN_TOTAL_RETURN,
    min_sharpe=BASELINE_MIN_SHARPE,
):
    """Cost-adjusted operational and performance gate for production release."""
    if not 0 < float(max_drawdown_limit) <= BASELINE_MAX_DRAWDOWN:
        raise ValueError(
            "max_drawdown_limit must be in (0, 0.35]; the baseline can only "
            "be tightened"
        )
    if float(min_total_return) < BASELINE_MIN_TOTAL_RETURN:
        raise ValueError("min_total_return cannot weaken the zero baseline")
    if float(min_sharpe) < BASELINE_MIN_SHARPE:
        raise ValueError("min_sharpe cannot weaken the zero baseline")
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
        n_jobs=-1,
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


def _fit_flat(train, validation, features, sample_weight=None):
    if not features:
        return None
    return train_model(
        train[features],
        train["Target_Ret"].to_numpy(),
        validation[features],
        validation["Target_Ret"].to_numpy(),
        sample_weight=sample_weight,
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


def run_process(
    base_dir,
    max_lookback_months=9,
    time_decay_half_life=60,
    *,
    strategy_id=None,
):
    print("\n" + "=" * 50)
    print("STEP 3: point-in-time model training")
    print("=" * 50)
    strategy_spec = get_strategy_spec(strategy_id)
    run_id, release_id = _release_identifiers(strategy_spec.fingerprint())
    candidate_dir = Path(base_dir) / "artifacts" / "experiments" / run_id

    factor_file = os.path.join(base_dir, "outputs", "df_factors.pkl")
    prediction_file = os.path.join(base_dir, "outputs", "df_predictions.pkl")
    model_file = os.path.join(base_dir, "models", "trained_model.pkl")
    importance_file = os.path.join(base_dir, "outputs", "feature_importance.csv")
    report_projection = os.path.join(
        base_dir, "outputs", "model_ablation_report.json"
    )
    candidate_prediction_file = candidate_dir / "df_predictions.pkl"
    candidate_model_file = candidate_dir / "trained_model.pkl"
    candidate_importance_file = candidate_dir / "feature_importance.csv"
    report_file = candidate_dir / "evaluation.json"
    if not os.path.exists(factor_file):
        print(f"[Model ERROR] missing factor file: {factor_file}")
        return False
    candidate_dir.mkdir(parents=True, exist_ok=False)
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    raw = pd.read_pickle(factor_file)
    raw, text_factor_metadata = _apply_jev_semantic_gate(raw, base_dir)
    label_spec = LabelSpec(horizon_bars=30, entry_lag_bars=1, price_col="adj_close")
    model_frame, _ = prepare_training_data(raw, label_spec=label_spec)
    split_spec = SplitSpec(validation_days=60, sealed_holdout_days=60, min_train_days=120)
    train, validation, holdout = split_train_validation_holdout(model_frame, split_spec)
    if max_lookback_months:
        cutoff = validation["date"].min() - pd.Timedelta(days=max_lookback_months * 30)
        train = train.loc[train["date"] >= cutoff].copy()

    gp_miner = None
    gp_report = None
    gp_enabled = os.environ.get("F_AGENT_GENETIC_MINING", "0").strip().lower() in {
        "1", "true", "yes"
    }
    if gp_enabled:
        from genetic_factor_miner import mine_genetic_factors

        gp_miner, gp_report = mine_genetic_factors(train, validation)
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

    flat = _fit_flat(train, validation, all_features, weights)
    micro = _fit_flat(train, validation, micro_features, weights)
    macro = _fit_flat(train, validation, macro_features, weights) if macro_features else None
    hierarchy = HierarchicalMacroMicroModel().fit(
        train, validation, feature_groups=groups
    )

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
    max_drawdown_limit = float(
        os.environ.get("F_AGENT_RELEASE_MAX_DRAWDOWN", "0.35")
    )
    min_total_return = float(
        os.environ.get("F_AGENT_RELEASE_MIN_TOTAL_RETURN", "0.0")
    )
    min_sharpe = float(os.environ.get("F_AGENT_RELEASE_MIN_SHARPE", "0.0"))
    strategy_gate_passed, strategy_gate_reason = _strategy_release_gate(
        strategy_metrics,
        max_drawdown_limit=max_drawdown_limit,
        min_total_return=min_total_return,
        min_sharpe=min_sharpe,
    )
    release_gate_passed = bool(release_gate_passed and strategy_gate_passed)
    report.update(
        {
            "release_gate_passed": release_gate_passed,
            "strategy_backtest": strategy_metrics,
            "strategy_gate_passed": strategy_gate_passed,
            "strategy_gate_reason": strategy_gate_reason,
            "max_drawdown_limit": max_drawdown_limit,
            "min_total_return": min_total_return,
            "min_sharpe": min_sharpe,
        }
    )

    holdout_out.to_pickle(candidate_prediction_file)

    importance = _feature_importance(flat, all_features)
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
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    with open(candidate_model_file, "wb") as handle:
        pickle.dump(artifacts, handle)
    report["gate_attestation"] = {
        "schema_version": 1,
        "release_id": release_id,
        "run_id": run_id,
        "strategy_id": strategy_spec.strategy_id,
        "strategy_fingerprint": strategy_spec.fingerprint(),
        "model_sha256": sha256_file(candidate_model_file),
        "gate_passed": release_gate_passed,
        "policy": {
            "min_observation_days": BASELINE_MIN_OBSERVATION_DAYS,
            "min_total_return": min_total_return,
            "min_sharpe": min_sharpe,
            "max_drawdown_limit": max_drawdown_limit,
        },
    }
    temporary_report = report_file.with_suffix(".json.tmp")
    try:
        with temporary_report.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        os.replace(temporary_report, report_file)
    finally:
        if temporary_report.exists():
            temporary_report.unlink()

    manifest_path = write_candidate_manifest(
        candidate_dir,
        release_id=release_id,
        run_id=run_id,
        artifacts={
            "model": candidate_model_file.name,
            "predictions": candidate_prediction_file.name,
            "feature_importance": candidate_importance_file.name,
            "evaluation": report_file.name,
        },
        metadata={
            "strategy_id": strategy_spec.strategy_id,
            "strategy_version": strategy_spec.version,
            "strategy_fingerprint": strategy_spec.fingerprint(),
            "release_gate_passed": release_gate_passed,
            "strategy_gate_reason": strategy_gate_reason,
        },
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
        return False

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
    return True
