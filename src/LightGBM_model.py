# -*- coding: utf-8 -*-
"""Leakage-safe LightGBM training with macro/micro ablations."""

from __future__ import annotations

import json
import os
import pickle
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from scipy import stats

from feature_registry import FeatureGroups, discover_feature_groups
from hierarchical_model import HierarchicalMacroMicroModel
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


def prepare_training_data(df, prediction_horizon=30, label_spec=None):
    """Build one label definition and discover a validated feature schema."""
    spec = label_spec or LabelSpec(horizon_bars=prediction_horizon)
    labelled = build_forward_labels(df, spec)
    groups = discover_feature_groups(labelled)
    features = list(groups.all_features)
    model = labelled.loc[labelled["label_valid"]].copy()
    model.attrs["feature_groups"] = groups.as_dict()
    model.attrs["label_spec"] = spec.__dict__
    print(
        f"[Model] features={len(features)} "
        f"micro={len(groups.micro)} market={len(groups.market)} "
        f"macro={len(groups.macro)} regime={len(groups.regime)} gp={len(groups.gp)}"
    )
    return model, features


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


def run_process(base_dir, max_lookback_months=9, time_decay_half_life=60):
    print("\n" + "=" * 50)
    print("STEP 3: point-in-time model training")
    print("=" * 50)
    factor_file = os.path.join(base_dir, "outputs/df_factors.pkl")
    prediction_file = os.path.join(base_dir, "outputs/df_predictions.pkl")
    model_file = os.path.join(base_dir, "models/trained_model.pkl")
    importance_file = os.path.join(base_dir, "outputs/feature_importance.csv")
    candidate_prediction_file = os.path.join(
        base_dir, "outputs/df_predictions_candidate.pkl"
    )
    candidate_model_file = os.path.join(
        base_dir, "models/trained_model_candidate.pkl"
    )
    candidate_importance_file = os.path.join(
        base_dir, "outputs/feature_importance_candidate.csv"
    )
    report_file = os.path.join(base_dir, "outputs/model_ablation_report.json")
    if not os.path.exists(factor_file):
        print(f"[Model ERROR] missing factor file: {factor_file}")
        return False
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    raw = pd.read_pickle(factor_file)
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
        "gp": gp_report,
    }
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    holdout_out = holdout.copy()
    holdout_out["Pred_Ret"] = holdout_prediction
    holdout_out["Model_Used"] = deploy_kind
    if deploy_kind == "hierarchical":
        components = hierarchy.components(holdout)
        for name in components:
            holdout_out[name] = components[name]
    output_cols = [
        "date", "trade_dt", "ticker", "close", "adj_close", "Target_Ret",
        "Pred_Ret", "Market_Regime", "Model_Used", "micro_alpha", "macro_state",
        "gate_weight", "event_alpha",
    ]
    holdout_out[[name for name in output_cols if name in holdout_out]].to_pickle(
        candidate_prediction_file
    )

    importance = _feature_importance(flat, all_features)
    importance.to_csv(
        candidate_importance_file, index=False, encoding="utf-8-sig"
    )
    artifacts = {
        "artifact_schema_version": 2,
        "model_base": flat,
        "model_active": None,
        "hierarchical_model": hierarchy,
        "selected_model_kind": deploy_kind,
        "scaler": IdentityScaler(),
        "features": all_features,
        "feature_groups": groups.as_dict(),
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
        return False

    os.replace(candidate_prediction_file, prediction_file)
    os.replace(candidate_importance_file, importance_file)
    os.replace(candidate_model_file, model_file)
    print("[Model] release gate passed; production artifacts promoted atomically")
    return True
