# -*- coding: utf-8 -*-
"""Train-only symbolic factor discovery with validation-only selection."""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import validate_data

try:
    from gplearn.functions import abs1, log1, make_function, sig1, sqrt1
    from gplearn.genetic import SymbolicTransformer
except ImportError:  # pragma: no cover - exercised by deployment environments
    SymbolicTransformer = None


def _sign(values):
    return np.sign(values)


def _relu(values):
    return np.maximum(values, 0)


if SymbolicTransformer is not None:
    SIGN = make_function(function=_sign, name="sign", arity=1)
    RELU = make_function(function=_relu, name="relu", arity=1)
    FUNCTION_SET = ("add", "sub", "mul", "div", sqrt1, log1, sig1, abs1, SIGN, RELU)

    class CompatibleSymbolicTransformer(SymbolicTransformer):
        """Compatibility shim for gplearn 0.4.2 on newer scikit-learn."""

        def _validate_data(self, X, y=None, **kwargs):
            return validate_data(self, X, y, **kwargs)
else:
    FUNCTION_SET = ()
    CompatibleSymbolicTransformer = None


@dataclass(frozen=True)
class SelectedProgram:
    candidate_index: int
    name: str
    expression: str
    validation_ic: float
    complexity: int


def _rank_ic(left, right, min_samples=30):
    pair = pd.DataFrame({"left": left, "right": right}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(pair) < min_samples or pair["left"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(pair["left"], pair["right"]).statistic)


class GeneticFactorMiner:
    """Discover candidates on training data, then freeze validation winners."""

    def __init__(
        self,
        population_size=1500,
        generations=15,
        hall_of_fame=100,
        n_components=50,
        parsimony_coefficient=0.002,
        max_samples=0.7,
        random_state=42,
        n_jobs=-1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.hall_of_fame = min(hall_of_fame, population_size)
        self.n_components = min(n_components, self.hall_of_fame)
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.selected_programs_: tuple[SelectedProgram, ...] = ()

    @staticmethod
    def _require_dependency():
        if CompatibleSymbolicTransformer is None:
            raise RuntimeError(
                "gplearn is required for GP discovery; install project requirements"
            )

    def _validate_frame(self, X, *, fitting=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with a stable schema")
        non_numeric = [
            name for name in X.columns
            if not pd.api.types.is_numeric_dtype(X[name])
        ]
        if non_numeric:
            raise ValueError(f"GP inputs must be numeric: {non_numeric}")
        names = list(X.columns)
        if fitting:
            self.feature_names_in_ = names
        elif names != self.feature_names_in_:
            raise ValueError(
                f"GP feature schema/order changed: expected {self.feature_names_in_}, got {names}"
            )
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y):
        """Discover symbolic candidates using discovery-train data only."""
        self._require_dependency()
        matrix = self._validate_frame(X, fitting=True)
        target = pd.to_numeric(pd.Series(y, index=X.index), errors="coerce")
        valid = target.notna().to_numpy()
        if valid.sum() < 100:
            raise ValueError("GP discovery requires at least 100 labelled rows")
        self.scaler_ = StandardScaler().fit(matrix[valid])
        scaled = self.scaler_.transform(matrix[valid])
        self.transformer_ = CompatibleSymbolicTransformer(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=min(20, self.population_size),
            hall_of_fame=self.hall_of_fame,
            n_components=self.n_components,
            function_set=FUNCTION_SET,
            metric="spearman",
            parsimony_coefficient=self.parsimony_coefficient,
            max_samples=self.max_samples,
            init_depth=(2, 5),
            const_range=(-3.0, 3.0),
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            stopping_criteria=0.2,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0,
            feature_names=self.feature_names_in_,
        )
        self.transformer_.fit(scaled, target.to_numpy()[valid])
        self.selected_programs_ = ()
        return self

    def _candidate_frame(self, X):
        if not hasattr(self, "transformer_"):
            raise RuntimeError("call fit on discovery-train data first")
        matrix = self._validate_frame(X)
        scaled = self.scaler_.transform(matrix)
        values = self.transformer_.transform(scaled)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(
            values,
            index=X.index,
            columns=[f"GP_CANDIDATE_{index:03d}" for index in range(values.shape[1])],
        )

    def select(
        self,
        X_validation: pd.DataFrame,
        y_validation,
        *,
        min_abs_ic=0.01,
        max_factors=20,
        max_correlation=0.9,
    ):
        """Select and freeze candidates using validation data only."""
        candidates = self._candidate_frame(X_validation)
        target = pd.to_numeric(
            pd.Series(y_validation, index=X_validation.index), errors="coerce"
        )
        programs = list(self.transformer_._best_programs)
        ranked = []
        for index, name in enumerate(candidates.columns):
            ic = _rank_ic(candidates[name], target)
            if np.isnan(ic) or abs(ic) < min_abs_ic:
                continue
            expression = str(programs[index])
            ranked.append((index, ic, len(expression), expression))
        ranked.sort(key=lambda item: (-abs(item[1]), item[2]))

        selected = []
        selected_values = []
        for index, ic, complexity, expression in ranked:
            value = candidates.iloc[:, index]
            if any(abs(value.corr(previous, method="spearman")) >= max_correlation
                   for previous in selected_values):
                continue
            selected.append(
                SelectedProgram(
                    candidate_index=index,
                    name=f"GP_{len(selected):03d}",
                    expression=expression,
                    validation_ic=round(ic, 6),
                    complexity=complexity,
                )
            )
            selected_values.append(value)
            if len(selected) >= max_factors:
                break
        self.selected_programs_ = tuple(selected)
        return self

    def transform(self, X: pd.DataFrame):
        """Apply the frozen training scaler and selected programs."""
        if not self.selected_programs_:
            raise RuntimeError("call select on validation data before transform")
        candidates = self._candidate_frame(X)
        return pd.DataFrame(
            {
                selected.name: candidates.iloc[:, selected.candidate_index]
                for selected in self.selected_programs_
            },
            index=X.index,
        )

    def report(self):
        return {
            "selection_basis": "validation_only",
            "feature_names": list(self.feature_names_in_),
            "programs": [program.__dict__ for program in self.selected_programs_],
        }

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)


def _gp_input_columns(frame, target_col):
    excluded = {
        target_col, "date", "trade_dt", "ticker", "open", "high", "low",
        "close", "adj_close", "volume", "money", "oi", "label_start_ts",
        "label_end_ts", "label_valid", "Hour", "Minute", "Minute_of_Day",
    }
    return [
        name for name in frame.select_dtypes(include=[np.number, "bool"]).columns
        if name not in excluded and not name.startswith(("_", "GP_"))
        and frame[name].dropna().nunique() > 1
    ]


def mine_genetic_factors(
    discovery_train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target_col="Target_Ret",
    population_size=1500,
    generations=15,
    max_new_factors=20,
    min_ic_threshold=0.01,
):
    """Fit on discovery train, select on validation, and return a frozen miner."""
    if target_col not in discovery_train or target_col not in validation:
        raise ValueError("both discovery_train and validation need a pre-built target")
    columns = _gp_input_columns(discovery_train, target_col)
    missing = [name for name in columns if name not in validation]
    if missing:
        raise ValueError(f"validation is missing GP inputs: {missing}")
    miner = GeneticFactorMiner(
        population_size=population_size,
        generations=generations,
        hall_of_fame=max(max_new_factors * 3, max_new_factors),
        n_components=max(max_new_factors * 2, max_new_factors),
    )
    miner.fit(discovery_train[columns], discovery_train[target_col])
    miner.select(
        validation[columns],
        validation[target_col],
        min_abs_ic=min_ic_threshold,
        max_factors=max_new_factors,
    )
    return miner, miner.report()


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe GP factor discovery")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", type=int, default=1500)
    parser.add_argument("--generations", type=int, default=15)
    args = parser.parse_args()
    from training_data import LabelSpec, SplitSpec, build_forward_labels, split_train_validation_holdout

    labelled = build_forward_labels(pd.read_pickle(args.data), LabelSpec())
    train, validation, _ = split_train_validation_holdout(labelled, SplitSpec())
    miner, report = mine_genetic_factors(
        train,
        validation,
        population_size=args.population,
        generations=args.generations,
    )
    miner.save(args.output)
    print(report)


if __name__ == "__main__":
    main()
