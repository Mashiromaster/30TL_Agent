# -*- coding: utf-8 -*-
"""Slow macro state plus fast minute-level alpha model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from feature_registry import FeatureGroups, validate_feature_schema


class HierarchicalMacroMicroModel:
    """Fit macro information at daily frequency and micro alpha at bar frequency."""

    def __init__(self, micro_model=None, macro_model=None, target_col="Target_Ret"):
        self.micro_model = micro_model or LGBMRegressor(
            n_estimators=200,
            learning_rate=0.01,
            num_leaves=8,
            max_depth=3,
            min_child_samples=200,
            lambda_l1=10.0,
            lambda_l2=10.0,
            objective="regression_l1",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        self.macro_model = macro_model or Ridge(alpha=10.0)
        self.target_col = target_col

    @staticmethod
    def _medians(frame, columns):
        return {
            name: float(pd.to_numeric(frame[name], errors="coerce").median())
            if frame[name].notna().any() else 0.0
            for name in columns
        }

    @staticmethod
    def _matrix(frame, columns, medians):
        if not columns:
            return np.empty((len(frame), 0))
        return frame.loc[:, columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(medians).astype(float)

    def fit(self, train, validation=None, *, feature_groups: FeatureGroups):
        if self.target_col not in train:
            raise ValueError(f"missing target column: {self.target_col}")
        self.feature_groups_ = feature_groups
        self.micro_features_ = tuple(
            dict.fromkeys(
                feature_groups.micro
                + feature_groups.market
                + feature_groups.regime
                + feature_groups.gp
            )
        )
        self.macro_features_ = tuple(feature_groups.macro)
        validate_feature_schema(train, self.micro_features_ + self.macro_features_)
        target = pd.to_numeric(train[self.target_col], errors="coerce")
        valid = target.notna()
        fit_frame = train.loc[valid].copy()
        y = target.loc[valid].to_numpy(dtype=float)

        self.micro_medians_ = self._medians(fit_frame, self.micro_features_)
        if self.micro_features_:
            micro_x = self._matrix(
                fit_frame, self.micro_features_, self.micro_medians_
            )
            self.micro_model.fit(micro_x, y)
            micro_pred = np.asarray(self.micro_model.predict(micro_x), dtype=float)
            self.micro_constant_ = None
        else:
            self.micro_constant_ = float(np.mean(y))
            micro_pred = np.full(len(y), self.micro_constant_)

        residual = y - micro_pred
        self.macro_scale_ = 1.0
        self.macro_training_rows_ = 0
        self.macro_medians_ = {}
        self.has_macro_model_ = False
        if self.macro_features_:
            daily = fit_frame.loc[:, ["date", *self.macro_features_]].copy()
            daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
            daily["_day"] = daily["date"].dt.normalize()
            daily["_residual"] = residual
            aggregations = {name: "last" for name in self.macro_features_}
            aggregations["_residual"] = "mean"
            daily = daily.groupby("_day", sort=True).agg(aggregations).dropna(
                subset=["_residual"]
            )
            self.macro_training_rows_ = len(daily)
            if len(daily) >= 3:
                self.macro_medians_ = self._medians(daily, self.macro_features_)
                macro_x = self._matrix(
                    daily, self.macro_features_, self.macro_medians_
                )
                self.macro_model.fit(macro_x, daily["_residual"].to_numpy())
                fitted = np.asarray(self.macro_model.predict(macro_x), dtype=float)
                self.macro_scale_ = max(float(np.nanstd(fitted)), 1e-9)
                self.has_macro_model_ = True
        return self

    def components(self, df):
        validate_feature_schema(df, self.micro_features_ + self.macro_features_)
        if self.micro_features_:
            micro_alpha = np.asarray(
                self.micro_model.predict(
                    self._matrix(df, self.micro_features_, self.micro_medians_)
                ),
                dtype=float,
            )
        else:
            micro_alpha = np.full(len(df), self.micro_constant_)

        if self.has_macro_model_:
            macro_x = self._matrix(df, self.macro_features_, self.macro_medians_)
            macro_state = np.asarray(self.macro_model.predict(macro_x), dtype=float)
            changed = df.loc[:, self.macro_features_].ne(
                df.loc[:, self.macro_features_].shift(1)
            ).any(axis=1).to_numpy()
        else:
            macro_state = np.zeros(len(df))
            changed = np.zeros(len(df), dtype=bool)
        gate_weight = np.clip(
            1.0 + 0.25 * np.tanh(macro_state / self.macro_scale_), 0.5, 1.5
        )
        event_alpha = np.where(changed, macro_state, 0.0)
        prediction = micro_alpha * gate_weight + macro_state
        return pd.DataFrame(
            {
                "micro_alpha": micro_alpha,
                "macro_state": macro_state,
                "gate_weight": gate_weight,
                "event_alpha": event_alpha,
                "prediction": prediction,
            },
            index=df.index,
        )

    def predict(self, df):
        return self.components(df)["prediction"].to_numpy()
