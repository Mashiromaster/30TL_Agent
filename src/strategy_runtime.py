# -*- coding: utf-8 -*-
"""Shared TL strategy contract, deterministic signal policy, and model runtime.

The module deliberately owns no persistence and performs no training.  It is a
small common core for CLI, ResearchAgent, dashboards, and backtests:

* :class:`StrategySpec` versions every rule that turns predictions into intent.
* :class:`SignalPolicy` is a prefix-stable transformation of predictions.
* :class:`ModelRuntime` loads an existing artifact and only performs inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


SignalKind = Literal["regime_breakout", "rank_band"]
ThresholdWindow = Literal["rolling", "expanding"]


@dataclass(frozen=True)
class StrategySpec:
    """Immutable, fingerprinted contract for one deterministic TL strategy."""

    strategy_id: str
    version: int
    signal_kind: SignalKind

    prediction_col: str = "Pred_Ret"
    timestamp_col: str = "date"
    price_col: str = "close"
    regime_col: str = "Market_Regime"
    model_used_col: str = "Model_Used"

    smooth_span: int = 160
    lookback_bars: int = 480
    min_periods: int = 100
    threshold_window: ThresholdWindow = "rolling"
    default_upper: float = 0.90
    default_lower: float = 0.05
    regime_thresholds: tuple[tuple[int, float, float], ...] = (
        (0, 0.90, 0.05),
        (1, 0.80, 0.20),
        (2, 0.80, 0.20),
    )

    short_band: tuple[float, float] = (0.15, 0.30)
    long_band: tuple[float, float] = (0.70, 0.85)

    confirm_bars: int = 0
    regime_confirm_bars: tuple[tuple[int, int], ...] = ()
    confirmation_ratio: float = 1.0

    regime_min_hold_bars: tuple[tuple[int, int], ...] = ()
    regime_cooldown_bars: tuple[tuple[int, int], ...] = ()
    regime_weights: tuple[tuple[int, float], ...] = (
        (0, 1.0),
        (1, 0.8),
        (2, 0.8),
    )
    max_hold_multiplier: float = 1.5
    confidence_scaled_weight: bool = False

    execution_lag_bars: int = 1
    commission_rate: float = 0.000023
    tick_size: float = 0.01
    slippage_ticks: float = 1.0

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must not be empty")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if self.smooth_span <= 0:
            raise ValueError("smooth_span must be positive")
        if self.lookback_bars <= 0 or self.min_periods <= 0:
            raise ValueError("lookback_bars and min_periods must be positive")
        if self.min_periods > self.lookback_bars and self.threshold_window == "rolling":
            raise ValueError("min_periods cannot exceed rolling lookback_bars")
        if self.confirm_bars < 0:
            raise ValueError("confirm_bars must be non-negative")
        if not 0 < self.confirmation_ratio <= 1:
            raise ValueError("confirmation_ratio must be in (0, 1]")
        if self.execution_lag_bars < 0:
            raise ValueError("execution_lag_bars must be non-negative")
        for lower, upper, name in (
            (*self.short_band, "short_band"),
            (*self.long_band, "long_band"),
        ):
            if not 0 <= lower < upper <= 1:
                raise ValueError(f"{name} must satisfy 0 <= lower < upper <= 1")

    def fingerprint(self) -> str:
        """Return a stable semantic fingerprint, independent of process/order."""
        payload = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{sha256(payload).hexdigest()}"


DEFAULT_STRATEGY_ID = "tl_regime_breakout_v1"


_REGIME_BREAKOUT = StrategySpec(
    strategy_id=DEFAULT_STRATEGY_ID,
    version=1,
    signal_kind="regime_breakout",
    smooth_span=160,
    lookback_bars=480,
    min_periods=100,
    threshold_window="rolling",
    regime_thresholds=((0, 0.90, 0.05), (1, 0.80, 0.20), (2, 0.80, 0.20)),
    regime_confirm_bars=((0, 60), (1, 25), (2, 50)),
    confirmation_ratio=0.70,
    regime_min_hold_bars=((0, 90), (1, 60), (2, 30)),
    regime_cooldown_bars=((0, 180), (1, 140), (2, 60)),
    confidence_scaled_weight=False,
    execution_lag_bars=1,
)

_INFERENCE_BREAKOUT_LEGACY = StrategySpec(
    strategy_id="tl_inference_breakout_legacy_v1",
    version=1,
    signal_kind="regime_breakout",
    smooth_span=160,
    lookback_bars=480,
    min_periods=100,
    threshold_window="expanding",
    regime_thresholds=((0, 0.90, 0.05), (1, 0.90, 0.05), (2, 0.90, 0.05)),
    confirm_bars=0,
    confirmation_ratio=1.0,
    confidence_scaled_weight=True,
    execution_lag_bars=1,
)

_RANK_SWEETSPOT_LEGACY = StrategySpec(
    strategy_id="tl_rank_sweetspot_legacy_v1",
    version=1,
    signal_kind="rank_band",
    smooth_span=60,
    lookback_bars=480,
    min_periods=480,
    threshold_window="rolling",
    short_band=(0.15, 0.30),
    long_band=(0.70, 0.85),
    confirm_bars=10,
    confirmation_ratio=1.0,
    confidence_scaled_weight=True,
    execution_lag_bars=1,
)

STRATEGY_SPECS: Mapping[str, StrategySpec] = MappingProxyType(
    {
        _REGIME_BREAKOUT.strategy_id: _REGIME_BREAKOUT,
        _INFERENCE_BREAKOUT_LEGACY.strategy_id: _INFERENCE_BREAKOUT_LEGACY,
        _RANK_SWEETSPOT_LEGACY.strategy_id: _RANK_SWEETSPOT_LEGACY,
    }
)


def get_strategy_spec(strategy_id: str | None = None) -> StrategySpec:
    """Resolve a registered strategy by id; omitted means the canonical spec."""
    resolved = strategy_id or DEFAULT_STRATEGY_ID
    try:
        return STRATEGY_SPECS[resolved]
    except KeyError as exc:
        choices = ", ".join(sorted(STRATEGY_SPECS))
        raise ValueError(f"unknown strategy_id {resolved!r}; available: {choices}") from exc


@dataclass(frozen=True)
class SignalDecision:
    """Typed latest-bar decision with policy and model provenance."""

    as_of: pd.Timestamp
    effective_from: pd.Timestamp | None
    execution_lag_bars: int
    direction: int
    target_weight: float
    confidence: float
    raw_direction: int
    confirmed_direction: int
    pred_rank_pct: float | None
    predicted_return: float
    predicted_return_smooth: float
    market_regime: int
    model_used: str
    close: float | None
    upper_threshold: float | None
    lower_threshold: float | None
    strategy_id: str
    strategy_version: int
    strategy_fingerprint: str
    model_release_id: str
    decision_id: str
    decision_ready: bool
    missing_reason: str

    def to_dict(self) -> dict[str, Any]:
        regime_names = {0: "正常", 1: "高波动", 2: "趋势"}
        direction_names = {1: "做多", -1: "做空", 0: "观望"}
        return {
            "timestamp": str(self.as_of),
            "effective_from": (
                str(self.effective_from) if self.effective_from is not None else None
            ),
            "execution_lag_bars": self.execution_lag_bars,
            "close": self.close,
            "market_regime": self.market_regime,
            "regime_name": regime_names.get(self.market_regime, "未知"),
            "predicted_return": self.predicted_return,
            "predicted_return_smooth": self.predicted_return_smooth,
            "pred_rank_pct": self.pred_rank_pct,
            "raw_direction": self.raw_direction,
            "confirmed_direction": self.confirmed_direction,
            "direction": self.direction,
            "direction_name": direction_names[self.direction],
            "confidence": self.confidence,
            "model_used": self.model_used,
            "suggested_weight": self.target_weight,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "strategy_fingerprint": self.strategy_fingerprint,
            "model_release_id": self.model_release_id,
            "decision_id": self.decision_id,
            "decision_ready": self.decision_ready,
            "missing_reason": self.missing_reason,
        }


class SignalPolicy:
    """Prefix-stable mapping from predictions to executable position intent."""

    def __init__(self, spec: StrategySpec):
        self.spec = spec

    @staticmethod
    def _mapping(items: tuple[tuple[Any, Any], ...]) -> dict[Any, Any]:
        return dict(items)

    def _validate_frame(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions is None or predictions.empty:
            raise ValueError("prediction frame must not be empty")
        required = {self.spec.timestamp_col, self.spec.prediction_col}
        missing = sorted(required.difference(predictions.columns))
        if missing:
            raise ValueError(f"prediction frame missing columns: {missing}")

        frame = predictions.copy()
        timestamps = pd.to_datetime(frame[self.spec.timestamp_col], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("timestamp column contains invalid values")
        if timestamps.duplicated().any():
            raise ValueError("timestamp column contains duplicate values")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("timestamp column must be strictly increasing")
        frame[self.spec.timestamp_col] = timestamps

        numeric_prediction = pd.to_numeric(
            frame[self.spec.prediction_col], errors="coerce"
        )
        if numeric_prediction.isna().any() or not np.isfinite(numeric_prediction).all():
            raise ValueError("prediction column must contain finite numeric values")
        frame[self.spec.prediction_col] = numeric_prediction.astype(float)
        return frame

    def _regimes(self, frame: pd.DataFrame) -> pd.Series:
        if self.spec.regime_col not in frame:
            return pd.Series(0, index=frame.index, dtype=int)
        regime = pd.to_numeric(frame[self.spec.regime_col], errors="coerce").fillna(0)
        return regime.astype(int)

    def _breakout_signals(
        self, smoothed: pd.Series, regimes: pd.Series
    ) -> tuple[pd.Series, pd.Series, np.ndarray, pd.Series]:
        lagged = smoothed.shift(1)
        upper = pd.Series(np.nan, index=smoothed.index, dtype=float)
        lower = pd.Series(np.nan, index=smoothed.index, dtype=float)
        threshold_map = {
            regime: (upper_q, lower_q)
            for regime, upper_q, lower_q in self.spec.regime_thresholds
        }

        for regime in regimes.unique():
            upper_q, lower_q = threshold_map.get(
                int(regime), (self.spec.default_upper, self.spec.default_lower)
            )
            if self.spec.threshold_window == "expanding":
                upper_values = lagged.expanding(
                    min_periods=self.spec.min_periods
                ).quantile(upper_q)
                lower_values = lagged.expanding(
                    min_periods=self.spec.min_periods
                ).quantile(lower_q)
            else:
                upper_values = lagged.rolling(
                    self.spec.lookback_bars, min_periods=self.spec.min_periods
                ).quantile(upper_q)
                lower_values = lagged.rolling(
                    self.spec.lookback_bars, min_periods=self.spec.min_periods
                ).quantile(lower_q)
            mask = regimes.eq(regime)
            upper.loc[mask] = upper_values.loc[mask]
            lower.loc[mask] = lower_values.loc[mask]

        raw = np.zeros(len(smoothed), dtype=int)
        raw[(smoothed > upper).fillna(False).to_numpy()] = 1
        raw[(smoothed < lower).fillna(False).to_numpy()] = -1

        prefix_std = smoothed.expanding(min_periods=2).std()
        distance = pd.Series(0.0, index=smoothed.index, dtype=float)
        long_mask = raw == 1
        short_mask = raw == -1
        distance.loc[long_mask] = (
            smoothed.loc[long_mask] - upper.loc[long_mask]
        )
        distance.loc[short_mask] = (
            lower.loc[short_mask] - smoothed.loc[short_mask]
        )
        confidence = (distance / prefix_std.replace(0.0, np.nan)).clip(0.0, 1.0)
        return upper, lower, raw, confidence.fillna(0.0)

    def _rank_signals(
        self, smoothed: pd.Series
    ) -> tuple[pd.Series, pd.Series, np.ndarray, pd.Series]:
        rank = pd.Series(np.nan, index=smoothed.index, dtype=float)
        for offset in range(len(smoothed)):
            start = max(0, offset - self.spec.lookback_bars)
            history = smoothed.iloc[start:offset]
            if len(history) < self.spec.min_periods:
                continue
            rank.iloc[offset] = float((history < smoothed.iloc[offset]).mean())

        raw = np.zeros(len(smoothed), dtype=int)
        short_lower, short_upper = self.spec.short_band
        long_lower, long_upper = self.spec.long_band
        raw[
            ((rank >= short_lower) & (rank < short_upper)).fillna(False).to_numpy()
        ] = -1
        raw[
            ((rank >= long_lower) & (rank < long_upper)).fillna(False).to_numpy()
        ] = 1
        confidence = ((rank - 0.5).abs() * 2.0).clip(0.0, 1.0).fillna(0.0)
        return rank, raw, confidence

    def _confirm(self, raw: np.ndarray, regimes: pd.Series) -> np.ndarray:
        confirmed = np.zeros(len(raw), dtype=int)
        confirm_by_regime = self._mapping(self.spec.regime_confirm_bars)
        for offset, raw_value in enumerate(raw):
            window = int(
                confirm_by_regime.get(
                    int(regimes.iloc[offset]), self.spec.confirm_bars
                )
            )
            if window <= 0:
                confirmed[offset] = int(raw_value)
                continue
            if offset < window:
                continue
            history = raw[offset - window:offset]
            non_zero = history[history != 0]
            required = int(np.ceil(window * self.spec.confirmation_ratio))
            if len(non_zero) >= required and len(np.unique(non_zero)) == 1:
                confirmed[offset] = int(non_zero[-1])
        return confirmed

    def _target_positions(
        self,
        confirmed: np.ndarray,
        confidence: pd.Series,
        regimes: pd.Series,
    ) -> np.ndarray:
        weights = self._mapping(self.spec.regime_weights)
        min_holds = self._mapping(self.spec.regime_min_hold_bars)
        cooldowns = self._mapping(self.spec.regime_cooldown_bars)

        # Stateless legacy policies express the current intent directly.  The
        # state machine below is reserved for specs that explicitly define
        # holding or cooldown rules.
        if not min_holds and not cooldowns:
            target = np.zeros(len(confirmed), dtype=float)
            for offset, signal in enumerate(confirmed):
                weight = float(weights.get(int(regimes.iloc[offset]), 1.0))
                if self.spec.confidence_scaled_weight:
                    weight *= float(confidence.iloc[offset])
                target[offset] = int(signal) * weight
            return target

        target = np.zeros(len(confirmed), dtype=float)
        current_direction = 0
        current_weight = 0.0
        current_regime = 0
        hold_start = 0
        last_trade = -1_000_000

        for offset, signal in enumerate(confirmed):
            regime = int(regimes.iloc[offset])
            min_hold = int(min_holds.get(regime, 0))
            cooldown = int(cooldowns.get(regime, 0))
            hold_time = offset - hold_start if current_direction else min_hold
            can_change = hold_time >= min_hold
            in_cooldown = (offset - last_trade) < cooldown
            regime_changed = regime != current_regime

            if (not in_cooldown and can_change) or regime_changed:
                if signal:
                    weight = float(weights.get(regime, 1.0))
                    if self.spec.confidence_scaled_weight:
                        weight *= float(confidence.iloc[offset])
                    if signal != current_direction or abs(weight - current_weight) > 0.1:
                        current_direction = int(signal)
                        current_weight = weight
                        current_regime = regime
                        hold_start = offset
                        last_trade = offset
                elif current_direction:
                    max_hold = int(np.ceil(min_hold * self.spec.max_hold_multiplier))
                    if min_hold == 0 or hold_time >= max_hold:
                        current_direction = 0
                        current_weight = 0.0
                        last_trade = offset

            target[offset] = current_direction * current_weight
        return target

    def evaluate_frame(
        self,
        predictions: pd.DataFrame,
        *,
        evaluation_start: pd.Timestamp | str | None = None,
        model_release_id: str = "legacy-unversioned",
    ) -> pd.DataFrame:
        """Evaluate the whole ordered frame without consulting future rows."""
        frame = self._validate_frame(predictions)
        regimes = self._regimes(frame)
        frame[self.spec.regime_col] = regimes
        smoothed = frame[self.spec.prediction_col].ewm(
            span=self.spec.smooth_span, adjust=False
        ).mean()

        frame["Pred_Smooth"] = smoothed
        if self.spec.signal_kind == "rank_band":
            rank, raw, confidence = self._rank_signals(smoothed)
            frame["Pred_Rank"] = rank
            frame["Upper_Q"] = self.spec.long_band[0]
            frame["Lower_Q"] = self.spec.short_band[1]
        else:
            upper, lower, raw, confidence = self._breakout_signals(
                smoothed, regimes
            )
            frame["Pred_Rank"] = np.nan
            frame["Upper_Q"] = upper
            frame["Lower_Q"] = lower

        if self.spec.signal_kind == "rank_band":
            decision_ready = frame["Pred_Rank"].notna()
        else:
            decision_ready = frame["Upper_Q"].notna() & frame["Lower_Q"].notna()
        frame["Decision_Ready"] = decision_ready
        frame["Missing_Reason"] = np.where(
            decision_ready, "", "insufficient_policy_history"
        )

        confirmed = self._confirm(raw, regimes)
        target = self._target_positions(confirmed, confidence, regimes)

        frame["Raw_Signal"] = raw
        frame["Signal_Confirmed"] = confirmed
        frame["Confidence"] = confidence.to_numpy(dtype=float)
        frame["Target_Position"] = target
        frame["Target_Weight"] = np.abs(target)
        frame["Direction"] = np.sign(target).astype(int)
        # Compatibility aliases consumed by existing charts and memory exports.
        frame["Signal"] = frame["Direction"]
        frame["Confirmed_Signal"] = frame["Signal_Confirmed"]
        frame["Effective_Position"] = (
            frame["Target_Position"]
            .shift(self.spec.execution_lag_bars)
            .fillna(0.0)
        )
        frame["Execution_Lag_Bars"] = self.spec.execution_lag_bars
        timestamps = frame[self.spec.timestamp_col]
        if self.spec.execution_lag_bars == 0:
            frame["Effective_From"] = timestamps
        else:
            # The next tradable bar cannot be inferred safely from wall-clock
            # medians across lunch breaks, weekends, holidays, or suspensions.
            # Keep it unknown and expose the explicit bar lag instead.
            frame["Effective_From"] = pd.NaT
        frame["Strategy_ID"] = self.spec.strategy_id
        frame["Strategy_Version"] = self.spec.version
        frame["Strategy_Fingerprint"] = self.spec.fingerprint()
        frame["Model_Release_ID"] = model_release_id

        if evaluation_start is not None:
            boundary = pd.Timestamp(evaluation_start)
            frame = frame.loc[frame[self.spec.timestamp_col] >= boundary].copy()
        return frame

    def latest(
        self,
        predictions: pd.DataFrame,
        *,
        model_release_id: str = "legacy-unversioned",
    ) -> SignalDecision:
        evaluated = self.evaluate_frame(
            predictions, model_release_id=model_release_id
        )
        latest = evaluated.iloc[-1]

        def optional_float(value: Any) -> float | None:
            return None if pd.isna(value) else float(value)

        target_position = float(latest["Target_Position"])
        regime_value = latest.get(self.spec.regime_col, 0)
        market_regime = 0 if pd.isna(regime_value) else int(regime_value)
        close = optional_float(latest.get(self.spec.price_col, np.nan))
        effective_from = latest.get("Effective_From")
        if pd.isna(effective_from):
            effective_from = None
        else:
            effective_from = pd.Timestamp(effective_from)
        strategy_fingerprint = self.spec.fingerprint()
        decision_id = self.decision_id_for_evaluated_row(
            latest, model_release_id=model_release_id
        )
        return SignalDecision(
            as_of=pd.Timestamp(latest[self.spec.timestamp_col]),
            effective_from=effective_from,
            execution_lag_bars=self.spec.execution_lag_bars,
            direction=int(np.sign(target_position)),
            target_weight=abs(target_position),
            confidence=float(latest["Confidence"]),
            raw_direction=int(latest["Raw_Signal"]),
            confirmed_direction=int(latest["Signal_Confirmed"]),
            pred_rank_pct=optional_float(latest["Pred_Rank"]),
            predicted_return=float(latest[self.spec.prediction_col]),
            predicted_return_smooth=float(latest["Pred_Smooth"]),
            market_regime=market_regime,
            model_used=str(latest.get(self.spec.model_used_col, "base")),
            close=close,
            upper_threshold=optional_float(latest["Upper_Q"]),
            lower_threshold=optional_float(latest["Lower_Q"]),
            strategy_id=self.spec.strategy_id,
            strategy_version=self.spec.version,
            strategy_fingerprint=strategy_fingerprint,
            model_release_id=model_release_id,
            decision_id=decision_id,
            decision_ready=bool(latest["Decision_Ready"]),
            missing_reason=str(latest["Missing_Reason"]),
        )

    def decision_id_for_evaluated_row(
        self,
        row: Mapping[str, Any] | pd.Series,
        *,
        model_release_id: str = "legacy-unversioned",
    ) -> str:
        """Return the canonical content-addressed ID for an evaluated row.

        Consumers that persist decisions must use this method instead of
        rebuilding an abbreviated timestamp-only hash.  The ID therefore
        changes whenever the executable intent or its model/strategy
        provenance changes, even if the observation timestamp is unchanged.
        """
        target_position = float(row["Target_Position"])
        regime_value = row.get(self.spec.regime_col, 0)
        market_regime = 0 if pd.isna(regime_value) else int(regime_value)
        decision_payload = json.dumps(
            {
                "as_of": pd.Timestamp(row[self.spec.timestamp_col]).isoformat(),
                "strategy_fingerprint": self.spec.fingerprint(),
                "model_release_id": str(model_release_id),
                "direction": int(np.sign(target_position)),
                "target_weight": abs(target_position),
                "predicted_return": float(row[self.spec.prediction_col]),
                "predicted_return_smooth": float(row["Pred_Smooth"]),
                "market_regime": market_regime,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"decision:{sha256(decision_payload).hexdigest()}"


class ModelRuntime:
    """Read-only inference adapter for current and legacy model artifacts."""

    def __init__(self, artifact: Mapping[str, Any], *, artifact_path: Path | None = None):
        if not isinstance(artifact, Mapping):
            raise ValueError("model artifact must be a mapping")
        required = {"model_base", "scaler", "features"}
        missing = sorted(required.difference(artifact))
        if missing:
            raise ValueError(f"model artifact missing keys: {missing}")

        self._artifact = dict(artifact)
        self.artifact_path = artifact_path
        self.model_base = artifact["model_base"]
        self.model_active = artifact.get("model_active")
        self.scaler = artifact["scaler"]
        self.features = tuple(artifact["features"])
        self.hierarchical_model = artifact.get("hierarchical_model")
        self.selected_model_kind = artifact.get(
            "selected_model_kind", "flat_combined"
        )
        self.gp_miner = artifact.get("gp_miner")

        legacy = "artifact_schema_version" not in artifact
        schema_version = int(artifact.get("artifact_schema_version", 1))
        release_id = str(
            artifact.get("model_release_id")
            or artifact.get("release_id")
            or artifact.get("artifact_id")
            or "legacy-unversioned"
        )
        self._metadata = MappingProxyType(
            {
                "artifact_schema_version": schema_version,
                "model_release_id": release_id,
                "selected_model_kind": self.selected_model_kind,
                "features": self.features,
                "has_gp_miner": self.gp_miner is not None,
                "legacy": legacy,
                "artifact_path": str(artifact_path) if artifact_path else None,
            }
        )

    @classmethod
    def from_path(cls, model_path: str | Path) -> "ModelRuntime":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found: {path}")
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        return cls(artifact, artifact_path=path)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    def _with_gp_features(self, factors: pd.DataFrame) -> pd.DataFrame:
        frame = factors.copy()
        selected = getattr(self.gp_miner, "selected_programs_", None)
        if self.gp_miner is None or not selected:
            return frame
        inputs = list(getattr(self.gp_miner, "feature_names_in_", ()))
        missing = sorted(set(inputs).difference(frame.columns))
        if missing:
            raise ValueError(f"GP input schema mismatch; missing: {missing}")
        generated = self.gp_miner.transform(frame[inputs])
        if isinstance(generated, pd.DataFrame):
            generated_items = generated.items()
        elif isinstance(generated, Mapping):
            generated_items = generated.items()
        else:
            raise ValueError("GP transform must return a mapping or DataFrame")
        for name, values in generated_items:
            frame[name] = np.asarray(values)
        return frame

    def predict_frame(self, factors: pd.DataFrame) -> pd.DataFrame:
        if factors is None or factors.empty:
            raise ValueError("factor frame must not be empty")
        frame = self._with_gp_features(factors)
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            raise ValueError(f"model feature schema mismatch; missing: {missing}")

        if self.selected_model_kind == "hierarchical":
            if self.hierarchical_model is None:
                raise ValueError(
                    "artifact selects hierarchical model but has no hierarchical_model"
                )
            components = self.hierarchical_model.components(frame)
            required = {
                "prediction",
                "micro_alpha",
                "macro_state",
                "gate_weight",
                "event_alpha",
            }
            missing_components = sorted(required.difference(components.columns))
            if missing_components:
                raise ValueError(
                    "hierarchical components missing columns: "
                    f"{missing_components}"
                )
            frame["Pred_Ret"] = np.asarray(components["prediction"], dtype=float)
            frame["Model_Used"] = "hierarchical"
            for name in (
                "micro_alpha",
                "macro_state",
                "gate_weight",
                "event_alpha",
            ):
                frame[name] = np.asarray(components[name], dtype=float)
            return frame

        matrix = self.scaler.transform(frame.loc[:, self.features])
        prediction = np.asarray(self.model_base.predict(matrix), dtype=float).copy()
        legacy_routing = self.model_active is not None
        if legacy_routing:
            if "Market_Regime" in frame:
                regimes = pd.to_numeric(
                    frame["Market_Regime"], errors="coerce"
                ).fillna(0).to_numpy()
            else:
                regimes = np.zeros(len(frame), dtype=int)
            active = np.isin(regimes, [1, 2])
            if active.any():
                prediction[active] = self.model_active.predict(matrix[active])
            model_used = np.where(active, "active", "base")
        else:
            model_used = np.full(len(frame), self.selected_model_kind, dtype=object)

        frame["Pred_Ret"] = prediction
        frame["Model_Used"] = model_used
        return frame


__all__ = [
    "DEFAULT_STRATEGY_ID",
    "STRATEGY_SPECS",
    "ModelRuntime",
    "SignalDecision",
    "SignalPolicy",
    "StrategySpec",
    "get_strategy_spec",
]
