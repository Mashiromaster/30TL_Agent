# -*- coding: utf-8 -*-
"""Cost-aware evaluation for the canonical TL strategy contract."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from strategy_runtime import SignalPolicy, get_strategy_spec


class Backtester:
    """Evaluate predictions with the exact SignalPolicy used in production."""

    def __init__(self, df, strategy_spec=None, *, model_release_id=None):
        self.df = df.copy()
        self.strategy_spec = strategy_spec or get_strategy_spec()
        self.signal_policy = SignalPolicy(self.strategy_spec)
        self.commission = self.strategy_spec.commission_rate
        self.tick_size = self.strategy_spec.tick_size
        self.slippage_ticks = self.strategy_spec.slippage_ticks
        self.target_vol = 0.12
        self.numeric_metrics = None
        if model_release_id is None and "Model_Release_ID" in self.df:
            release_ids = self.df["Model_Release_ID"].dropna().astype(str).unique()
            if len(release_ids) > 1:
                raise ValueError("backtest frame mixes multiple model releases")
            model_release_id = release_ids[0] if len(release_ids) else None
        self.model_release_id = model_release_id or "legacy-unversioned"

    def run_strategy(self):
        print(
            f"[Backtest] 执行统一策略 {self.strategy_spec.strategy_id} "
            f"({self.strategy_spec.fingerprint()[:20]})"
        )
        self.df = self.signal_policy.evaluate_frame(
            self.df, model_release_id=self.model_release_id
        )

        # Effective_Position is the post-trade position at this bar's close.
        # A close-to-close return ending at t can only be earned by the
        # position held after t-1, so returns use one further shift.  This
        # aligns execution_lag_bars=1 with the label contract's close[t+1]
        # entry instead of granting the pre-entry close[t] -> close[t+1] move.
        self.df["Position"] = self.df["Effective_Position"].astype(float)
        self.df["Position_Weight"] = self.df["Target_Position"].abs()
        self.df["Asset_Ret"] = self.df["close"].pct_change().fillna(0.0)
        self.df["Unlevered_Ret"] = self.df["Position"] * self.df["Asset_Ret"]
        roll_vol = (
            self.df["Unlevered_Ret"]
            .shift(1)
            .rolling(window=240 * 5, min_periods=240)
            .std()
            * np.sqrt(240 * 252)
        )
        self.df["Target_Vol"] = self.target_vol
        if "Market_Regime" in self.df.columns:
            effective_regime = self.df["Market_Regime"].shift(
                self.strategy_spec.execution_lag_bars
            )
            self.df.loc[
                effective_regime == 1, "Target_Vol"
            ] = self.target_vol * 1.5
            self.df.loc[
                effective_regime == 2, "Target_Vol"
            ] = self.target_vol * 0.8
        self.df["Leverage"] = (
            self.df["Target_Vol"] / roll_vol
        ).fillna(1.0).clip(0.3, 2.0)
        self.df.loc[roll_vol > 0.25, "Leverage"] = 0.3
        self.df.loc[roll_vol > 0.35, "Leverage"] = 0.0
        self.df["Executed_Position"] = (
            self.df["Position"] * self.df["Leverage"]
        )
        self.df["Held_Position"] = self.df["Executed_Position"].shift(1).fillna(0.0)
        self.df["Trades"] = self.df["Executed_Position"].diff().fillna(
            self.df["Executed_Position"]
        ).abs()
        rel_slippage = (self.slippage_ticks * self.tick_size) / self.df["close"]
        self.df["Cost"] = self.df["Trades"] * (
            self.commission + rel_slippage
        )
        self.df["Net_Ret"] = (
            self.df["Held_Position"] * self.df["Asset_Ret"]
            - self.df["Cost"]
        )
        self.df["Strategy_Ret"] = self.df["Net_Ret"]
        self.df["Cum_Ret"] = (1 + self.df["Strategy_Ret"]).cumprod()
        self._print_diagnostics()

    def _daily_frame(self):
        if "Strategy_Ret" not in self.df:
            raise RuntimeError("run_strategy must be called before metrics")
        dates = pd.to_datetime(self.df["date"], errors="raise").dt.date
        grouped = self.df.assign(_trade_date=dates).groupby("_trade_date")
        daily = pd.DataFrame(
            {
                "ret": grouped["Strategy_Ret"].apply(
                    lambda values: float((1.0 + values).prod() - 1.0)
                ),
                "position": grouped["Executed_Position"].last(),
                "trades": grouped["Trades"].sum(),
            }
        ).reset_index(names="date")
        daily["cum"] = (1 + daily["ret"]).cumprod()
        daily["peak"] = daily["cum"].cummax()
        daily["drawdown"] = (daily["cum"] - daily["peak"]) / daily["peak"]
        return daily

    def compute_numeric_metrics(self):
        daily = self._daily_frame()
        n_days = len(daily)
        if n_days == 0:
            raise ValueError("backtest has no daily observations")

        final_nav = float(daily["cum"].iloc[-1])
        total_return = final_nav - 1.0
        annual_return = (
            final_nav ** (252.0 / n_days) - 1.0 if final_nav > 0 else -1.0
        )
        annual_volatility = float(daily["ret"].std(ddof=1) * np.sqrt(252))
        if not np.isfinite(annual_volatility):
            annual_volatility = 0.0
        sharpe = (
            (annual_return - 0.02) / annual_volatility
            if annual_volatility > 0 else 0.0
        )
        max_drawdown = float(daily["drawdown"].min())

        in_drawdown = daily["drawdown"] < -0.001
        durations = []
        start = None
        for offset, active in enumerate(in_drawdown):
            if active and start is None:
                start = offset
            elif not active and start is not None:
                durations.append(offset - start)
                start = None
        if start is not None:
            durations.append(n_days - start)
        max_drawdown_duration = max(durations) if durations else 0

        total_trades = float(daily["trades"].sum() / 2.0)
        annual_turnover = total_trades * (252.0 / n_days)
        trading_days = int((daily["ret"].abs() > 0.0001).sum())
        winning_days = int((daily["ret"] > 0.0001).sum())
        win_rate = winning_days / trading_days if trading_days else 0.0
        calmar = annual_return / abs(max_drawdown) if max_drawdown else 0.0
        active_ratio = float((daily["position"].abs() > 0).mean())

        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_duration_days": int(max_drawdown_duration),
            "annual_turnover": float(annual_turnover),
            "daily_win_rate": float(win_rate),
            "calmar": float(calmar),
            "active_ratio": float(active_ratio),
            "observation_days": int(n_days),
            "strategy_id": self.strategy_spec.strategy_id,
            "strategy_fingerprint": self.strategy_spec.fingerprint(),
            "model_release_id": self.model_release_id,
        }
        numeric_values = [
            value for value in metrics.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("backtest produced non-finite metrics")
        self.numeric_metrics = metrics
        return metrics, daily

    def compute_metrics(self):
        metrics, daily = self.compute_numeric_metrics()
        display = {
            "累计收益": f"{metrics['total_return']:.2%}",
            "年化收益": f"{metrics['annual_return']:.2%}",
            "年化波动": f"{metrics['annual_volatility']:.2%}",
            "夏普比率": f"{metrics['sharpe']:.3f}",
            "最大回撤": f"{metrics['max_drawdown']:.2%}",
            "回撤持续": f"{metrics['max_drawdown_duration_days']}天",
            "年化换手": f"{metrics['annual_turnover']:.1f}x",
            "日胜率": f"{metrics['daily_win_rate']:.2%}",
            "Calmar比率": f"{metrics['calmar']:.2f}",
            "持仓比例": f"{metrics['active_ratio']:.1%}",
        }
        print("\n" + "=" * 50)
        print("详细风险指标")
        print("=" * 50)
        print(f"测试期: {metrics['observation_days']} 天")
        print(f"持仓比例: {metrics['active_ratio']:.1%}")
        print(f"总收益: {metrics['total_return']:.2%}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        return display, daily

    def _print_diagnostics(self):
        changes = int((self.df["Position"].diff().abs() > 0).sum())
        trading_days = max(1, len(pd.to_datetime(self.df["date"]).dt.date.unique()))
        long_pct = float((self.df["Position"] > 0).mean())
        short_pct = float((self.df["Position"] < 0).mean())
        flat_pct = float((self.df["Position"] == 0).mean())
        print(f"[Backtest] 换仓次数: {changes}; 日均 {changes / trading_days:.2f}")
        print(
            f"[Backtest] 做多: {long_pct:.1%}, 做空: {short_pct:.1%}, "
            f"空仓: {flat_pct:.1%}"
        )

    def plot_report(self, daily, save_path):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        axes[0].plot(daily["date"], daily["cum"], color="darkgreen", linewidth=2)
        axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("Canonical TL Strategy - Cumulative Return")
        axes[1].fill_between(
            daily["date"], daily["drawdown"], 0, color="red", alpha=0.5
        )
        axes[1].set_title(f"Drawdown (Max: {daily['drawdown'].min():.2%})")
        colors = ["green" if value > 0 else "red" for value in daily["ret"]]
        axes[2].bar(daily["date"], daily["ret"], color=colors, alpha=0.7)
        axes[2].set_title("Daily Returns")
        axes[3].fill_between(
            daily["date"], 0, daily["position"],
            where=daily["position"] > 0, color="green", alpha=0.5, label="Long",
        )
        axes[3].fill_between(
            daily["date"], 0, daily["position"],
            where=daily["position"] < 0, color="red", alpha=0.5, label="Short",
        )
        axes[3].set_title("Effective Position")
        axes[3].legend()
        for axis in axes:
            axis.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def evaluate_predictions(
    df,
    *,
    strategy_id=None,
    strategy_spec=None,
    model_release_id=None,
):
    """Return one cost-aware evaluation without writing project artifacts."""
    spec = strategy_spec or get_strategy_spec(strategy_id)
    tester = Backtester(
        df,
        strategy_spec=spec,
        model_release_id=model_release_id,
    )
    tester.run_strategy()
    metrics, daily = tester.compute_numeric_metrics()
    return tester, metrics, daily


def _atomic_csv(frame, destination):
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def run_process(base_dir, *, strategy_id=None, prediction_path=None):
    print("\n" + "=" * 50)
    print("STEP 4: 策略回测")
    print("=" * 50)
    if prediction_path is None:
        from model_release import ReleaseManager

        manager = ReleaseManager(
            os.path.join(base_dir, "artifacts", "models"),
            legacy_model_path=os.path.join(base_dir, "models", "trained_model.pkl"),
        )
        managed = manager.active_artifact_path("predictions")
        prediction_path = os.fspath(managed) if managed else os.path.join(
            base_dir, "outputs", "df_predictions.pkl"
        )
    metrics_path = os.path.join(base_dir, "outputs", "backtest_metrics.csv")
    if not os.path.exists(prediction_path):
        print(f"[Backtest ERROR] 找不到预测文件: {prediction_path}")
        return False

    predictions = pd.read_pickle(prediction_path)
    print(f"[Backtest] 加载测试集数据: {len(predictions)} 行")
    tester, numeric, daily = evaluate_predictions(
        predictions, strategy_id=strategy_id
    )
    display, _ = tester.compute_metrics()
    print("\n" + "=" * 50)
    print("样本外回测表现")
    print("=" * 50)
    for name, value in display.items():
        print(f"{name:12s}: {value}")
    print("=" * 50)

    output = dict(display)
    output.update({
        "strategy_id": numeric["strategy_id"],
        "strategy_fingerprint": numeric["strategy_fingerprint"],
    })
    _atomic_csv(pd.DataFrame([output]), metrics_path)

    report_path = os.path.join(base_dir, "outputs", "strategy_report.png")
    report_temp = report_path + ".tmp.png"
    try:
        tester.plot_report(daily, report_temp)
        os.replace(report_temp, report_path)
    finally:
        if os.path.exists(report_temp):
            os.unlink(report_temp)
    return True


__all__ = ["Backtester", "evaluate_predictions", "run_process"]
