# -*- coding: utf-8 -*-
# inference.py — Phase 2: 实时信号生成（训练/推理模式分离）

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime


class SignalGenerator:
    """Load trained model and generate trading signals from factor data."""

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.model_base = artifacts['model_base']
        self.model_active = artifacts.get('model_active')
        self.scaler = artifacts['scaler']
        self.features = artifacts['features']

        self.signal_smooth_span = 160

        # Regime-specific quantile thresholds (matching backtest logic)
        self.thresholds = {
            0: {'upper': 0.90, 'lower': 0.05},
            1: {'upper': 0.80, 'lower': 0.20},
            2: {'upper': 0.80, 'lower': 0.20},
        }

        print(f"[SignalGen] 模型已加载: {len(self.features)} 个特征")
        print(f"[SignalGen] 基础模型: {'OK' if self.model_base else 'MISSING'}")
        print(f"[SignalGen] 高波动/趋势模型: {'OK' if self.model_active else '未训练'}")

    def predict(self, df_factors):
        """
        Run prediction on factor DataFrame.
        Routes rows to base or active model based on Market_Regime column.
        """
        df = df_factors.copy()

        # Ensure all required features exist
        available = [f for f in self.features if f in df.columns]
        missing = set(self.features) - set(available)
        if missing:
            print(f"[SignalGen] WARNING: 缺失 {len(missing)} 个特征，补0")
            for m in missing:
                df[m] = 0
            available = self.features

        X = df[available].fillna(0).values
        X_scaled = self.scaler.transform(X)

        n = len(df)
        y_pred = np.zeros(n)
        model_used = np.full(n, 'base', dtype=object)

        regime = df['Market_Regime'].values if 'Market_Regime' in df.columns else np.zeros(n)

        # Route: active model for regime 1 (high vol) or 2 (trend)
        if self.model_active is not None:
            active_mask = np.isin(regime, [1, 2])
            if active_mask.sum() > 0:
                y_pred[active_mask] = self.model_active.predict(X_scaled[active_mask])
                model_used[active_mask] = 'active'

        # Base model for remainder
        base_mask = ~np.isin(regime, [1, 2]) if self.model_active is not None else np.ones(n, dtype=bool)
        if base_mask.sum() > 0:
            y_pred[base_mask] = self.model_base.predict(X_scaled[base_mask])

        df['Pred_Ret'] = y_pred
        df['Model_Used'] = model_used

        return df

    def generate_signal(self, df_factors):
        """
        Generate trading signal for the LATEST bar.
        Returns (signal_dict, df_with_predictions).
        """
        df = self.predict(df_factors)

        # EMA smooth predictions
        df['Pred_Smooth'] = df['Pred_Ret'].ewm(span=self.signal_smooth_span, adjust=False).mean()

        # Lagged smoothed prediction for threshold calculation (no look-ahead)
        pred_lagged = df['Pred_Smooth'].shift(1)

        # Rolling quantile thresholds over expanding window
        upper_q = pred_lagged.expanding(min_periods=100).quantile(
            self.thresholds[0]['upper'])
        lower_q = pred_lagged.expanding(min_periods=100).quantile(
            self.thresholds[0]['lower'])

        # Latest values
        latest_smooth = df['Pred_Smooth'].iloc[-1]
        latest_upper = upper_q.iloc[-1]
        latest_lower = lower_q.iloc[-1]
        latest_pred = df['Pred_Ret'].iloc[-1]
        latest_regime = int(df['Market_Regime'].iloc[-1]) if 'Market_Regime' in df.columns else 0

        # Direction from threshold breakout
        if pd.notna(latest_smooth) and pd.notna(latest_upper) and pd.notna(latest_lower):
            if latest_smooth > latest_upper:
                direction = 1
            elif latest_smooth < latest_lower:
                direction = -1
            else:
                direction = 0
        else:
            direction = 0
            latest_upper = 0.0
            latest_lower = 0.0

        # Confidence: distance from threshold normalized by prediction std
        pred_std = df['Pred_Smooth'].std()
        if direction == 1 and pred_std > 0:
            confidence = min(1.0, (latest_smooth - latest_upper) / (pred_std + 1e-9))
        elif direction == -1 and pred_std > 0:
            confidence = min(1.0, (latest_lower - latest_smooth) / (pred_std + 1e-9))
        else:
            confidence = 0.0

        # Suggested position weight (regime-adjusted)
        weight_map = {0: 1.0, 1: 0.8, 2: 0.8}
        suggested_weight = weight_map.get(latest_regime, 1.0) * confidence

        latest = df.iloc[-1]
        ts = str(latest['date']) if 'date' in df.columns else datetime.now().isoformat()

        signal = {
            'timestamp': ts,
            'close': float(latest['close']) if 'close' in df.columns else None,
            'market_regime': latest_regime,
            'regime_name': ['正常', '高波动', '趋势'][latest_regime],
            'predicted_return': round(float(latest_pred), 6),
            'predicted_return_smooth': round(float(latest_smooth), 6),
            'direction': direction,
            'direction_name': {1: '做多', -1: '做空', 0: '观望'}[direction],
            'confidence': round(float(confidence), 4),
            'model_used': str(latest.get('Model_Used', 'base')),
            'suggested_weight': round(float(suggested_weight), 4),
            'upper_threshold': round(float(latest_upper), 6),
            'lower_threshold': round(float(latest_lower), 6),
        }

        return signal, df

    def print_signal(self, signal):
        """Pretty-print signal to console."""
        direction_mark = {1: '[LONG]', -1: '[SHORT]', 0: '[FLAT ]'}

        print("\n" + "=" * 50)
        print("         实时交易信号")
        print("=" * 50)
        print(f"  时间:       {signal['timestamp']}")
        print(f"  价格:       {signal['close']}")
        print(f"  市场状态:   {signal['regime_name']} (regime={signal['market_regime']})")
        print(f"  预测收益:   {signal['predicted_return']:.4f}%")
        print(f"  平滑预测:   {signal['predicted_return_smooth']:.4f}%")
        print(f"  上阈值:     {signal['upper_threshold']:.4f}%")
        print(f"  下阈值:     {signal['lower_threshold']:.4f}%")
        print(f"  " + "-" * 40)
        print(f"  信号方向:   {direction_mark.get(signal['direction'], '[????]')} {signal['direction_name']}")
        print(f"  置信度:     {signal['confidence']:.1%}")
        print(f"  建议仓位:   {signal['suggested_weight']:.1%}")
        print(f"  使用模型:   {signal['model_used']}")
        print("=" * 50)


def run_inference(base_dir):
    """Full inference pipeline: factor extraction -> load model -> generate signal."""
    import factor_extraction

    MODEL_FILE = os.path.join(base_dir, "models/trained_model.pkl")
    FACTOR_FILE = os.path.join(base_dir, "outputs/df_factors.pkl")
    SIGNAL_JSON = os.path.join(base_dir, "outputs/signal.json")
    SIGNAL_CSV = os.path.join(base_dir, "outputs/signal_history.csv")

    # 1. Check model exists
    if not os.path.exists(MODEL_FILE):
        print(f"[Inference ERROR] 未找到模型文件: {MODEL_FILE}")
        print(f"[Inference] 请先运行训练模式: python main.py --mode train")
        return False

    # 2. Run factor extraction (rebuilds df_factors.pkl)
    print("\n[1/3] 因子计算...")
    if not factor_extraction.run_process(base_dir, tick_subdir="data/tick"):
        print("[Inference ERROR] 因子计算失败")
        return False

    # 3. Load factors + model, generate signal
    print("\n[2/3] 加载模型并生成信号...")
    df_factors = pd.read_pickle(FACTOR_FILE)

    generator = SignalGenerator(MODEL_FILE)
    signal, df_pred = generator.generate_signal(df_factors)

    # 4. Output
    print("\n[3/3] 信号输出...")
    generator.print_signal(signal)

    # JSON output
    signal_out = {k: v for k, v in signal.items()}
    with open(SIGNAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(signal_out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Inference] 信号JSON已保存: {SIGNAL_JSON}")

    # CSV history append
    signal_row = pd.DataFrame([signal_out])
    if os.path.exists(SIGNAL_CSV):
        signal_row.to_csv(SIGNAL_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        signal_row.to_csv(SIGNAL_CSV, index=False, encoding='utf-8-sig')
    print(f"[Inference] 信号历史已追加: {SIGNAL_CSV}")

    return True
