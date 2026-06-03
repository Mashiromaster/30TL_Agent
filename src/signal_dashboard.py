# -*- coding: utf-8 -*-
# signal_dashboard.py — 真实信号看板 (替代原demo信号页面)
# 供 dashboard.py 调用, 提供完整的信号生成+预测可视化+仓位管理

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta


class SignalDashboard:
    """实时信号看板引擎 — 替代原SignalGenerator + 静态signal.json"""

    # 甜点区过滤参数 (基于219笔记忆回测优化, 2026-06-03)
    SHORT_INNER = 0.15
    SHORT_OUTER = 0.30
    LONG_INNER  = 0.70
    LONG_OUTER  = 0.85
    SMOOTH_SPAN = 60
    LOOKBACK    = 480
    CONFIRM     = 10

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.base_dir, "models", "trained_model.pkl")
        if not os.path.exists(model_path):
            return
        import pickle
        from inference import SignalGenerator
        self.model = SignalGenerator(model_path)

    @property
    def is_ready(self):
        return self.model is not None

    def generate_live_signal(self, df_factors):
        """基于最新因子数据生成信号 — 包含排名信号逻辑"""
        if not self.is_ready or df_factors is None or len(df_factors) < 500:
            return None

        df = df_factors.tail(3000).copy()
        df['date'] = pd.to_datetime(df['date'])

        try:
            df_pred = self.model.predict(df)
        except Exception:
            return None

        # 排名信号 (甜点区过滤: 跳过最极端排名)
        df_pred['Pred_Smooth'] = df_pred['Pred_Ret'].ewm(span=self.SMOOTH_SPAN, adjust=False).mean()

        df_pred['Pred_Rank'] = np.nan
        for i in range(self.LOOKBACK, len(df_pred)):
            window = df_pred['Pred_Smooth'].iloc[i - self.LOOKBACK:i]
            current = df_pred['Pred_Smooth'].iloc[i]
            df_pred.loc[df_pred.index[i], 'Pred_Rank'] = (window < current).sum() / len(window)

        # 甜点区信号: 仅交易中间置信度区域
        df_pred['Raw_Signal'] = 0
        df_pred.loc[(df_pred['Pred_Rank'] >= self.SHORT_INNER) & (df_pred['Pred_Rank'] < self.SHORT_OUTER), 'Raw_Signal'] = -1
        df_pred.loc[(df_pred['Pred_Rank'] >= self.LONG_INNER) & (df_pred['Pred_Rank'] < self.LONG_OUTER), 'Raw_Signal'] = 1

        confirm = self.CONFIRM
        df_pred['Signal'] = 0
        for i in range(confirm, len(df_pred)):
            ws = df_pred['Raw_Signal'].iloc[i - confirm:i]
            if (ws == 1).all():
                df_pred.loc[df_pred.index[i], 'Signal'] = 1
            elif (ws == -1).all():
                df_pred.loc[df_pred.index[i], 'Signal'] = -1

        # 提取最后一行信号
        latest = df_pred.iloc[-1]
        regime = int(latest.get('Market_Regime', 0)) if 'Market_Regime' in df_pred.columns else 0

        return {
            'timestamp': str(latest['date']),
            'close': float(latest.get('close', 0)),
            'market_regime': regime,
            'regime_name': {0: '正常', 1: '高波动', 2: '趋势'}.get(regime, '未知'),
            'predicted_return': round(float(latest['Pred_Ret']), 6),
            'predicted_return_smooth': round(float(latest['Pred_Smooth']), 6),
            'pred_rank_pct': round(float(latest['Pred_Rank']), 3),
            'direction': int(latest['Signal']),
            'direction_name': {1: '做多 LONG', -1: '做空 SHORT', 0: '观望 FLAT'}[int(latest['Signal'])],
            'confidence': round(abs(float(latest['Pred_Rank']) - 0.5) * 2, 3),
            'suggested_weight': round(abs(float(latest['Pred_Rank']) - 0.5) * 2 * {0: 1.0, 1: 0.8, 2: 0.8}.get(regime, 1.0), 3),
            'model_used': str(latest.get('Model_Used', 'base')),
            'df_full': df_pred,
        }

    def build_prediction_chart(self, df_pred, days=7):
        """构建预测走势图"""
        if df_pred is None or len(df_pred) < 100:
            return None

        df = df_pred.tail(min(len(df_pred), 240 * 5 * days)).copy()

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.30, 0.25],
            vertical_spacing=0.04,
            subplot_titles=("预测信号与价格", "预测排名 (Percentile)", "原始预测 vs 平滑预测"),
        )

        # Row 1: Price + Signal markers
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['close'], mode='lines', name='价格',
            line=dict(color='#B0BEC5', width=1.2),
        ), row=1, col=1)

        # Mark signals on price
        for sig_val, color, name in [(1, '#00E676', '做多信号'), (-1, '#FF5252', '做空信号')]:
            mask = df['Signal'] == sig_val
            if mask.any():
                sig_dates = df.loc[mask, 'date']
                sig_prices = df.loc[mask, 'close']
                fig.add_trace(go.Scatter(
                    x=sig_dates, y=sig_prices,
                    mode='markers', name=name,
                    marker=dict(color=color, size=8, symbol='triangle-up' if sig_val == 1 else 'triangle-down'),
                ), row=1, col=1)

        # Row 2: Pred Rank percentile
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['Pred_Rank'] * 100,
            mode='lines', name='预测排名',
            line=dict(color='#FFB74D', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,183,77,0.08)',
        ), row=2, col=1)
        fig.add_hline(y=self.LONG_OUTER * 100, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)
        fig.add_hline(y=self.LONG_INNER * 100, line_dash="dash", line_color="#00E676", opacity=0.5, row=2, col=1,
                      annotation_text="做多甜点区")
        fig.add_hline(y=self.SHORT_OUTER * 100, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)
        fig.add_hline(y=self.SHORT_INNER * 100, line_dash="dash", line_color="#FF5252", opacity=0.5, row=2, col=1,
                      annotation_text="做空甜点区")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

        # Row 3: Raw Pred vs Smooth
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['Pred_Ret'],
            mode='lines', name='原始预测',
            line=dict(color='#64B5F6', width=0.8), opacity=0.6,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['Pred_Smooth'],
            mode='lines', name='平滑预测',
            line=dict(color='#FFB74D', width=2),
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, row=3, col=1)

        fig.update_layout(
            height=600, margin=dict(l=10, r=10, t=35, b=10),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.08, x=0.5, xanchor='center'),
            template='plotly_dark',
        )
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.25, 9.5], pattern="hour"),
            dict(bounds=[11.5, 13], pattern="hour"),
        ], row=1, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="百分位%", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="预测收益%", row=3, col=1)

        return fig

    def compute_performance_snapshot(self, df_pred):
        """计算近期预测表现"""
        if df_pred is None or 'Target_Ret' not in df_pred.columns:
            return None

        recent = df_pred.tail(min(len(df_pred), 5000))
        mask = recent['Target_Ret'].notna()
        if mask.sum() < 100:
            return None

        pred = recent.loc[mask, 'Pred_Ret']
        actual = recent.loc[mask, 'Target_Ret']
        ic = np.corrcoef(pred, actual)[0, 1]

        # Hit rate by signal direction
        if 'Signal' in recent.columns:
            traded = recent[mask & (recent['Signal'] != 0)]
            if len(traded) > 0:
                hit = (np.sign(traded['Signal']) == np.sign(traded['Target_Ret'])).mean()
            else:
                hit = 0
        else:
            hit = 0

        # Volatility context
        vol = actual.std() * np.sqrt(240)

        return {
            'ic': round(ic, 4),
            'hit_rate': round(hit, 3),
            'n_signals': len(traded) if 'traded' in locals() else 0,
            'pred_vol': round(vol, 4),
            'pred_mean': round(pred.mean(), 6),
            'actual_mean': round(actual.mean(), 6),
        }
