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
    """Read-only dashboard facade over the canonical TL signal policy."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.base_dir, "models", "trained_model.pkl")
        from model_release import ReleaseManager

        manager = ReleaseManager(
            os.path.join(self.base_dir, "artifacts", "models"),
            legacy_model_path=model_path,
        )
        signature = manager.current_signature()
        if signature is None or signature.source != "release":
            return
        from inference import SignalGenerator
        self.model = SignalGenerator(model_path)

    @property
    def is_ready(self):
        return self.model is not None

    def generate_live_signal(self, df_factors):
        """Generate exactly the same decision used by CLI and ResearchAgent."""
        if not self.is_ready or df_factors is None or len(df_factors) < 2:
            return None
        signal, evaluated = self.model.generate_signal(df_factors)
        signal['df_full'] = evaluated
        return signal

    def build_prediction_chart(self, df_pred, days=7):
        """构建预测走势图"""
        if df_pred is None or len(df_pred) < 100:
            return None

        df = df_pred.tail(min(len(df_pred), 240 * 5 * days)).copy()

        has_rank = 'Pred_Rank' in df and df['Pred_Rank'].notna().any()
        middle_title = "预测排名 (Percentile)" if has_rank else "平滑预测与策略阈值"
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.45, 0.30, 0.25],
            vertical_spacing=0.04,
            subplot_titles=("预测信号与价格", middle_title, "原始预测 vs 平滑预测"),
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

        if has_rank:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['Pred_Rank'] * 100,
                mode='lines', name='预测排名',
                line=dict(color='#FFB74D', width=1.5),
                fill='tozeroy', fillcolor='rgba(255,183,77,0.08)',
            ), row=2, col=1)
            short_inner, short_outer = self.model.spec.short_band
            long_inner, long_outer = self.model.spec.long_band
            for level, color in (
                (long_outer * 100, 'gray'),
                (long_inner * 100, '#00E676'),
                (short_outer * 100, 'gray'),
                (short_inner * 100, '#FF5252'),
                (50, 'gray'),
            ):
                fig.add_hline(
                    y=level, line_dash='dash', line_color=color,
                    opacity=0.4, row=2, col=1,
                )
        else:
            for column, name, color in (
                ('Pred_Smooth', '平滑预测', '#FFB74D'),
                ('Upper_Q', '上阈值', '#00E676'),
                ('Lower_Q', '下阈值', '#FF5252'),
            ):
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[column], mode='lines', name=name,
                    line=dict(color=color, width=1.5),
                ), row=2, col=1)

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
        if has_rank:
            fig.update_yaxes(title_text="百分位%", row=2, col=1, range=[0, 100])
        else:
            fig.update_yaxes(title_text="预测/阈值", row=2, col=1)
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
