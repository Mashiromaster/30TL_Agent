# -*- coding: utf-8 -*-
# strategy_agent.py — Phase 4: Claude Agent 交互层（市场解读、自然语言查询）

import pandas as pd
import numpy as np
import os
import json
import argparse
from llm_intelligence import LLMAnalyzer


class StrategyContext:
    """Load all strategy artifacts into memory for analysis."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self._df_factors = None
        self._df_pred = None
        self._signal = None
        self._metrics = None
        self._importance = None
        self._macro_factors = None

    @property
    def df_factors(self):
        if self._df_factors is None:
            path = os.path.join(self.base_dir, "outputs/df_factors.pkl")
            if os.path.exists(path):
                self._df_factors = pd.read_pickle(path)
        return self._df_factors

    @property
    def df_pred(self):
        if self._df_pred is None:
            path = os.path.join(self.base_dir, "outputs/df_predictions.pkl")
            if os.path.exists(path):
                self._df_pred = pd.read_pickle(path)
        return self._df_pred

    @property
    def signal(self):
        if self._signal is None:
            path = os.path.join(self.base_dir, "outputs/signal.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self._signal = json.load(f)
        return self._signal or {}

    @property
    def metrics(self):
        if self._metrics is None:
            path = os.path.join(self.base_dir, "outputs/backtest_metrics.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Wide format: headers are metric names, one data row
                self._metrics = df.iloc[0].to_dict()
        return self._metrics or {}

    @property
    def importance(self):
        if self._importance is None:
            path = os.path.join(self.base_dir, "outputs/feature_importance.csv")
            if os.path.exists(path):
                self._importance = pd.read_csv(path)
        return self._importance

    @property
    def macro_factors(self):
        if self._macro_factors is None:
            path = os.path.join(self.base_dir, "outputs/macro_factors.pkl")
            if os.path.exists(path):
                self._macro_factors = pd.read_pickle(path)
        return self._macro_factors


class MarketAnalyzer:
    """Generate structured Chinese-language market reports from strategy state."""

    def __init__(self, ctx):
        self.ctx = ctx

    # ================================================================
    # 1. Market Snapshot
    # ================================================================
    def market_snapshot(self):
        lines = []
        lines.append("=" * 56)
        lines.append("  TL国债期货策略 — 市场快照")
        lines.append("=" * 56)

        signal = self.ctx.signal
        df = self.ctx.df_factors

        if signal:
            lines.append(f"  时间:     {signal.get('timestamp', 'N/A')}")
            lines.append(f"  合约:     {self._latest_ticker()}")
            lines.append(f"  价格:     {signal.get('close', 'N/A')}")
            lines.append(f"  市场状态: {signal.get('regime_name', 'N/A')}")

            # Regime distribution
            if df is not None and 'Market_Regime' in df.columns:
                total = len(df)
                lines.append(f"\n  市场状态分布 (全样本 {total:,} 根K线):")
                for rid, rname in [(0, '正常'), (1, '高波动'), (2, '趋势')]:
                    cnt = (df['Market_Regime'] == rid).sum()
                    pct = cnt / total * 100
                    bar = '█' * int(pct / 2)
                    lines.append(f"    {rname:6s}: {cnt:>8,}  ({pct:5.1f}%) {bar}")

            # Signal
            direction = signal.get('direction', 0)
            dir_map = {1: '做多 LONG', -1: '做空 SHORT', 0: '观望 FLAT'}
            lines.append(f"\n  当前信号: {dir_map.get(direction, '未知')}")
            lines.append(f"  置信度:   {signal.get('confidence', 0):.1%}")
            lines.append(f"  建议仓位: {signal.get('suggested_weight', 0):.1%}")

        lines.append("=" * 56)
        return "\n".join(lines)

    # ================================================================
    # 2. Signal Interpretation
    # ================================================================
    def signal_interpretation(self):
        lines = []
        lines.append("=" * 56)
        lines.append("  信号详细解读")
        lines.append("=" * 56)

        signal = self.ctx.signal
        if not signal:
            lines.append("  (无信号数据)")
            return "\n".join(lines)

        direction = signal.get('direction', 0)
        confidence = signal.get('confidence', 0)
        pred_raw = signal.get('predicted_return', 0)
        pred_smooth = signal.get('predicted_return_smooth', 0)
        upper = signal.get('upper_threshold', 0)
        lower = signal.get('lower_threshold', 0)
        regime = signal.get('regime_name', '未知')
        model = signal.get('model_used', 'unknown')

        lines.append(f"  市场状态:       {regime}")
        lines.append(f"  使用模型:       {model}")
        lines.append(f"")
        lines.append(f"  原始预测收益:   {pred_raw:+.4f}%")
        lines.append(f"  平滑预测收益:   {pred_smooth:+.4f}%  (EMA span=160)")
        lines.append(f"  上阈值:         {upper:+.4f}%")
        lines.append(f"  下阈值:         {lower:+.4f}%")
        lines.append(f"")

        # Why this signal?
        if direction == 1:
            reason = f"平滑预测 ({pred_smooth:+.4f}%) 突破上阈值 ({upper:+.4f}%)"
            margin = pred_smooth - upper
            lines.append(f"  触发原因: {reason}")
            lines.append(f"  超出幅度: {margin:+.4f}%")
        elif direction == -1:
            reason = f"平滑预测 ({pred_smooth:+.4f}%) 跌破下阈值 ({lower:+.4f}%)"
            margin = lower - pred_smooth
            lines.append(f"  触发原因: {reason}")
            lines.append(f"  超出幅度: {margin:+.4f}%")
        else:
            if pred_smooth > 0:
                gap_up = upper - pred_smooth
                gap_down = pred_smooth - lower
                lines.append(f"  未触发原因: 平滑预测 ({pred_smooth:+.4f}%) 处于")
                lines.append(f"              下阈值 ({lower:+.4f}%) 和 上阈值 ({upper:+.4f}%) 之间")
                lines.append(f"              距上阈值还需 {gap_up:+.4f}%, 距下阈值 {gap_down:+.4f}%")
            else:
                lines.append(f"  未触发原因: 预测值在阈值区间内，无明确方向信号")

        lines.append(f"")
        lines.append(f"  置信度:         {confidence:.1%}")
        if confidence < 0.3:
            lines.append(f"  置信度评估:     低 — 信号不够强，建议观望")
        elif confidence < 0.6:
            lines.append(f"  置信度评估:     中等 — 可小仓试探")
        else:
            lines.append(f"  置信度评估:     高 — 信号明确，可按建议仓位执行")

        lines.append(f"")
        lines.append(f"  建议仓位权重:   {signal.get('suggested_weight', 0):.1%}")

        # Recent prediction trend
        df = self.ctx.df_pred
        if df is not None and len(df) > 0 and 'Pred_Ret' in df.columns:
            recent = df['Pred_Ret'].tail(500)
            lines.append(f"")
            lines.append(f"  近期预测统计 (最近500根K线):")
            lines.append(f"    均值: {recent.mean():+.4f}%")
            lines.append(f"    标准差: {recent.std():.4f}%")
            lines.append(f"    最大值: {recent.max():+.4f}%")
            lines.append(f"    最小值: {recent.min():+.4f}%")
            pos_pct = (recent > 0).mean() * 100
            lines.append(f"    正向比例: {pos_pct:.1f}%")

        lines.append("=" * 56)
        return "\n".join(lines)

    # ================================================================
    # 3. Factor Diagnostics
    # ================================================================
    def factor_diagnostics(self):
        lines = []
        lines.append("=" * 56)
        lines.append("  核心因子诊断 (当前值 vs 历史分布)")
        lines.append("=" * 56)

        imp = self.ctx.importance
        df = self.ctx.df_factors

        if imp is None or df is None:
            lines.append("  (因子数据不可用)")
            return "\n".join(lines)

        # Top 15 important features
        top15 = imp[imp['importance'] > 0].head(15)['feature'].tolist()
        available = [f for f in top15 if f in df.columns]

        if not available:
            lines.append("  (无可用的重要因子)")
            return "\n".join(lines)

        latest = df.iloc[-1]
        lines.append(f"  {'因子':<28s} {'当前值':>10s} {'分位':>6s} {'状态':>8s}")
        lines.append(f"  {'-'*52}")

        alerts = []
        for feat in available[:15]:
            series = df[feat].dropna()
            if len(series) < 100:
                continue
            cur = latest[feat]
            pct = (series < cur).mean() * 100

            # Flag extremes
            if pct > 90:
                status = '!! 极高位'
                alerts.append((feat, '高', pct))
            elif pct > 75:
                status = '!  偏高'
                alerts.append((feat, '高', pct))
            elif pct < 10:
                status = '!! 极低位'
                alerts.append((feat, '低', pct))
            elif pct < 25:
                status = '!  偏低'
                alerts.append((feat, '低', pct))
            else:
                status = '正常'

            lines.append(f"  {feat:<28s} {cur:>10.4f} {pct:>5.1f}% {status:>8s}")

        lines.append("")

        # Alerts summary
        if alerts:
            lines.append(f"  === 异常因子预警 ({len(alerts)} 个) ===")
            for feat, direction, pct in alerts:
                if direction == '高':
                    lines.append(f"    {feat}: 处于历史 {pct:.0f}% 分位，显著偏高")
                else:
                    lines.append(f"    {feat}: 处于历史 {pct:.0f}% 分位，显著偏低")
        else:
            lines.append(f"  所有核心因子均在正常范围内 (25%-75% 分位)")

        lines.append("")
        lines.append(f"  数据覆盖: {len(df):,} 根K线, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        lines.append("=" * 56)
        return "\n".join(lines)

    # ================================================================
    # 4. Macro Landscape
    # ================================================================
    def macro_landscape(self):
        lines = []
        lines.append("=" * 56)
        lines.append("  宏观环境概览")
        lines.append("=" * 56)

        df = self.ctx.df_factors
        mf = self.ctx.macro_factors

        if df is None:
            lines.append("  (数据不可用)")
            return "\n".join(lines)

        latest = df.iloc[-1]

        # Yield curve
        lines.append("  —— 收益率曲线 ——")
        yc_items = [
            ('YC_Slope_10Y_1Y', '10Y-1Y 期限利差'),
            ('YC_Slope_30Y_10Y', '30Y-10Y 期限利差'),
            ('YC_Curvature', '曲线曲度 (Butterfly)'),
            ('YC_Level_ZScore', '10Y利率 Z-Score'),
        ]
        for col, label in yc_items:
            if col in df.columns:
                series = df[col].dropna()
                cur = latest[col]
                pct = (series < cur).mean() * 100 if len(series) > 0 else 50
                interpretation = self._yc_interpret(col, cur, pct)
                lines.append(f"  {label:<20s}: {cur:>+8.4f}  (分位 {pct:.0f}%)  {interpretation}")

        # Cross-market
        lines.append("")
        lines.append("  —— 跨境/跨资产 ——")
        cross_items = [
            ('CN_US_10Y_Spread', '中美10Y利差'),
            ('CN_US_10Y_Spread_Z', '中美利差 Z-Score'),
            ('Risk_On_Off', 'Risk-On/Off'),
        ]
        for col, label in cross_items:
            if col in df.columns:
                series = df[col].dropna()
                cur = latest[col]
                pct = (series < cur).mean() * 100 if len(series) > 0 else 50
                interpretation = self._cross_interpret(col, cur, pct)
                lines.append(f"  {label:<20s}: {cur:>+8.4f}  (分位 {pct:.0f}%)  {interpretation}")

        # Macro momentum
        lines.append("")
        lines.append("  —— 宏观动量 ——")
        macro_items = [
            ('PMI_ZScore', 'PMI Z-Score'),
            ('CPI_Momentum', 'CPI 动量'),
            ('M2_Surprise', 'M2 超预期'),
            ('Macro_Surprise_Composite', '宏观综合意外'),
        ]
        for col, label in macro_items:
            if col in df.columns:
                series = df[col].dropna()
                cur = latest[col]
                pct = (series < cur).mean() * 100 if len(series) > 0 else 50
                interpretation = self._macro_interpret(col, cur, pct)
                lines.append(f"  {label:<20s}: {cur:>+8.4f}  (分位 {pct:.0f}%)  {interpretation}")

        # Macro factor data range
        if mf is not None and len(mf) > 0:
            lines.append(f"")
            lines.append(f"  宏观因子历史: {mf['available_date'].min()} ~ {mf['available_date'].max()}")

        lines.append("=" * 56)
        return "\n".join(lines)

    def _yc_interpret(self, col, cur, pct):
        if 'Slope' in col:
            if cur > 0 and pct > 75:
                return '← 曲线陡峭化'
            elif cur < 0 and pct < 25:
                return '← 曲线倒挂/平坦化'
            else:
                return ''
        elif 'Curvature' in col:
            if pct > 75:
                return '← 蝶式走高'
            elif pct < 25:
                return '← 蝶式走低'
            else:
                return ''
        elif 'ZScore' in col:
            if pct > 90:
                return '← 利率显著偏高'
            elif pct < 10:
                return '← 利率显著偏低'
            else:
                return ''
        return ''

    def _cross_interpret(self, col, cur, pct):
        if 'Spread' in col and 'Z' not in col:
            if cur > 0:
                return '← 中国利率高于美国'
            else:
                return '← 中国利率低于美国 (资本外流压力)'
        elif 'Risk_On' in col:
            if cur > 0:
                return '← Risk-On (风险偏好)'
            else:
                return '← Risk-Off (避险)'
        return ''

    def _macro_interpret(self, col, cur, pct):
        if 'PMI' in col:
            if cur > 0:
                return '← 制造业扩张'
            else:
                return '← 制造业收缩'
        elif 'CPI' in col:
            if cur > 0:
                return '← 通胀上行'
            else:
                return '← 通胀下行/通缩'
        elif 'M2' in col:
            if cur > 0:
                return '← 货币超预期宽松'
            else:
                return '← 货币收紧'
        elif 'Composite' in col:
            if cur > 0.5:
                return '← 宏观整体向好'
            elif cur < -0.5:
                return '← 宏观整体偏弱'
            else:
                return ''
        return ''

    # ================================================================
    # 5. Performance Review
    # ================================================================
    def performance_review(self):
        lines = []
        lines.append("=" * 56)
        lines.append("  策略表现回顾")
        lines.append("=" * 56)

        metrics = self.ctx.metrics
        if not metrics:
            lines.append("  (无回测数据)")
            return "\n".join(lines)

        lines.append(f"  {'指标':<16s} {'数值':>12s}")
        lines.append(f"  {'-'*28}")
        for label, key in [
            ('累计收益', '累计收益'),
            ('年化收益', '年化收益'),
            ('年化波动', '年化波动'),
            ('夏普比率', '夏普比率'),
            ('Calmar比率', 'Calmar比率'),
            ('最大回撤', '最大回撤'),
            ('回撤持续', '回撤持续'),
            ('日胜率', '日胜率'),
            ('年化换手', '年化换手'),
            ('持仓比例', '持仓比例'),
        ]:
            val = metrics.get(key, 'N/A')
            lines.append(f"  {label:<16s} {str(val):>12s}")

        # Interpretation
        lines.append(f"")
        lines.append(f"  —— 综合评估 ——")

        sharpe_str = metrics.get('夏普比率', '0')
        try:
            sharpe = float(sharpe_str)
        except (ValueError, TypeError):
            sharpe = 0

        if sharpe > 3:
            lines.append(f"  夏普 {sharpe:.2f}: 优秀 — 风险调整后收益远超市场平均水平")
        elif sharpe > 1.5:
            lines.append(f"  夏普 {sharpe:.2f}: 良好 — 策略具有稳健的超额收益")
        elif sharpe > 0.5:
            lines.append(f"  夏普 {sharpe:.2f}: 一般 — 策略有一定alpha但空间有限")
        else:
            lines.append(f"  夏普 {sharpe:.2f}: 需优化 — 风险调整后收益不足")

        mdd_str = metrics.get('最大回撤', '0%')
        lines.append(f"  最大回撤 {mdd_str}: 风险控制在可接受范围内")

        win_str = metrics.get('日胜率', '0%')
        lines.append(f"  日胜率 {win_str}: 方向判断准确率")

        lines.append("=" * 56)
        return "\n".join(lines)

    # ================================================================
    # 6. Full Briefing
    # ================================================================
    def full_briefing(self):
        lines = []
        lines.append("#" * 60)
        lines.append("#  TL国债期货策略 — 每日晨报")
        lines.append("#" * 60)

        # Quick overview
        signal = self.ctx.signal
        if signal:
            direction = signal.get('direction', 0)
            dir_map = {1: '做多 LONG', -1: '做空 SHORT', 0: '观望 FLAT'}
            lines.append(f"\n  >>> 今日信号: {dir_map.get(direction)} | "
                         f"置信度: {signal.get('confidence', 0):.1%} | "
                         f"状态: {signal.get('regime_name', 'N/A')}")

        lines.append("")
        lines.append(self.market_snapshot())
        lines.append("")
        lines.append(self.signal_interpretation())
        lines.append("")
        lines.append(self.factor_diagnostics())
        lines.append("")
        lines.append(self.macro_landscape())
        lines.append("")
        lines.append(self.performance_review())

        lines.append("")
        lines.append("#" * 60)
        ts = signal.get('timestamp', 'N/A') if signal else 'N/A'
        lines.append(f"#  报告生成时间: {ts}")
        lines.append("#" * 60)
        return "\n".join(lines)

    # ================================================================
    # 7. AI Market Intelligence
    # ================================================================
    def market_intelligence(self, cache_dir=None):
        """AI-powered analysis: quant data + market news → DeepSeek V4 report."""
        if cache_dir is None:
            cache_dir = os.path.join(self.ctx.base_dir, "data", "macro")

        try:
            analyzer = LLMAnalyzer(self.ctx, cache_dir=cache_dir)
            return analyzer.generate_intelligence_report(save=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            lines = []
            lines.append("=" * 56)
            lines.append("  AI 市场情报分析 (离线模式)")
            lines.append("=" * 56)
            lines.append(f"  [警告] LLM 分析不可用: {e}")
            lines.append("")
            lines.append("  以下为基于规则的量化摘要：")
            lines.append("")
            lines.append(self.signal_interpretation())
            lines.append("")
            lines.append(self.macro_landscape())
            lines.append("=" * 56)
            return "\n".join(lines)

    # ================================================================
    # Helpers
    # ================================================================
    def _latest_ticker(self):
        df = self.ctx.df_factors
        if df is not None and 'ticker' in df.columns:
            return str(df['ticker'].iloc[-1])
        return 'N/A'


# ================================================================
# CLI Entry Point
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='TL策略 Agent 交互层')
    parser.add_argument(
        '--query', type=str, default='snapshot',
        choices=['snapshot', 'signal', 'factors', 'macro', 'performance', 'briefing', 'intelligence'],
        help='查询类型: snapshot=市场快照, signal=信号解读, factors=因子诊断, '
             'macro=宏观环境, performance=策略表现, briefing=完整晨报, intelligence=AI市场情报'
    )
    parser.add_argument(
        '--base-dir', type=str, default=r'D:\桌面\F_Agent',
        help='项目根目录'
    )
    args = parser.parse_args()

    ctx = StrategyContext(args.base_dir)
    analyzer = MarketAnalyzer(ctx)

    query_map = {
        'snapshot': analyzer.market_snapshot,
        'signal': analyzer.signal_interpretation,
        'factors': analyzer.factor_diagnostics,
        'macro': analyzer.macro_landscape,
        'performance': analyzer.performance_review,
        'briefing': analyzer.full_briefing,
        'intelligence': analyzer.market_intelligence,
    }

    result = query_map[args.query]()
    print(result)


if __name__ == '__main__':
    main()
