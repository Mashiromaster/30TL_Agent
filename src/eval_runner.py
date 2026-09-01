# -*- coding: utf-8 -*-
# eval_runner.py — 模型评估套件 (借鉴 Dexter eval system)
#
# 功能:
#   1. 滚动窗口评估 — 在连续时间段上持续测试模型IC
#   2. 分市场状态评估 — 正常/高波动/趋势市分别评分
#   3. 退化检测 — 对比当前窗口与基线窗口IC
#   4. 评估报告 — 自动生成Markdown报告
#
# 使用:
#   python eval_runner.py                        # 全量评估
#   python eval_runner.py --window 30            # 30天滚动窗口
#   python eval_runner.py --baseline 2025-07     # 指定基线期

import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict


BASE_DIR = r"D:\桌面\F_Agent"


class ModelEvaluator:
    """滚动窗口模型评估器"""

    def __init__(self, base_dir, window_days=30, step_days=10):
        self.base_dir = base_dir
        self.window_days = window_days
        self.step_days = step_days

    def load_data(self):
        pred_path = os.path.join(self.base_dir, "outputs", "df_predictions.pkl")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")
        df = pd.read_pickle(pred_path)
        df['date'] = pd.to_datetime(df['date'])
        df['trade_date'] = df['date'].dt.date
        return df

    def compute_window_metrics(self, df_window):
        """计算单个窗口的评估指标"""
        mask = df_window['Target_Ret'].notna() & df_window['Pred_Ret'].notna()
        if mask.sum() < 50:
            return None

        pred = df_window.loc[mask, 'Pred_Ret']
        actual = df_window.loc[mask, 'Target_Ret']

        # IC
        ic = np.corrcoef(pred, actual)[0, 1]

        # MSE / MAE
        mse = np.mean((pred - actual) ** 2)
        mae = np.mean(np.abs(pred - actual))

        # Hit rate (sign match)
        signs_match = np.sign(pred) == np.sign(actual)
        hit_rate = signs_match.mean()

        # Directional accuracy (when both sides predict non-zero)
        nonzero = (pred != 0) & (actual != 0)
        if nonzero.sum() > 0:
            dir_acc = (np.sign(pred[nonzero]) == np.sign(actual[nonzero])).mean()
        else:
            dir_acc = 0

        # Prediction statistics
        pred_std = pred.std()
        pred_range = pred.max() - pred.min()

        return {
            'ic': round(ic, 4),
            'mse': round(mse, 6),
            'mae': round(mae, 6),
            'hit_rate': round(hit_rate, 4),
            'dir_acc': round(dir_acc, 4),
            'pred_std': round(pred_std, 6),
            'pred_range': round(pred_range, 6),
            'n_samples': mask.sum(),
        }

    def compute_regime_metrics(self, df_window):
        """分市场状态评估"""
        if 'Market_Regime' not in df_window.columns:
            return {}

        results = {}
        for rid, rname in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
            df_r = df_window[df_window['Market_Regime'] == rid]
            metrics = self.compute_window_metrics(df_r)
            if metrics:
                results[rname] = metrics
        return results

    def rolling_evaluate(self, df):
        """滚动窗口评估"""
        all_dates = sorted(df['trade_date'].unique())
        results = []

        start_idx = 0
        window_end = None

        for i in range(0, len(all_dates) - self.window_days + 1, self.step_days):
            window_dates = all_dates[i:i + self.window_days]
            df_window = df[df['trade_date'].isin(window_dates)]

            metrics = self.compute_window_metrics(df_window)
            if metrics is None:
                continue

            regime_metrics = self.compute_regime_metrics(df_window)

            window_start = window_dates[0]
            window_end = window_dates[-1]

            results.append({
                'window_start': str(window_start),
                'window_end': str(window_end),
                'n_days': len(window_dates),
                **metrics,
                'regime_metrics': regime_metrics,
            })

        return results

    def compare_baseline(self, results, baseline_start=None):
        """对比基线期与最近期的表现"""
        if not results:
            return None

        # Baseline: first complete window
        baseline = results[0]
        # Latest: last complete window
        latest = results[-1]

        changes = {}
        for metric in ['ic', 'hit_rate', 'dir_acc', 'mae']:
            if metric in baseline and metric in latest:
                delta = latest[metric] - baseline[metric]
                pct = delta / (abs(baseline[metric]) + 1e-9)
                changes[metric] = {
                    'baseline': baseline[metric],
                    'latest': latest[metric],
                    'delta': round(delta, 4),
                    'pct_change': round(pct, 4),
                    'degraded': delta < 0 if metric != 'mae' else delta > 0,  # lower IC/hit_rate = bad; higher MAE = bad
                }

        return {
            'baseline_window': f"{baseline['window_start']} ~ {baseline['window_end']}",
            'latest_window': f"{latest['window_start']} ~ {latest['window_end']}",
            'changes': changes,
        }

    def detect_degradation(self, results, threshold=0.02):
        """检测模型退化: 连续N个窗口IC下降"""
        if len(results) < 3:
            return False, None

        ics = [r['ic'] for r in results]
        recent = ics[-3:]

        # 最近3个窗口IC持续下降
        if len(recent) >= 3 and all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            drop = ics[0] - ics[-1]
            if drop > threshold:
                return True, {
                    'ic_trend': [round(x, 4) for x in ics],
                    'total_drop': round(drop, 4),
                    'severity': 'high' if drop > 0.05 else 'medium',
                }

        return False, None

    def generate_report(self, results, comparison, degradation):
        """生成Markdown评估报告"""
        lines = [
            "# F_Agent 模型评估报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"评估模式: {self.window_days}天滚动窗口, {self.step_days}天步长",
            "",
            "## 评估摘要",
            "",
        ]

        if results:
            latest = results[-1]
            lines.append(f"- 评估窗口数: {len(results)}")
            lines.append(f"- 最新窗口: {latest['window_start']} ~ {latest['window_end']} ({latest['n_days']}天)")
            lines.append(f"- 最新IC: {latest['ic']:.4f}")
            lines.append(f"- 最新Hit Rate: {latest['hit_rate']:.2%}")
            lines.append(f"- 最新MAE: {latest['mae']:.6f}")
            lines.append("")

        # Degradation warning
        if degraded:
            info = degradation[1]
            lines.append("## ⚠️ 退化警告")
            lines.append(f"- IC趋势: {' → '.join(str(x) for x in info['ic_trend'])}")
            lines.append(f"- 总降幅: {info['total_drop']:.4f}")
            lines.append(f"- 严重程度: **{info['severity']}**")
            lines.append("")
            if info['severity'] == 'high':
                lines.append("> **建议**: 立即执行重训练")
            lines.append("")

        # Baseline comparison
        if comparison:
            lines.append("## 基线对比")
            lines.append(f"- 基线期: {comparison['baseline_window']}")
            lines.append(f"- 最新期: {comparison['latest_window']}")
            lines.append("")
            lines.append("| 指标 | 基线 | 最新 | 变化 | 状态 |")
            lines.append("|------|------|------|------|------|")
            for metric, info in comparison['changes'].items():
                status = '⚠️ 恶化' if info['degraded'] else '✅ 改善'
                lines.append(
                    f"| {metric} | {info['baseline']:.4f} | {info['latest']:.4f} "
                    f"| {info['delta']:+.4f} ({info['pct_change']:+.1%}) | {status} |"
                )
            lines.append("")

        # Rolling IC table
        if results:
            lines.append("## 滚动IC序列")
            lines.append("")
            lines.append("| 窗口 | IC | Hit Rate | MAE | Dir Acc | Samples |")
            lines.append("|------|-----|----------|-----|---------|---------|")
            for r in results:
                lines.append(
                    f"| {r['window_start']}~{r['window_end']} "
                    f"| {r['ic']:.4f} | {r['hit_rate']:.2%} | {r['mae']:.6f} "
                    f"| {r['dir_acc']:.2%} | {r['n_samples']} |"
                )

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='F_Agent 模型评估套件')
    parser.add_argument('--window', type=int, default=30, help='窗口天数 (默认30)')
    parser.add_argument('--step', type=int, default=10, help='步长天数 (默认10)')
    parser.add_argument('--output', type=str, default=None, help='报告输出路径')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  F_Agent 模型评估 (借鉴 Dexter eval system)")
    print(f"  窗口={args.window}天, 步长={args.step}天")
    print(f"{'='*60}")

    evaluator = ModelEvaluator(BASE_DIR, window_days=args.window, step_days=args.step)

    try:
        df = evaluator.load_data()
        date_range = f"{df['date'].min().date()} ~ {df['date'].max().date()}"
        print(f"\n数据范围: {date_range} ({len(df)} 行)")

        results = evaluator.rolling_evaluate(df)
        print(f"评估窗口数: {len(results)}")

        if results:
            # Latest window summary
            latest = results[-1]
            print(f"\n最新窗口 ({latest['window_start']} ~ {latest['window_end']}):")
            print(f"  IC: {latest['ic']:.4f} | Hit Rate: {latest['hit_rate']:.2%} | "
                  f"MAE: {latest['mae']:.6f} | Dir Acc: {latest['dir_acc']:.2%}")

            # Regime breakdown
            regime_metrics = latest.get('regime_metrics', {})
            if regime_metrics:
                print(f"\n分状态IC:")
                for rname, m in regime_metrics.items():
                    print(f"  {rname}: IC={m['ic']:.4f} (n={m['n_samples']})")

        # Baseline comparison
        comparison = evaluator.compare_baseline(results)
        if comparison:
            print(f"\n基线对比:")
            for metric, info in comparison['changes'].items():
                status = '⚠️' if info['degraded'] else '✅'
                print(f"  {status} {metric}: {info['baseline']:.4f} → {info['latest']:.4f} "
                      f"({info['delta']:+.4f})")

        # Degradation check
        degraded, info = evaluator.detect_degradation(results)
        if degraded:
            print(f"\n⚠️ 检测到模型退化! IC趋势: {' → '.join(str(x) for x in info['ic_trend'])}")
            print(f"   建议: 执行重训练")
        else:
            print(f"\n✅ 未检测到模型退化")

        # Generate & save report
        report = evaluator.generate_report(results, comparison, (degraded, info) if degraded else (False, None))
        output_path = args.output or os.path.join(BASE_DIR, "outputs",
                                                   f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n评估报告已保存: {output_path}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
