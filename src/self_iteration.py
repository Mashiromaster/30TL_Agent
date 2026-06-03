# -*- coding: utf-8 -*-
# self_iteration.py — 自我迭代引擎
# 从交易记忆中学习, 检测模型退化, 自动触发重训练/参数调整
#
# 功能:
#   1. 性能监控 — 滚动IC/胜率/夏普跟踪
#   2. 制度漂移检测 — IC连续下滑 → 建议重训练窗口
#   3. 失败模式识别 — 聚类失败交易, 提取特征
#   4. 自动调参 — 基于记忆统计反馈调整信号参数
#   5. 迭代报告 — 生成可操作的优化建议

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict


class PerformanceMonitor:
    """滚动窗口性能监控"""

    def __init__(self, window_days=30):
        self.window_days = window_days

    def compute_rolling_ic(self, df_predictions):
        """计算滚动IC序列"""
        df = df_predictions.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 按日聚合
        df['trade_date'] = df['date'].dt.date
        daily = df.groupby('trade_date').agg({
            'Pred_Ret': 'mean',
            'Target_Ret': 'sum',
        }).dropna()

        if len(daily) < self.window_days:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # 滚动IC
        rolling_ic = daily['Pred_Ret'].rolling(self.window_days).corr(daily['Target_Ret'])
        return rolling_ic, daily

    def detect_regime_drift(self, df_predictions, threshold=-0.01):
        """检测制度漂移: 滚动IC是否持续低于阈值"""
        rolling_ic, daily = self.compute_rolling_ic(df_predictions)
        if len(rolling_ic.dropna()) < 5:
            return False, None

        recent = rolling_ic.dropna().tail(10)
        # 最近10天中有7天IC转负 → 制度漂移
        negative_days = (recent < threshold).sum()
        drift_detected = negative_days >= 7

        drift_info = None
        if drift_detected:
            drift_info = {
                'recent_ic_mean': round(recent.mean(), 4),
                'recent_ic_min': round(recent.min(), 4),
                'negative_ratio': round(negative_days / len(recent), 2),
                'suggested_action': 'retrain',
                'suggested_window_months': self._suggest_window(recent),
            }

        return drift_detected, drift_info

    def _suggest_window(self, ic_series):
        """基于IC下降速度建议新训练窗口"""
        if len(ic_series) < 10:
            return 12
        decline_rate = ic_series.iloc[-1] - ic_series.iloc[0]
        if decline_rate < -0.05:
            return 6  # 快速恶化 → 短窗口
        elif decline_rate < -0.02:
            return 9
        else:
            return 12


class FailurePatternAnalyzer:
    """失败模式分析"""

    def __init__(self):
        self.patterns = []

    def analyze(self, records):
        """从交易记录中分析失败模式"""
        if not records:
            return []

        # 提取已评估记录
        evaluated = [r for r in records if r.get('is_correct') is not None]
        if len(evaluated) < 20:
            return []

        failures = [r for r in evaluated if not r['is_correct']]
        if len(failures) < 5:
            return []

        patterns = []

        # Pattern 1: 特定市场状态的失败率
        regime_fails = defaultdict(lambda: {'total': 0, 'wrong': 0})
        for r in evaluated:
            regime = r.get('market_regime', 0)
            regime_fails[regime]['total'] += 1
            if not r['is_correct']:
                regime_fails[regime]['wrong'] += 1

        for rid, counts in regime_fails.items():
            if counts['total'] >= 5:
                fail_rate = counts['wrong'] / counts['total']
                if fail_rate > 0.5:
                    patterns.append({
                        'type': 'regime_weakness',
                        'regime': {0: '正常', 1: '高波动', 2: '趋势'}.get(rid, str(rid)),
                        'fail_rate': round(fail_rate, 3),
                        'samples': counts['total'],
                        'suggestion': f"在{['正常','高波动','趋势'][rid]}市降低仓位或暂停交易",
                    })

        # Pattern 2: 连续的失败
        streak = 0
        max_consecutive_fails = 0
        fail_streaks = []
        for r in evaluated:
            if not r['is_correct']:
                streak += 1
            else:
                if streak >= 3:
                    fail_streaks.append(streak)
                streak = 0
        if streak >= 3:
            fail_streaks.append(streak)
        max_consecutive_fails = max(fail_streaks) if fail_streaks else 0

        if max_consecutive_fails >= 5:
            patterns.append({
                'type': 'consecutive_fails',
                'max_streak': max_consecutive_fails,
                'total_streaks': len(fail_streaks),
                'suggestion': f"连续失败{max_consecutive_fails}次, 建议暂停交易并检查模型",
            })

        # Pattern 3: 做多vs做空不对称
        long_total = sum(1 for r in evaluated if r.get('direction') == 1)
        short_total = sum(1 for r in evaluated if r.get('direction') == -1)
        long_wrong = sum(1 for r in failures if r.get('direction') == 1)
        short_wrong = sum(1 for r in failures if r.get('direction') == -1)

        if long_total >= 5 and short_total >= 5:
            long_fail = long_wrong / long_total
            short_fail = short_wrong / short_total
            diff = abs(long_fail - short_fail)
            if diff > 0.3:
                worse = '做多' if long_fail > short_fail else '做空'
                patterns.append({
                    'type': 'directional_bias',
                    'long_fail_rate': round(long_fail, 3),
                    'short_fail_rate': round(short_fail, 3),
                    'suggestion': f'{worse}方向显著更差, 考虑单向交易或降低该方向权重',
                })

        return patterns


class SelfIterationEngine:
    """自我迭代引擎 — 交易记忆 → 策略优化闭环"""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.monitor = PerformanceMonitor(window_days=30)
        self.analyzer = FailurePatternAnalyzer()

    # --- 时间衰减权重 (借鉴 Dexter temporal-decay) ---
    @staticmethod
    def temporal_weight(record, half_life_days=30, now=None):
        """对交易记忆应用指数时间衰减 — 近期记录权重更高"""
        if now is None:
            now = datetime.now()
        try:
            dt = datetime.strptime(record.get('trade_dt', ''), '%Y-%m-%d')
        except ValueError:
            return 1.0  # 无法解析日期，给满权重
        age_days = (now - dt).days
        if age_days <= 0:
            return 1.0
        # 指数衰减: weight = exp(-ln(2) * age / half_life)
        return np.exp(-np.log(2) * age_days / half_life_days)

    def weighted_accuracy(self, records, half_life_days=30):
        """计算时间衰减加权准确率"""
        evaluated = [r for r in records if r.get('is_correct') is not None]
        if not evaluated:
            return 0.5, 0
        weights = [self.temporal_weight(r, half_life_days) for r in evaluated]
        correct = [1.0 if r['is_correct'] else 0.0 for r in evaluated]
        weighted_acc = sum(c * w for c, w in zip(correct, weights)) / sum(weights)
        return round(weighted_acc, 4), len(evaluated)

    def load_memory(self):
        """加载交易记忆"""
        memory_path = os.path.join(self.base_dir, "outputs/trade_memory.jsonl")
        if not os.path.exists(memory_path):
            return []
        records = []
        with open(memory_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def load_predictions(self):
        """加载预测数据"""
        pred_path = os.path.join(self.base_dir, "outputs/df_predictions.pkl")
        if not os.path.exists(pred_path):
            return None
        return pd.read_pickle(pred_path)

    def run_diagnostic(self):
        """运行完整诊断 — 返回可操作的优化建议"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': {},
            'drift': {},
            'patterns': [],
            'recommendations': [],
        }

        # 1. 性能分析
        records = self.load_memory()
        df_pred = self.load_predictions()

        if df_pred is not None and len(df_pred) > 0:
            # IC分析
            if 'Target_Ret' in df_pred.columns and 'Pred_Ret' in df_pred.columns:
                mask = df_pred['Target_Ret'].notna() & df_pred['Pred_Ret'].notna()
                if mask.sum() > 100:
                    ic = np.corrcoef(df_pred.loc[mask, 'Pred_Ret'],
                                    df_pred.loc[mask, 'Target_Ret'])[0, 1]
                    report['performance']['overall_ic'] = round(ic, 4)

            # 滚动IC
            rolling_ic, _ = self.monitor.compute_rolling_ic(df_pred)
            recent_ic = rolling_ic.dropna()
            if len(recent_ic) > 0:
                report['performance']['rolling_ic_latest'] = round(recent_ic.iloc[-1], 4)
                report['performance']['rolling_ic_mean_10d'] = round(recent_ic.tail(10).mean(), 4)

            # 制度漂移检测
            drifted, drift_info = self.monitor.detect_regime_drift(df_pred)
            report['drift'] = {
                'detected': drifted,
                'info': drift_info,
            }
            if drifted and drift_info:
                report['recommendations'].append({
                    'priority': 'high',
                    'action': 'retrain',
                    'detail': f"检测到制度漂移 (滚动IC={drift_info['recent_ic_mean']:.4f}), "
                             f"建议用{drift_info['suggested_window_months']}月窗口重新训练",
                })

        # 2. 失败模式分析
        if records:
            patterns = self.analyzer.analyze(records)
            report['patterns'] = patterns
            for p in patterns:
                report['recommendations'].append({
                    'priority': 'medium',
                    'action': p['type'],
                    'detail': p.get('suggestion', ''),
                })

        # 3. 记忆统计 (含时间衰减加权)
        evaluated = [r for r in records if r.get('is_correct') is not None]
        if evaluated:
            correct = sum(1 for r in evaluated if r['is_correct'])
            report['performance']['memory_accuracy'] = round(correct / len(evaluated), 4)
            report['performance']['total_records'] = len(records)
            report['performance']['evaluated_records'] = len(evaluated)

            # 时间衰减加权准确率 (借鉴 Dexter temporal-decay)
            w_acc, _ = self.weighted_accuracy(records, half_life_days=30)
            report['performance']['weighted_accuracy_30d'] = w_acc

            w_acc_14, _ = self.weighted_accuracy(records, half_life_days=14)
            report['performance']['weighted_accuracy_14d'] = w_acc_14

            # 准确率趋势 (最近20 vs 全部)
            recent_20 = evaluated[-20:]
            if recent_20:
                recent_acc = sum(1 for r in recent_20 if r['is_correct']) / len(recent_20)
                report['performance']['recent_20_accuracy'] = round(recent_acc, 4)

                # 如果最近准确率显著下降
                overall_acc = correct / len(evaluated)
                if recent_acc < overall_acc - 0.1:
                    report['recommendations'].append({
                        'priority': 'high',
                        'action': 'signal_tune',
                        'detail': f"近期准确率({recent_acc:.1%})显著低于历史({overall_acc:.1%}), "
                                 f"建议调整排名信号参数",
                    })

        # 4. 自动生成信号参数建议
        if len(records) >= 30:
            param_suggestion = self._suggest_signal_params(records)
            if param_suggestion:
                report['recommendations'].append(param_suggestion)

        return report

    def _suggest_signal_params(self, records):
        """基于记忆统计建议信号参数"""
        evaluated = [r for r in records if r.get('is_correct') is not None]
        if len(evaluated) < 30:
            return None

        correct = [r for r in evaluated if r['is_correct']]
        wrong = [r for r in evaluated if not r['is_correct']]

        if not correct or not wrong:
            return None

        avg_conf_correct = np.mean([r.get('confidence', 0) for r in correct])
        avg_conf_wrong = np.mean([r.get('confidence', 0) for r in wrong])

        # 低置信度信号准确率
        low_conf = [r for r in evaluated if r.get('confidence', 0) < 0.3]
        if low_conf:
            low_acc = sum(1 for r in low_conf if r['is_correct']) / len(low_conf)
            if low_acc < 0.4:
                return {
                    'priority': 'medium',
                    'action': 'confidence_filter',
                    'detail': f"低置信度信号(<0.3)准确率仅{low_acc:.1%}, 建议提高置信度阈值到0.3",
                }

        return None

    def generate_report_text(self, report):
        """生成可读报告文本"""
        lines = [
            "=" * 55,
            "  F_Agent 自我迭代诊断报告",
            f"  生成时间: {report['timestamp'][:19]}",
            "=" * 55,
            "",
        ]

        perf = report.get('performance', {})
        lines.append("【性能概览】")
        if 'overall_ic' in perf:
            lines.append(f"  整体IC: {perf['overall_ic']}")
        if 'rolling_ic_latest' in perf:
            lines.append(f"  最新滚动IC(30天): {perf['rolling_ic_latest']}")
        if 'memory_accuracy' in perf:
            lines.append(f"  交易记忆准确率: {perf['memory_accuracy']:.1%} ({perf.get('evaluated_records', 0)}笔)")
        if 'recent_20_accuracy' in perf:
            lines.append(f"  最近20笔准确率: {perf['recent_20_accuracy']:.1%}")
        lines.append("")

        drift = report.get('drift', {})
        if drift.get('detected'):
            info = drift.get('info', {})
            lines.append(f"【⚠ 制度漂移警报】")
            lines.append(f"  状态: 已检测到制度漂移")
            lines.append(f"  近期IC均值: {info.get('recent_ic_mean', 'N/A')}")
            lines.append(f"  负IC占比: {info.get('negative_ratio', 0):.0%}")
            lines.append(f"  建议窗口: {info.get('suggested_window_months', 12)}个月")
            lines.append("")

        patterns = report.get('patterns', [])
        if patterns:
            lines.append("【失败模式识别】")
            for p in patterns:
                lines.append(f"  [{p['type']}] {p.get('suggestion', '')}")
            lines.append("")

        recs = report.get('recommendations', [])
        if recs:
            lines.append("【优化建议】")
            for r in recs:
                priority_icon = {'high': '⚠', 'medium': '•', 'low': ' '}.get(r['priority'], ' ')
                lines.append(f"  {priority_icon} [{r['priority'].upper()}] {r['action']}: {r['detail']}")
            lines.append("")

        return '\n'.join(lines)

    def auto_adjust_signal_params(self):
        """自动调整信号参数 — 基于近期表现统计"""
        records = self.load_memory()
        if len(records) < 30:
            return None

        evaluated = [r for r in records if r.get('is_correct') is not None]
        if len(evaluated) < 20:
            return None

        # 扫描不同参数组合在记忆中的表现
        # (这里用记忆中的confidence筛选来模拟参数效果)
        adjustments = {}

        # 高/低置信度分组
        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
            filtered = [r for r in evaluated if r.get('confidence', 0) >= threshold]
            if len(filtered) >= 10:
                acc = sum(1 for r in filtered if r['is_correct']) / len(filtered)
                adjustments[f'conf_threshold_{threshold}'] = {
                    'accuracy': round(acc, 3),
                    'samples': len(filtered),
                }

        if not adjustments:
            return None

        # 找到准确率最高的阈值
        best_key = max(adjustments, key=lambda k: adjustments[k]['accuracy'])
        best_threshold = float(best_key.split('_')[-1])

        return {
            'suggested_confidence_threshold': best_threshold,
            'expected_accuracy': adjustments[best_key]['accuracy'],
            'all_results': adjustments,
        }
