# -*- coding: utf-8 -*-
# memory.py — Trading Memory System (V1.0)
# 记录每日交易决策、预测vs实际对比、反思归因

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


class TradingMemory:
    """Trading decision journal with historical backfill and reflection."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.memory_path = os.path.join(base_dir, "outputs/trade_memory.jsonl")
        self.signal_smooth_span = 160

        self.thresholds = {
            0: {'upper': 0.90, 'lower': 0.05},
            1: {'upper': 0.80, 'lower': 0.20},
            2: {'upper': 0.80, 'lower': 0.20},
        }

    # ─── I/O ──────────────────────────────────────────────

    def _load_all(self):
        if not os.path.exists(self.memory_path):
            return []
        records = []
        with open(self.memory_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _append_record(self, record):
        with open(self.memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def _rewrite_all(self, records):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')

    def get_recent(self, n=10):
        records = self._load_all()
        return records[-n:]

    # ─── Core: record daily signal ─────────────────────────

    def record_signal(self, signal_dict):
        """Append one day's trading decision to the memory journal."""
        existing = self._load_all()
        trade_dt = str(signal_dict.get('timestamp', ''))[:10]
        # Dedup: overwrite if same trade_dt exists
        existing = [r for r in existing if r.get('trade_dt') != trade_dt]

        record = {
            'trade_dt': trade_dt,
            'timestamp': signal_dict.get('timestamp', ''),
            'close': signal_dict.get('close'),
            'market_regime': signal_dict.get('market_regime', 0),
            'regime_name': signal_dict.get('regime_name', ''),
            'direction': signal_dict.get('direction', 0),
            'direction_name': signal_dict.get('direction_name', ''),
            'confidence': signal_dict.get('confidence', 0),
            'predicted_return_smooth': signal_dict.get('predicted_return_smooth', 0),
            'upper_threshold': signal_dict.get('upper_threshold', 0),
            'lower_threshold': signal_dict.get('lower_threshold', 0),
            'suggested_weight': signal_dict.get('suggested_weight', 0),
            'model_used': signal_dict.get('model_used', ''),
            'actual_return': None,
            'is_correct': None,
            'recorded_at': datetime.now().isoformat(),
        }

        existing.append(record)
        self._rewrite_all(existing)
        return record

    # ─── Historical backfill ───────────────────────────────

    def backfill_from_predictions(self):
        """Reconstruct daily signals from df_predictions.pkl using the
        same EMA + expanding-window quantile logic as SignalGenerator."""
        pred_path = os.path.join(self.base_dir, "outputs/df_predictions.pkl")
        if not os.path.exists(pred_path):
            print(f"[Memory] 找不到预测文件: {pred_path}")
            return 0

        df = pd.read_pickle(pred_path)
        required = ['date', 'trade_dt', 'close', 'Pred_Ret', 'Target_Ret', 'Market_Regime']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[Memory] 预测文件缺少列: {missing}")
            return 0

        df = df.sort_values('date').reset_index(drop=True)

        # EMA smooth (same as SignalGenerator)
        df['Pred_Smooth'] = df['Pred_Ret'].ewm(span=self.signal_smooth_span, adjust=False).mean()
        pred_lagged = df['Pred_Smooth'].shift(1)

        # Expanding-window quantile thresholds
        upper_q = pred_lagged.expanding(min_periods=100).quantile(
            self.thresholds[0]['upper'])
        lower_q = pred_lagged.expanding(min_periods=100).quantile(
            self.thresholds[0]['lower'])

        df['Upper_Q'] = upper_q
        df['Lower_Q'] = lower_q

        # Group by trade_dt, take last bar of each day as the signal
        unique_dates = df['trade_dt'].unique()
        pred_std = df['Pred_Smooth'].std()

        records = []
        for dt in unique_dates:
            day_bars = df[df['trade_dt'] == dt]
            if len(day_bars) == 0:
                continue

            last = day_bars.iloc[-1]
            smooth = last['Pred_Smooth']
            upper = last['Upper_Q']
            lower = last['Lower_Q']
            regime = int(last['Market_Regime']) if pd.notna(last['Market_Regime']) else 0

            if pd.notna(smooth) and pd.notna(upper) and pd.notna(lower):
                if smooth > upper:
                    direction = 1
                elif smooth < lower:
                    direction = -1
                else:
                    direction = 0
            else:
                direction = 0
                upper = 0.0
                lower = 0.0

            if direction == 1 and pred_std > 0:
                confidence = min(1.0, (smooth - upper) / (pred_std + 1e-9))
            elif direction == -1 and pred_std > 0:
                confidence = min(1.0, (lower - smooth) / (pred_std + 1e-9))
            else:
                confidence = 0.0

            weight_map = {0: 1.0, 1: 0.8, 2: 0.8}
            suggested_weight = weight_map.get(regime, 1.0) * confidence

            record = {
                'trade_dt': str(dt)[:10],
                'timestamp': str(last['date']),
                'close': float(last['close']) if pd.notna(last.get('close')) else None,
                'market_regime': regime,
                'regime_name': ['正常', '高波动', '趋势'][regime],
                'direction': direction,
                'direction_name': {1: '做多', -1: '做空', 0: '观望'}[direction],
                'confidence': round(float(confidence), 4),
                'predicted_return_smooth': round(float(smooth), 6) if pd.notna(smooth) else 0,
                'upper_threshold': round(float(upper), 6) if pd.notna(upper) else 0,
                'lower_threshold': round(float(lower), 6) if pd.notna(lower) else 0,
                'suggested_weight': round(float(suggested_weight), 4),
                'model_used': 'base',
                'actual_return': None,
                'is_correct': None,
                'recorded_at': datetime.now().isoformat(),
            }
            records.append(record)

        # Write all records
        self._rewrite_all(records)
        print(f"[Memory] 历史回填完成: {len(records)} 条记录")

        # Backfill actual outcomes
        self.update_actuals()
        return len(records)

    # ─── Outcome tracking ──────────────────────────────────

    def update_actuals(self):
        """Backfill actual_return and is_correct for records whose outcomes are now known."""
        pred_path = os.path.join(self.base_dir, "outputs/df_predictions.pkl")
        if not os.path.exists(pred_path):
            print("[Memory] 无可用于回填实际结果的预测文件")
            return

        df = pd.read_pickle(pred_path)
        records = self._load_all()
        if not records:
            return

        updated = 0
        for r in records:
            if r.get('actual_return') is not None:
                continue

            dt = r.get('trade_dt', '')
            # Match on date portion only (YYYY-MM-DD)
            day_bars = df[df['trade_dt'].astype(str).str[:10] == str(dt)[:10]]
            if len(day_bars) == 0:
                continue

            # Use the last bar's Target_Ret as the actual outcome
            last = day_bars.iloc[-1]
            actual = last.get('Target_Ret', None)
            if pd.isna(actual):
                continue

            r['actual_return'] = round(float(actual), 6)
            direction = r.get('direction', 0)
            if direction != 0:
                r['is_correct'] = bool(np.sign(direction) == np.sign(actual))
            updated += 1

        if updated > 0:
            self._rewrite_all(records)
            print(f"[Memory] 实际结果回填: {updated} 条")

    # ─── Reflection ────────────────────────────────────────

    def reflection_stats(self):
        """Rule-based accuracy analysis by regime and direction."""
        records = self._load_all()
        if not records:
            return {'error': '无记忆记录'}

        evaluated = [r for r in records if r.get('is_correct') is not None]
        if not evaluated:
            return {'error': '无已评估记录（is_correct 均为空）'}

        total = len(evaluated)
        correct = sum(1 for r in evaluated if r['is_correct'])
        overall_acc = correct / total

        # By regime
        by_regime = {}
        for regime_id in [0, 1, 2]:
            subset = [r for r in evaluated if r.get('market_regime') == regime_id]
            if subset:
                acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                by_regime[['正常', '高波动', '趋势'][regime_id]] = {
                    'accuracy': round(acc, 4),
                    'count': len(subset),
                }

        # By direction
        by_direction = {}
        for d, name in [(1, '做多'), (-1, '做空')]:
            subset = [r for r in evaluated if r.get('direction') == d]
            if subset:
                acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                by_direction[name] = {
                    'accuracy': round(acc, 4),
                    'count': len(subset),
                }

        # By regime × direction
        by_regime_dir = {}
        for regime_id in [0, 1, 2]:
            rname = ['正常', '高波动', '趋势'][regime_id]
            by_regime_dir[rname] = {}
            for d, dname in [(1, '做多'), (-1, '做空')]:
                subset = [r for r in evaluated
                          if r.get('market_regime') == regime_id and r.get('direction') == d]
                if subset:
                    acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                    by_regime_dir[rname][dname] = {
                        'accuracy': round(acc, 4),
                        'count': len(subset),
                    }

        # Streak analysis
        recent = evaluated[-30:]
        streak = 0
        max_streak = 0
        for r in recent:
            if r['is_correct']:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # Recent trend (last 10)
        recent_10 = evaluated[-10:]
        recent_acc = sum(1 for r in recent_10 if r['is_correct']) / len(recent_10) if recent_10 else 0

        return {
            'total_records': len(records),
            'evaluated': total,
            'overall_accuracy': round(overall_acc, 4),
            'by_regime': by_regime,
            'by_direction': by_direction,
            'by_regime_direction': by_regime_dir,
            'recent_10_accuracy': round(recent_acc, 4),
            'current_win_streak': streak,
            'max_win_streak': max_streak,
        }

    def _format_stats_text(self, stats):
        """Format reflection_stats as readable Chinese text."""
        if 'error' in stats:
            return stats['error']

        lines = [
            "=" * 40,
            "  交易记忆 · 规则化统计",
            "=" * 40,
            f"总记录: {stats['total_records']} | 已评估: {stats['evaluated']}",
            f"整体准确率: {stats['overall_accuracy']:.2%}",
            f"最近10天准确率: {stats['recent_10_accuracy']:.2%}",
            f"当前连胜: {stats['current_win_streak']} | 最大连胜: {stats['max_win_streak']}",
            "",
            "--- 按市场状态 ---",
        ]
        for regime, info in stats.get('by_regime', {}).items():
            lines.append(f"  {regime}: {info['accuracy']:.2%} (n={info['count']})")

        lines.append("")
        lines.append("--- 按方向 ---")
        for dname, info in stats.get('by_direction', {}).items():
            lines.append(f"  {dname}: {info['accuracy']:.2%} (n={info['count']})")

        lines.append("")
        lines.append("--- 状态 × 方向交叉 ---")
        for regime, dirs in stats.get('by_regime_direction', {}).items():
            parts = []
            for dname, info in dirs.items():
                parts.append(f"{dname}:{info['accuracy']:.2%}(n={info['count']})")
            lines.append(f"  {regime}: {' | '.join(parts)}")

        return '\n'.join(lines)

    def llm_reflection(self):
        """Use DeepSeek to analyze recent memory records and identify failure patterns.
        Returns a Chinese narrative reflection text."""
        try:
            from llm_intelligence import LLMAnalyzer
        except ImportError:
            return self._format_stats_text(self.reflection_stats())

        stats = self.reflection_stats()
        if 'error' in stats:
            return stats['error']

        recent = self.get_recent(20)
        recent_text = '\n'.join(
            f"  {r['trade_dt']} | {r['regime_name']} | {r['direction_name']} "
            f"| 预测={r['predicted_return_smooth']:.4f}% "
            f"| 实际={r.get('actual_return', '?')} "
            f"| {'✓' if r.get('is_correct') else '✗' if r.get('is_correct') == False else '—'}"
            for r in recent
        )

        stats_text = self._format_stats_text(stats)

        prompt = f"""基于以下TL国债期货量化策略的交易记忆数据，进行归因反思分析：

## 统计摘要
{stats_text}

## 最近20条交易记录
{recent_text}

请分析：
1. 模型在哪些市场状态下判断最准确/最不稳定？
2. 做多 vs 做空信号，哪个方向可靠性更高？为什么？
3. 最近预测质量是在改善还是恶化？有什么规律？
4. 是否有明显的失败模式（例如：高波动市做多、趋势转折点等）？
5. 对策略改进的2-3条具体建议

请用简洁中文回答，重点放在可操作的归因分析上。"""

        try:
            analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
            analyzer.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not analyzer.api_key:
                return self._format_stats_text(stats) + "\n\n[LLM 反思不可用: 未设置 DEEPSEEK_API_KEY]"

            import openai
            client = openai.OpenAI(
                api_key=analyzer.api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是专业中国国债期货量化策略分析师，擅长从交易记录中归因分析模型弱点和失败模式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._format_stats_text(stats) + f"\n\n[LLM 反思调用失败: {e}]"


def run_backfill(base_dir):
    """Quick backfill from predictions for initialization."""
    mem = TradingMemory(base_dir)
    n = mem.backfill_from_predictions()
    if n > 0:
        stats = mem.reflection_stats()
        print(mem._format_stats_text(stats))
    return n


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else r"D:\桌面\F_Agent"
    run_backfill(base)
