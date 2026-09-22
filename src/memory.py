# -*- coding: utf-8 -*-
# memory.py — Trading Memory System (V1.0)
# 记录每日交易决策、预测vs实际对比、反思归因

import pandas as pd
import numpy as np
import os
import json
import tempfile
from datetime import datetime

from strategy_runtime import SignalPolicy, get_strategy_spec


class TradingMemory:
    """Trading decision journal with historical backfill and reflection."""

    def __init__(self, base_dir, strategy_id=None, *, policy=None):
        self.base_dir = base_dir
        self.memory_path = os.path.join(base_dir, "outputs/trade_memory.jsonl")
        self.strategy_spec = get_strategy_spec(strategy_id)
        self.policy = policy or SignalPolicy(self.strategy_spec)

    def _prediction_path(self):
        from model_release import ReleaseManager

        manager = ReleaseManager(
            os.path.join(self.base_dir, "artifacts", "models"),
            legacy_model_path=os.path.join(
                self.base_dir, "models", "trained_model.pkl"
            ),
        )
        managed = manager.active_artifact_path("predictions")
        if managed is not None:
            return os.fspath(managed)
        return os.path.join(self.base_dir, "outputs", "df_predictions.pkl")

    @staticmethod
    def _prediction_release_id(df):
        """Require one provenance domain for a prediction snapshot."""
        if 'Model_Release_ID' not in df.columns or len(df) == 0:
            return 'legacy-unversioned'
        if df['Model_Release_ID'].isna().any():
            raise ValueError("prediction frame has missing Model_Release_ID values")
        release_ids = df['Model_Release_ID'].astype(str).unique()
        if len(release_ids) != 1:
            raise ValueError("prediction frame mixes multiple model releases")
        return str(release_ids[0])

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

    def _rewrite_all(self, records):
        output_dir = os.path.dirname(self.memory_path)
        os.makedirs(output_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix='.trade_memory.', suffix='.tmp', dir=output_dir
        )
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
            os.replace(temporary_path, self.memory_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    def get_recent(self, n=10, *, include_other_strategies=False):
        records = self._load_all()
        if not include_other_strategies:
            fingerprint = self.strategy_spec.fingerprint()
            records = [
                record for record in records
                if record.get('strategy_fingerprint') == fingerprint
            ]
        return records[-n:]

    # ─── Core: record daily signal ─────────────────────────

    def record_signal(self, signal_dict):
        """Append one day's trading decision to the memory journal."""
        existing = self._load_all()
        trade_dt = str(signal_dict.get('timestamp', ''))[:10]
        decision_id = str(signal_dict.get('decision_id', '') or '')
        strategy_fingerprint = str(
            signal_dict.get('strategy_fingerprint', 'legacy-unknown')
        )
        if decision_id:
            existing = [r for r in existing if r.get('decision_id') != decision_id]
        else:
            existing = [
                r for r in existing
                if not (
                    r.get('trade_dt') == trade_dt
                    and r.get('strategy_fingerprint', 'legacy-unknown')
                    == strategy_fingerprint
                )
            ]

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
            'pred_rank_pct': signal_dict.get('pred_rank_pct', 0.5),
            'upper_threshold': signal_dict.get('upper_threshold', 0),
            'lower_threshold': signal_dict.get('lower_threshold', 0),
            'suggested_weight': signal_dict.get('suggested_weight', 0),
            'model_used': signal_dict.get('model_used', ''),
            'decision_id': decision_id,
            'strategy_id': signal_dict.get('strategy_id', 'legacy-unknown'),
            'strategy_version': signal_dict.get('strategy_version'),
            'strategy_fingerprint': strategy_fingerprint,
            'model_release_id': signal_dict.get(
                'model_release_id', 'legacy-unversioned'
            ),
            'effective_from': signal_dict.get('effective_from'),
            # Fusion feedback (defaults fall back to raw signal → backward compatible)
            'fused_direction': signal_dict.get('fused_direction', signal_dict.get('direction', 0)),
            'fused_weight': signal_dict.get('fused_weight', signal_dict.get('suggested_weight', 0)),
            'fusion_level': signal_dict.get('fusion_level', 'none'),
            'fusion_reasons': signal_dict.get('fusion_reasons', []),
            'research_proposed_direction': signal_dict.get(
                'research_proposed_direction'
            ),
            'research_proposed_weight': signal_dict.get(
                'research_proposed_weight'
            ),
            'actual_return': None,
            'is_correct': None,
            'recorded_at': datetime.now().isoformat(),
        }

        existing.append(record)
        self._rewrite_all(existing)
        return record

    # ─── Historical backfill ───────────────────────────────

    def backfill_from_predictions(self):
        """Rebuild research memory with the canonical SignalPolicy."""
        pred_path = self._prediction_path()
        if not os.path.exists(pred_path):
            print(f"[Memory] 找不到预测文件: {pred_path}")
            return 0

        df = pd.read_pickle(pred_path)
        required = {'date', 'close', 'Pred_Ret'}
        missing = sorted(required.difference(df.columns))
        if missing:
            print(f"[Memory] 预测文件缺少列: {missing}")
            return 0
        if 'trade_dt' not in df.columns:
            df = df.copy()
            df['trade_dt'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        release_id = self._prediction_release_id(df)
        evaluated = self.policy.evaluate_frame(
            df, model_release_id=release_id
        )
        current_records = []
        for _, day_bars in evaluated.groupby(evaluated['trade_dt'].astype(str)):
            last = day_bars.iloc[-1]
            if not bool(last.get('Decision_Ready', False)):
                continue
            timestamp = pd.Timestamp(last['date'])
            decision_id = self.policy.decision_id_for_evaluated_row(
                last, model_release_id=release_id
            )
            direction = int(last['Direction'])
            regime = int(last.get('Market_Regime', 0) or 0)
            actual = last.get('Target_Ret')
            actual_value = (
                round(float(actual), 6) if pd.notna(actual) else None
            )
            record = {
                'trade_dt': str(last['trade_dt'])[:10],
                'timestamp': str(timestamp),
                'effective_from': (
                    str(last['Effective_From'])
                    if pd.notna(last['Effective_From']) else None
                ),
                'close': float(last['close']) if pd.notna(last['close']) else None,
                'market_regime': regime,
                'regime_name': {0: '正常', 1: '高波动', 2: '趋势'}.get(regime, '未知'),
                'direction': direction,
                'direction_name': {1: '做多', -1: '做空', 0: '观望'}[direction],
                'confidence': round(float(last['Confidence']), 4),
                'predicted_return_smooth': round(float(last['Pred_Smooth']), 6),
                'pred_rank_pct': (
                    round(float(last['Pred_Rank']), 4)
                    if pd.notna(last['Pred_Rank']) else None
                ),
                'upper_threshold': (
                    round(float(last['Upper_Q']), 6)
                    if pd.notna(last['Upper_Q']) else None
                ),
                'lower_threshold': (
                    round(float(last['Lower_Q']), 6)
                    if pd.notna(last['Lower_Q']) else None
                ),
                'suggested_weight': round(abs(float(last['Target_Position'])), 4),
                'model_used': str(last.get('Model_Used', 'base')),
                'decision_id': decision_id,
                'strategy_id': self.strategy_spec.strategy_id,
                'strategy_version': self.strategy_spec.version,
                'strategy_fingerprint': self.strategy_spec.fingerprint(),
                'model_release_id': release_id,
                'actual_return': actual_value,
                'actual_source': 'prediction_label' if actual_value is not None else None,
                'is_correct': (
                    bool(np.sign(direction) == np.sign(actual_value))
                    if direction and actual_value is not None else None
                ),
                'recorded_at': datetime.now().isoformat(),
            }
            current_records.append(record)

        fingerprint = self.strategy_spec.fingerprint()
        retained_records = [
            record for record in self._load_all()
            if record.get('strategy_fingerprint') != fingerprint
        ]
        self._rewrite_all(retained_records + current_records)
        print(
            f"[Memory] 历史回填完成: {len(current_records)} 条 "
            f"({self.strategy_spec.strategy_id})"
        )
        return len(current_records)

    # ─── Outcome tracking ──────────────────────────────────

    def update_actuals(self):
        """Backfill actual_return and is_correct for records whose outcomes are now known."""
        pred_path = self._prediction_path()
        if not os.path.exists(pred_path):
            print("[Memory] 无可用于回填实际结果的预测文件")
            return

        df = pd.read_pickle(pred_path)
        if 'trade_dt' not in df.columns and 'date' in df.columns:
            df = df.copy()
            df['trade_dt'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        release_id = self._prediction_release_id(df)
        records = self._load_all()
        if not records:
            return

        updated = 0
        for r in records:
            if r.get('strategy_fingerprint') != self.strategy_spec.fingerprint():
                continue
            if r.get('model_release_id', 'legacy-unversioned') != release_id:
                continue
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
        all_records = self._load_all()
        fingerprint = self.strategy_spec.fingerprint()
        records = [
            record for record in all_records
            if record.get('strategy_fingerprint') == fingerprint
        ]
        if not records:
            return {'error': '当前策略版本无记忆记录'}

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

        # Shadow research context. Executed accuracy belongs to the immutable
        # base decision; proposal accuracy evaluates the non-executed research
        # suggestion separately so a shadow proposal is never labelled as a
        # trade result.
        by_shadow_context = {}
        for level in ['none', 'rule', 'rag', 'llm']:
            subset = [r for r in evaluated if r.get('fusion_level', 'none') == level]
            if subset:
                executed = [
                    r for r in subset
                    if r.get('fused_direction', r.get('direction', 0)) != 0
                    and r.get('actual_return') is not None
                ]
                proposals = [
                    r for r in subset
                    if r.get('research_proposed_direction') not in (None, 0)
                    and r.get('actual_return') is not None
                ]
                executed_correct = sum(
                    int(
                        np.sign(r.get('fused_direction', r.get('direction', 0)))
                        == np.sign(r['actual_return'])
                    )
                    for r in executed
                )
                proposal_correct = sum(
                    int(
                        np.sign(r['research_proposed_direction'])
                        == np.sign(r['actual_return'])
                    )
                    for r in proposals
                )
                by_shadow_context[level] = {
                    'executed_accuracy': (
                        round(executed_correct / len(executed), 4)
                        if executed else None
                    ),
                    'executed_acted': len(executed),
                    'proposal_accuracy': (
                        round(proposal_correct / len(proposals), 4)
                        if proposals else None
                    ),
                    'proposal_acted': len(proposals),
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
            'by_shadow_context': by_shadow_context,
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

        contexts = stats.get('by_shadow_context', {})
        if contexts:
            lines.append("")
            lines.append("--- 影子研究情境：基础执行 vs 非执行建议 ---")
            for level, info in contexts.items():
                executed = (
                    f"{info['executed_accuracy']:.2%}"
                    if info['executed_accuracy'] is not None else "N/A"
                )
                proposal = (
                    f"{info['proposal_accuracy']:.2%}"
                    if info['proposal_accuracy'] is not None else "N/A"
                )
                lines.append(
                    f"  {level}: 基础执行={executed}"
                    f"(n={info['executed_acted']}) | "
                    f"影子建议={proposal}(n={info['proposal_acted']})"
                )

        return '\n'.join(lines)

    def llm_reflection(self, *, api_key=None):
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
            analyzer.api_key = (
                str(api_key)
                if api_key is not None
                else os.environ.get("DEEPSEEK_API_KEY", "")
            )
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
    sys.path.insert(0, os.path.dirname(__file__))
    from config import BASE_DIR
    base = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR
    run_backfill(base)
