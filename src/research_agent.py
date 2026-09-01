# -*- coding: utf-8 -*-
# research_agent.py — 自动化 Research Agent 闭环编排 (V1.0)
#
# spec: docs/superpowers/specs/2026-09-01-research-agent-loop-design.md
#
# 把散落的子系统串成一条可审计的自动化闭环:
#   ① 因子更新 → ② 信号生成 → ③ 融合决策 → ④ 记录记忆 → ⑤ 回填校准 → ⑥ 闭环报告
#
# 各子系统保持单一职责; 本类只负责编排协调。每步独立 try/except 降级,
# 单步失败不中断整链,记入 step_status。

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


class ResearchAgent:
    """Research Agent 闭环编排器。

    使用:
        agent = ResearchAgent(base_dir)
        report = agent.run_cycle(rebuild_factors=True, use_llm=True)
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_cycle(self, rebuild_factors=True, use_llm=True):
        """运行完整闭环。返回 report dict (含 step_status / fused_signal / direction_changed)。"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'step_status': {},   # step_name -> 'ok' | 'skipped:reason' | 'error:msg'
            'base_signal': None,
            'fused_signal': None,
            'direction_changed': False,
        }

        # ── ① 因子更新 (可选; 缺因子文件则硬失败) ──
        factor_file = os.path.join(self.output_dir, "df_factors.pkl")
        if rebuild_factors:
            try:
                import factor_extraction
                ok = factor_extraction.run_process(self.base_dir, tick_subdir="data/tick")
                report['step_status']['factors'] = 'ok' if ok else 'error:run_process returned False'
            except Exception as e:
                report['step_status']['factors'] = f'error:{e}'
        else:
            report['step_status']['factors'] = 'skipped:rebuild_factors=False'

        if not os.path.exists(factor_file):
            report['step_status']['factors'] = 'error:df_factors.pkl 不存在'
            self._save_report(report)
            print("[ResearchAgent] 因子文件缺失，闭环中止")
            return report

        # ── ② 信号生成 (MoE/base 自动路由) ──
        base_signal = None
        try:
            import pandas as pd
            from inference import SignalGenerator
            model_path = os.path.join(self.base_dir, "models", "trained_model.pkl")
            if not os.path.exists(model_path):
                report['step_status']['signal'] = 'error:trained_model.pkl 不存在'
            else:
                df_factors = pd.read_pickle(factor_file)
                gen = SignalGenerator(model_path)
                base_signal, _ = gen.generate_signal(df_factors)
                report['base_signal'] = base_signal
                report['step_status']['signal'] = 'ok'
        except Exception as e:
            report['step_status']['signal'] = f'error:{e}'

        if base_signal is None:
            self._save_report(report)
            print("[ResearchAgent] 信号生成失败，闭环中止")
            return report

        # ── ③ 融合决策 (Memory规则 + RAG政策 + LLM推理，各层内部已降级) ──
        fused = None
        try:
            from signal_fusion import run_fusion
            fused = run_fusion(base_dir=self.base_dir, base_signal=base_signal, use_llm=use_llm)
            if fused is None:
                report['step_status']['fusion'] = 'skipped:run_fusion returned None'
            else:
                report['fused_signal'] = fused.to_dict()
                report['direction_changed'] = (fused.adjusted_direction != fused.raw_direction)
                report['step_status']['fusion'] = f'ok:level={fused.fusion_level}'
        except Exception as e:
            report['step_status']['fusion'] = f'error:{e}'

        # ── ④ 记录到记忆 (含 fusion 字段回流) ──
        try:
            from memory import TradingMemory
            mem = TradingMemory(self.base_dir)
            record_dict = dict(base_signal)
            if fused is not None:
                record_dict['fused_direction'] = fused.adjusted_direction
                record_dict['fused_weight'] = fused.adjusted_weight
                record_dict['fusion_level'] = fused.fusion_level
                record_dict['fusion_reasons'] = list(fused.reasons)[:5]
            mem.record_signal(record_dict)
            report['step_status']['record'] = 'ok'
        except Exception as e:
            report['step_status']['record'] = f'error:{e}'

        # ── ⑤ 回填校准 (有实际收益时) ──
        try:
            from memory import TradingMemory
            mem = TradingMemory(self.base_dir)
            mem.update_actuals()
            report['step_status']['backfill'] = 'ok'
        except Exception as e:
            report['step_status']['backfill'] = f'error:{e}'

        # ── ⑥ 闭环报告 ──
        self._save_report(report)
        self._print_summary(report)
        return report

    def _save_report(self, report):
        dt = str(report['timestamp'])[:10]
        path = os.path.join(self.output_dir, f"research_cycle_{dt}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report['_report_path'] = path
        except Exception as e:
            print(f"[ResearchAgent] 报告保存失败: {e}")

    def _print_summary(self, report):
        print("\n" + "=" * 56)
        print("  Research Agent 闭环报告")
        print("=" * 56)
        for step, status in report['step_status'].items():
            mark = 'OK ' if status.startswith('ok') else ('-- ' if status.startswith('skipped') else '!! ')
            print(f"  [{mark}] {step:10s} {status}")
        fs = report.get('fused_signal')
        if fs:
            dmap = {1: '做多', -1: '做空', 0: '观望'}
            print(f"  ---")
            print(f"  原始方向: {dmap.get(fs.get('raw_direction'), '?')} "
                  f"→ 融合方向: {dmap.get(fs.get('adjusted_direction'), '?')} "
                  f"(仓位 {fs.get('adjusted_weight', 0):.1%}, 层级 {fs.get('fusion_level')})")
            if report['direction_changed']:
                print(f"  ⚠ 融合调整了信号方向")
        print("=" * 56)


def run_research(base_dir, rebuild_factors=True, use_llm=True):
    """便捷入口。"""
    agent = ResearchAgent(base_dir)
    return agent.run_cycle(rebuild_factors=rebuild_factors, use_llm=use_llm)
