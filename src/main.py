# -*- coding: utf-8 -*-
# main.py (Phase 2 - 训练/推理模式分离)

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import BASE_DIR


def main():
    parser = argparse.ArgumentParser(description='TL国债期货量化策略系统')
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'inference', 'iterate', 'evolve', 'moe', 'research'],
        help='运行模式: train=因子构建+训练+回测, inference=因子更新+实时信号, iterate=自我迭代诊断, evolve=自我进化(每周适配/双月重训), moe=双层MoE训练+基线对比, research=Research Agent 完整闭环(因子→信号→融合→记忆→回填)'
    )
    args = parser.parse_args()

    TICK_SUBDIR = "data/tick"

    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] 目录不存在: {BASE_DIR}")
        return

    print("\n" + "#" * 60)
    print(f"#  TL期货投机策略 — 模式: {args.mode.upper()}" + " " * 22 + "#")
    print("#" * 60)

    if args.mode == 'train':
        run_train(BASE_DIR, TICK_SUBDIR)
    elif args.mode == 'inference':
        run_inference_mode(BASE_DIR)
    elif args.mode == 'iterate':
        run_iteration(BASE_DIR)
    elif args.mode == 'evolve':
        run_evolution(BASE_DIR)
    elif args.mode == 'moe':
        run_moe(BASE_DIR)
    elif args.mode == 'research':
        run_research_mode(BASE_DIR)


def run_train(base_dir, tick_subdir):
    import factor_extraction
    import LightGBM_model
    import backtest

    # Step 1: 因子构建
    print("\n[1/3] 因子构建")
    if not factor_extraction.run_process(base_dir, tick_subdir=tick_subdir):
        print("[STOP] 因子构建失败")
        sys.exit(1)

    # Step 2: 模型训练
    print("\n[2/3] 模型训练")
    if not LightGBM_model.run_process(base_dir, max_lookback_months=9, time_decay_half_life=60):
        print("[STOP] 模型训练失败")
        sys.exit(1)

    # Step 3: 策略回测
    print("\n[3/3] 策略回测")
    if not backtest.run_process(base_dir):
        print("[STOP] 回测失败")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  训练管道完成" + " " * 48 + "#")
    print("#" * 60)


def run_inference_mode(base_dir):
    from inference import run_inference

    if not run_inference(base_dir):
        print("[STOP] 推理失败")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  实时信号生成完成" + " " * 44 + "#")
    print("#" * 60)


def run_iteration(base_dir):
    """运行自我迭代诊断"""
    from self_iteration import SelfIterationEngine

    engine = SelfIterationEngine(base_dir)
    report = engine.run_diagnostic()
    report_text = engine.generate_report_text(report)

    print(report_text)

    # Save report
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"iteration_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n诊断报告已保存: {report_path}")

    # Auto-suggest actions
    adj = engine.auto_adjust_signal_params()
    if adj:
        print(f"\n自动参数建议: 置信度阈值={adj['suggested_confidence_threshold']}")


def run_research_mode(base_dir):
    """Research Agent 完整闭环: 因子→信号→融合决策→记录记忆→回填校准→报告"""
    from research_agent import run_research

    run_research(base_dir, rebuild_factors=True, use_llm=True)

    print("\n" + "#" * 60)
    print("#  Research Agent 闭环完成" + " " * 36 + "#")
    print("#" * 60)


def run_moe(base_dir):
    """双层 LightGBM MoE 训练 (需先运行过 train 生成 df_factors.pkl)"""
    import moe_model

    if not moe_model.run_process(base_dir, max_lookback_months=12, time_decay_half_life=90):
        print("[STOP] MoE 训练失败")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  MoE 训练完成" + " " * 48 + "#")
    print("#" * 60)


def run_evolution(base_dir):
    """运行自我进化引擎 (LoRA-inspired自主适配)"""
    from self_evolution import SelfEvolutionEngine, run_status

    engine = SelfEvolutionEngine(base_dir)

    # Show current status first
    run_status(base_dir)

    print("\n可用子命令:")
    print("  python self_evolution.py weekly      — 每周适配 (训练新适配器)")
    print("  python self_evolution.py bimonthly   — 双月全量重训练")
    print("  python self_evolution.py status      — 查看引擎状态")
    print("  python self_evolution.py predict     — 运行组合预测+保存")
    print("\n推荐流程:")
    print("  1. python self_evolution.py status   — 检查状态")
    print("  2. python self_evolution.py weekly   — 每周适配")
    print("  3. python self_evolution.py predict  — 用组合模型预测")
    print("\n或通过定时任务自动运行:")
    print("  python cron_scheduler.py run weekly_adapt")
    print("  python cron_scheduler.py run bimonthly_retrain")


if __name__ == '__main__':
    main()
