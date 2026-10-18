# -*- coding: utf-8 -*-
# main.py (Phase 2 - 训练/推理模式分离)

import os
import sys
import json
import argparse
import time
from contextlib import redirect_stdout
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import BASE_DIR
from intelligence_events import redact_sensitive_text


def main():
    from strategy_runtime import DEFAULT_STRATEGY_ID, STRATEGY_SPECS

    parser = argparse.ArgumentParser(description='TL国债期货量化策略系统')
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'inference', 'iterate', 'evolve', 'moe', 'research', 'hybrid', 'readiness', 'prepare-factors', 'train-candidate', 'monitor', 'rollback'],
        help='运行模式: train=训练+门禁+版本发布, inference=统一策略实时信号, research=Research Agent闭环, hybrid=受限混合决策研究, monitor=舆情监控, rollback=切回历史release'
    )
    parser.add_argument(
        '--strategy', default=DEFAULT_STRATEGY_ID,
        choices=sorted(STRATEGY_SPECS),
        help='SignalPolicy策略版本；默认使用统一TL策略',
    )
    parser.add_argument(
        '--watch', action='store_true',
        help='monitor模式持续轮询；默认只运行一次',
    )
    parser.add_argument(
        '--interval-seconds', type=int, default=None,
        help='monitor --watch轮询间隔，最小30秒；默认读取监控配置',
    )
    parser.add_argument('--monitor-config', help='监控配置JSON路径，默认config/bond_monitor.json')
    parser.add_argument('--notify', action='store_true',
                        help='monitor模式显式允许发送已配置的外部通知；默认只生成本地记录')
    parser.add_argument(
        '--no-sentiment', action='store_true',
        help='research模式跳过舆情抓取',
    )
    parser.add_argument(
        '--no-llm', action='store_true',
        help='research跳过LLM研究推理；monitor禁用可选语义分析和Jev',
    )
    parser.add_argument(
        '--rebuild-factors', action='store_true',
        help='research模式显式重建因子；默认复用已验证因子文件',
    )
    parser.add_argument(
        '--release-id',
        help='rollback模式要重新激活的不可变release ID',
    )
    parser.add_argument('--network-mode', choices=['offline', 'cache', 'online'], default='offline',
                        help='hybrid模型网络权限；默认offline，online显式允许向配置服务发送受限研究快照')
    parser.add_argument('--question', default='宏观与微观证据是否支持当前TL策略？', help='hybrid研究问题')
    parser.add_argument('--as-of', help='hybrid时点截止时间；不使用此时点之后的行情或证据')
    parser.add_argument('--no-persist', action='store_true', help='hybrid不保存运行审计（在线Jev仍可写其模型缓存）')
    parser.add_argument('--factor-path', help='hybrid/readiness/train-candidate显式选择独立因子候选；不覆盖生产因子')
    parser.add_argument('--minute-snapshot', help='prepare-factors显式选择隔离分钟快照ID；不替换原始行情')
    args = parser.parse_args()
    preparation_modes = {'readiness', 'prepare-factors', 'train-candidate'}
    json_modes = preparation_modes | {'hybrid'}
    if args.minute_snapshot and args.mode != 'prepare-factors':
        parser.error('--minute-snapshot only applies to prepare-factors')
    if args.factor_path and args.mode not in {'hybrid', 'readiness', 'train-candidate'}:
        parser.error('--factor-path only applies to hybrid/readiness/train-candidate')
    if args.mode in preparation_modes and (args.network_mode != 'offline' or args.rebuild_factors or args.notify or args.watch):
        parser.error('preparation modes are explicit local-only actions; conflicting network/rebuild/notify/watch flags')
    if args.mode in {'prepare-factors', 'train-candidate'} and args.no_persist:
        parser.error('candidate preparation writes isolated artifacts; --no-persist is not supported')
    if args.mode == 'hybrid' and args.no_llm and args.network_mode != 'offline':
        parser.error('hybrid --no-llm requires --network-mode offline')
    if args.mode == 'hybrid' and args.rebuild_factors:
        parser.error('hybrid does not rebuild factors; use the explicit factor pipeline separately')

    TICK_SUBDIR = "data/tick"

    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] 目录不存在: {BASE_DIR}")
        return

    # Hybrid stdout is one machine-readable report; diagnostics use stderr.
    banner_stream = sys.stderr if args.mode in json_modes else sys.stdout
    print("\n" + "#" * 60, file=banner_stream)
    print(f"#  TL期货投机策略 — 模式: {args.mode.upper()}" + " " * 22 + "#", file=banner_stream)
    print("#" * 60, file=banner_stream)
    if args.mode in preparation_modes:
        from research_readiness import inspect_research_readiness
        with redirect_stdout(sys.stderr):
            if args.mode == 'prepare-factors':
                from factor_candidates import build_local_factor_candidate
                report = build_local_factor_candidate(BASE_DIR, **({'minute_snapshot_id': args.minute_snapshot} if args.minute_snapshot else {}))
            else:
                readiness = inspect_research_readiness(BASE_DIR, factor_path=args.factor_path, strategy_id=args.strategy)
                if args.mode == 'readiness':
                    report = readiness
                elif readiness['training']['eligible']:
                    from LightGBM_model import train_candidate
                    report = train_candidate(BASE_DIR, factor_path=args.factor_path, strategy_id=args.strategy)
                else:
                    report = {'status': 'blocked', 'published': False, 'reason': 'training_prerequisites_not_met', 'readiness': readiness}
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    elif args.mode == 'train':
        run_train(BASE_DIR, TICK_SUBDIR, strategy_id=args.strategy)
    elif args.mode == 'inference':
        run_inference_mode(BASE_DIR, strategy_id=args.strategy)
    elif args.mode == 'iterate':
        run_iteration(BASE_DIR)
    elif args.mode == 'evolve':
        run_evolution(BASE_DIR)
    elif args.mode == 'moe':
        run_moe(BASE_DIR)
    elif args.mode == 'research':
        run_research_mode(
            BASE_DIR,
            strategy_id=args.strategy,
            rebuild_factors=args.rebuild_factors,
            use_llm=not args.no_llm,
            monitor_sentiment=not args.no_sentiment,
        )
    elif args.mode == 'hybrid':
        from hybrid_runtime import HybridConfig
        from research_agent import ResearchAgent

        report = ResearchAgent(BASE_DIR, strategy_id=args.strategy).run_hybrid_cycle(
            config=HybridConfig(network_mode=args.network_mode), question=args.question,
            as_of=args.as_of, persist=not args.no_persist, factor_path=args.factor_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    elif args.mode == 'monitor':
        if not run_monitor_mode(
            BASE_DIR,
            watch=args.watch,
            interval_seconds=args.interval_seconds,
            config_path=args.monitor_config,
            allow_notifications=args.notify,
            use_llm=not args.no_llm,
        ):
            sys.exit(1)
    elif args.mode == 'rollback':
        if not args.release_id:
            parser.error("--mode rollback requires --release-id")
        run_rollback(BASE_DIR, args.release_id)


def run_train(base_dir, tick_subdir, *, strategy_id=None):
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
    if not LightGBM_model.run_process(
        base_dir,
        max_lookback_months=9,
        time_decay_half_life=60,
        strategy_id=strategy_id,
    ):
        print("[STOP] 模型训练失败")
        sys.exit(1)

    # Step 3: 策略回测
    print("\n[3/3] 策略回测")
    if not backtest.run_process(base_dir, strategy_id=strategy_id):
        print("[STOP] 回测失败")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  训练管道完成" + " " * 48 + "#")
    print("#" * 60)


def run_inference_mode(base_dir, *, strategy_id=None):
    from inference import run_inference

    if not run_inference(base_dir, strategy_id=strategy_id):
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


def run_research_mode(
    base_dir,
    *,
    strategy_id=None,
    rebuild_factors=False,
    use_llm=True,
    monitor_sentiment=True,
):
    """Research Agent 完整闭环: 因子→信号→融合决策→记录记忆→回填校准→报告"""
    from research_agent import run_research

    report = run_research(
        base_dir,
        rebuild_factors=rebuild_factors,
        use_llm=use_llm,
        monitor_sentiment=monitor_sentiment,
        strategy_id=strategy_id,
    )
    if not report.get('base_signal') or report.get('step_status', {}).get('signal') != 'ok':
        print("[STOP] Research Agent 未生成基础信号")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  Research Agent 闭环完成" + " " * 36 + "#")
    print("#" * 60)


def run_monitor_mode(
    base_dir,
    *,
    watch=False,
    interval_seconds=None,
    monitor=None,
    config_path=None,
    allow_notifications=False,
    use_llm=True,
    sleep_fn=None,
    max_iterations=None,
):
    """Run the read-only sentiment monitor once or as a bounded polling loop."""
    if monitor is None:
        from dataclasses import replace
        from bond_monitor_config import MonitorConfig
        from jev_client import JevClient
        from sentiment_monitor import SentimentMonitor

        try:
            config = MonitorConfig.load(base_dir, config_path)
            if not use_llm:
                config = replace(config, semantic_mode="off")
            monitor = SentimentMonitor(base_dir, config=config,
                jev_adapter=JevClient(base_dir) if use_llm else None,
                allow_notifications=allow_notifications)
        except Exception as exc:
            print(f"[Monitor CONFIG ERROR] {redact_sensitive_text(exc)}")
            return False
    configured_interval = getattr(getattr(monitor, "config", None), "interval_seconds", 600)
    interval = max(30, int(interval_seconds if interval_seconds is not None else configured_interval))
    sleep_fn = sleep_fn or time.sleep
    attempts = 0
    last_success = False
    consecutive_failures = 0
    while True:
        attempts += 1
        try:
            run = monitor.run_once()
            last_success = run.status == "ok"
            consecutive_failures = 0 if last_success else consecutive_failures + 1
            print(
                "[Monitor] "
                f"status={run.status} documents={run.documents_inserted} "
                f"assessments={run.assessments_created} alerts={run.alerts_opened}"
            )
        except KeyboardInterrupt:
            print("\n[Monitor] 已停止")
            return True
        except Exception as exc:
            last_success = False
            consecutive_failures += 1
            print(f"[Monitor ERROR] {redact_sensitive_text(exc)}")
            if not watch:
                return False
        if not watch:
            return last_success
        if max_iterations is not None and attempts >= int(max_iterations):
            return last_success
        try:
            delay = min(
                interval * (2 ** min(6, max(0, consecutive_failures - 1))),
                max(interval, 3600),
            )
            sleep_fn(delay)
        except KeyboardInterrupt:
            print("\n[Monitor] 已停止")
            return True


def run_rollback(base_dir, release_id):
    """Reactivate one verified immutable release and refresh legacy views."""
    from model_release import ReleaseManager

    manager = ReleaseManager(
        os.path.join(base_dir, "artifacts", "models"),
        legacy_model_path=os.path.join(base_dir, "models", "trained_model.pkl"),
    )
    signature = manager.activate_release(release_id, project_legacy=True)
    projections = {
        "predictions": os.path.join(base_dir, "outputs", "df_predictions.pkl"),
        "feature_importance": os.path.join(
            base_dir, "outputs", "feature_importance.csv"
        ),
        "evaluation": os.path.join(
            base_dir, "outputs", "model_ablation_report.json"
        ),
    }
    for name, destination in projections.items():
        try:
            manager.project_active_artifact(name, destination)
        except Exception as exc:
            print(
                f"[Rollback WARNING] {name} compatibility view: "
                f"{redact_sensitive_text(exc)}"
            )
    print(f"[Rollback] active release: {signature.release_id}")


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
    """展示旧自我进化入口的退役状态与安全替代路径。"""
    from self_evolution import run_status

    run_status(base_dir)

    print("\n旧版 AdapterStack 已停用，不能作为生产或影子预测源。")
    print("安全流程:")
    print("  1. python main.py --mode train       — 生成候选并执行统一回测门禁")
    print("  2. python main.py --mode inference   — 仅使用当前 managed release")
    print("  3. python main.py --mode iterate     — 运行只读漂移诊断")


if __name__ == '__main__':
    main()
