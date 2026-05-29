# -*- coding: utf-8 -*-
# main.py (Phase 2 - 训练/推理模式分离)

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='TL国债期货量化策略系统')
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'inference'],
        help='运行模式: train=因子构建+训练+回测, inference=因子更新+实时信号'
    )
    args = parser.parse_args()

    BASE_DIR = r"D:\桌面\F_Agent"
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


if __name__ == '__main__':
    main()
