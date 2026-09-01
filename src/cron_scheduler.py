# -*- coding: utf-8 -*-
# cron_scheduler.py — 定时任务调度系统 (借鉴 Dexter cron tool)
#
# 为 F_Agent 添加自动化调度能力:
#   - 每日行情更新 (收盘后)
#   - 每周模型评估
#   - 每月自动重训练 + 参数优化
#   - 制度漂移警报后自动触发
#
# 使用:
#   python cron_scheduler.py list           # 列出所有任务
#   python cron_scheduler.py run daily      # 手动运行每日任务
#   python cron_scheduler.py run weekly     # 手动运行每周任务
#   python cron_scheduler.py daemon         # 启动后台调度守护进程

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = r"D:\桌面\F_Agent"
CRON_STORE = os.path.join(BASE_DIR, "outputs", "cron_jobs.json")


# ============================================================
# 任务定义 (借鉴 Dexter 的 cron + every + at 模式)
# ============================================================

DEFAULT_JOBS = [
    {
        "id": "daily_update",
        "name": "每日行情更新",
        "kind": "cron",
        "expr": "0 16 * * 1-5",  # 工作日16:00 (收盘后)
        "command": "update",
        "description": "拉取最新分钟行情数据并追加到原始文件",
        "enabled": True,
    },
    {
        "id": "daily_signal",
        "name": "每日信号生成",
        "kind": "cron",
        "expr": "0 16 * * 1-5",
        "command": "inference",
        "description": "基于最新数据生成交易信号",
        "enabled": True,
    },
    {
        "id": "weekly_eval",
        "name": "每周模型评估",
        "kind": "cron",
        "expr": "0 17 * * 5",  # 周五17:00
        "command": "eval",
        "description": "滚动窗口评估，检测模型退化",
        "enabled": True,
    },
    {
        "id": "weekly_memory",
        "name": "每周交易记忆回填",
        "kind": "cron",
        "expr": "0 17 * * 5",
        "command": "memory",
        "description": "从预测文件回填交易记忆并更新实际结果",
        "enabled": True,
    },
    {
        "id": "monthly_retrain",
        "name": "月度模型重训练",
        "kind": "cron",
        "expr": "0 9 1 * *",  # 每月1号09:00
        "command": "retrain",
        "description": "用最新数据全量重训练+窗口扫描+回测",
        "enabled": True,
    },
    {
        "id": "monthly_iteration",
        "name": "月度自迭代诊断",
        "kind": "cron",
        "expr": "0 10 1 * *",
        "command": "iterate",
        "description": "运行自我迭代引擎，检测制度漂移，优化参数",
        "enabled": True,
    },
    # === 自我进化任务 (LoRA-inspired) ===
    {
        "id": "weekly_adapt",
        "name": "每周适配器训练 (LoRA)",
        "kind": "cron",
        "expr": "0 8 * * 6",  # 周六08:00
        "command": "weekly_adapt",
        "description": "冻结基模型，基于近30天反馈训练残差适配器(20棵树/深度2)，推入适配器堆栈",
        "enabled": True,
    },
    {
        "id": "bimonthly_retrain",
        "name": "双月全量重训练",
        "kind": "cron",
        "expr": "0 9 1 */2 *",  # 每两月1号09:00
        "command": "bimonthly_retrain",
        "description": "吸收所有适配器经验，全量重训练基模型，清空适配器堆栈",
        "enabled": True,
    },
]


# ============================================================
# 调度器核心
# ============================================================

class CronScheduler:
    """轻量级定时任务调度器 (借鉴 Dexter cron)"""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.store_path = CRON_STORE
        self.jobs = self._load_jobs()

    def _load_jobs(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 首次初始化
        jobs = DEFAULT_JOBS
        self._save_jobs(jobs)
        return jobs

    def _save_jobs(self, jobs=None):
        if jobs is None:
            jobs = self.jobs
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

    def list_jobs(self):
        print(f"\n{'='*60}")
        print(f"  定时任务列表")
        print(f"{'='*60}")
        for j in self.jobs:
            status = "✅" if j.get('enabled', True) else "⏸️"
            last_run = j.get('last_run', '从未')
            print(f"  {status} [{j['id']}] {j['name']}")
            print(f"    调度: {j['kind']} {j.get('expr', '')}")
            print(f"    命令: {j['command']}")
            print(f"    上次运行: {last_run}")
            if j.get('last_error'):
                print(f"    ⚠️ 上次错误: {j['last_error'][:80]}")
            print()

    def _mark_run(self, job_id, success, error=None):
        for j in self.jobs:
            if j['id'] == job_id:
                j['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if success:
                    j.pop('last_error', None)
                else:
                    j['last_error'] = str(error)[:200]
                self._save_jobs()
                return

    def _run_command(self, job):
        """执行单个任务"""
        src_dir = os.path.join(self.base_dir, "src")
        cmd = job['command']

        print(f"\n{'='*50}")
        print(f"  执行任务: {job['name']} ({job['id']})")
        print(f"  命令: {cmd}")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        start = time.time()

        try:
            if cmd == 'update':
                result = subprocess.run(
                    [sys.executable, "update_market_data.py"],
                    cwd=src_dir, capture_output=True, text=True, timeout=300
                )
            elif cmd == 'inference':
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "inference"],
                    cwd=src_dir, capture_output=True, text=True, timeout=300
                )
            elif cmd == 'eval':
                result = subprocess.run(
                    [sys.executable, "eval_runner.py", "--window", "30"],
                    cwd=src_dir, capture_output=True, text=True, timeout=300
                )
            elif cmd == 'memory':
                result = subprocess.run(
                    [sys.executable, "-c",
                     "from memory import TradingMemory; m=TradingMemory(r'D:\\桌面\\F_Agent'); "
                     "m.backfill_from_predictions(); print('OK')"],
                    cwd=src_dir, capture_output=True, text=True, timeout=120
                )
            elif cmd == 'retrain':
                result = subprocess.run(
                    [sys.executable, "retrain_optimized.py"],
                    cwd=src_dir, capture_output=True, text=True, timeout=900
                )
            elif cmd == 'iterate':
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "iterate"],
                    cwd=src_dir, capture_output=True, text=True, timeout=300
                )
            elif cmd == 'weekly_adapt':
                result = subprocess.run(
                    [sys.executable, "self_evolution.py", "weekly", BASE_DIR],
                    cwd=src_dir, capture_output=True, text=True, timeout=600
                )
            elif cmd == 'bimonthly_retrain':
                result = subprocess.run(
                    [sys.executable, "self_evolution.py", "bimonthly", BASE_DIR],
                    cwd=src_dir, capture_output=True, text=True, timeout=1800
                )
            else:
                raise ValueError(f"未知命令: {cmd}")

            elapsed = time.time() - start

            if result.returncode == 0:
                last_lines = result.stdout.strip().split('\n')[-5:]
                print(f"  ✅ 成功 (耗时 {elapsed:.0f}s)")
                for line in last_lines:
                    print(f"     {line[:100]}")
                self._mark_run(job['id'], success=True)
            else:
                print(f"  ❌ 失败 (exit={result.returncode}, 耗时 {elapsed:.0f}s)")
                print(f"     stderr: {result.stderr[:200]}")
                self._mark_run(job['id'], success=False, error=result.stderr)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ⏱️ 超时 (>{elapsed:.0f}s)")
            self._mark_run(job['id'], success=False, error="Timeout")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 异常: {e}")
            self._mark_run(job['id'], success=False, error=str(e))

    def run_job(self, job_id):
        """手动运行指定任务"""
        for j in self.jobs:
            if j['id'] == job_id:
                self._run_command(j)
                return
        print(f"任务不存在: {job_id}")

    def run_all(self):
        """运行所有已启用的任务"""
        enabled = [j for j in self.jobs if j.get('enabled', True)]
        print(f"运行 {len(enabled)} 个任务...")
        for j in enabled:
            self._run_command(j)

    def daemon(self, check_interval=60):
        """后台守护进程模式 (每60秒检查是否需要执行)"""
        print(f"\n🔄 F_Agent 调度守护进程已启动 (检查间隔: {check_interval}s)")
        print(f"   按 Ctrl+C 停止")
        print(f"   配置文件: {self.store_path}")
        print()

        try:
            while True:
                now = datetime.now()
                # 简化版: 每分钟检查一次
                # 生产环境建议用 schedule 库或系统 crontab
                for j in self.jobs:
                    if not j.get('enabled', True):
                        continue
                    last_run = j.get('last_run', '')
                    # 简单逻辑: 如果今天还没运行过，且当前时间匹配
                    if j['kind'] == 'cron':
                        hour_match = self._cron_hour_match(j['expr'], now)
                        if hour_match:
                            today_str = now.strftime('%Y-%m-%d')
                            if not last_run.startswith(today_str):
                                print(f"\n[{now.strftime('%H:%M:%S')}] 触发: {j['name']}")
                                self._run_command(j)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n守护进程已停止")

    @staticmethod
    def _cron_hour_match(expr, now):
        """简单cron小时匹配 (0 16 * * 1-5 → 工作日16:00)"""
        parts = expr.split()
        if len(parts) < 5:
            return False
        try:
            minute = int(parts[0])
            hour = int(parts[1])
            # 星期: 1-5 表示周一到周五 (cron: 0=Sun)
            weekday_part = parts[4]

            if now.minute != minute or now.hour != hour:
                return False
            if '-' in weekday_part:
                lo, hi = map(int, weekday_part.split('-'))
                # Python: weekday() 0=Mon, cron: 0=Sun
                py_weekday = now.weekday() + 1  # 转为 1=Mon
                if py_weekday == 7:
                    py_weekday = 0  # 转为 0=Sun
                # 简化: 直接比较 python weekday
                if not (lo <= now.weekday() + 1 <= hi):
                    return False
            return True
        except:
            return False


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='F_Agent 定时任务调度系统')
    parser.add_argument('action', choices=['list', 'run', 'run-all', 'daemon'],
                        help='操作: list=列出, run=运行指定, run-all=全运行, daemon=守护进程')
    parser.add_argument('job_id', nargs='?', default=None,
                        help='任务ID (list/run时可用)')
    args = parser.parse_args()

    scheduler = CronScheduler(BASE_DIR)

    if args.action == 'list':
        scheduler.list_jobs()
    elif args.action == 'run':
        if args.job_id:
            scheduler.run_job(args.job_id)
        else:
            print("请指定任务ID, 例如: python cron_scheduler.py run daily_update")
    elif args.action == 'run-all':
        scheduler.run_all()
    elif args.action == 'daemon':
        scheduler.daemon()


if __name__ == '__main__':
    main()
