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
import copy
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from config import BASE_DIR
from intelligence_events import redact_sensitive_text

CRON_STORE = os.path.join(BASE_DIR, "outputs", "cron_jobs.json")


# ============================================================
# 任务定义 (借鉴 Dexter 的 cron + every + at 模式)
# ============================================================

DEFAULT_JOBS = [
    {
        "id": "sentiment_monitor",
        "name": "TL 舆情增量监控",
        "kind": "cron",
        "expr": "*/10 * * * 1-5",
        "command": "monitor",
        "description": "增量采集并生成结构化舆情事件和研究告警",
        "enabled": True,
    },
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
        "description": "训练候选，经统一回测门禁后由 ReleaseManager 发布",
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
        "name": "旧每周适配器训练（已停用）",
        "kind": "cron",
        "expr": "0 8 * * 6",  # 周六08:00
        "command": "weekly_adapt",
        "description": "遗留 AdapterStack 入口，调度器强制拒绝执行",
        "enabled": False,
    },
    {
        "id": "bimonthly_retrain",
        "name": "旧双月重训练（已停用）",
        "kind": "cron",
        "expr": "0 9 1 */2 *",  # 每两月1号09:00
        "command": "bimonthly_retrain",
        "description": "遗留 AdapterStack 入口，调度器强制拒绝执行",
        "enabled": False,
    },
]


def _parse_cron_field(field, minimum, maximum):
    """Expand one cron field into a validated integer set."""
    values = set()
    for token in field.split(','):
        token = token.strip()
        if not token:
            raise ValueError("empty cron field token")
        base, separator, step_text = token.partition('/')
        step = int(step_text) if separator else 1
        if step <= 0:
            raise ValueError("cron step must be positive")
        if base == '*':
            start, end = minimum, maximum
        elif '-' in base:
            start_text, end_text = base.split('-', 1)
            start, end = int(start_text), int(end_text)
        else:
            value = int(base)
            start, end = value, value
        if start < minimum or end > maximum or start > end:
            raise ValueError("cron field out of range")
        values.update(range(start, end + 1, step))
    return values


def cron_matches(expr, now):
    """Match a standard five-field cron expression against ``now``.

    Day-of-month and day-of-week use the conventional cron OR behavior when
    both fields are restricted. Both 0 and 7 represent Sunday.
    """
    parts = str(expr).split()
    if len(parts) != 5:
        return False
    try:
        minutes = _parse_cron_field(parts[0], 0, 59)
        hours = _parse_cron_field(parts[1], 0, 23)
        month_days = _parse_cron_field(parts[2], 1, 31)
        months = _parse_cron_field(parts[3], 1, 12)
        weekdays = {
            0 if value == 7 else value
            for value in _parse_cron_field(parts[4], 0, 7)
        }
    except (TypeError, ValueError):
        return False

    if now.minute not in minutes or now.hour not in hours or now.month not in months:
        return False

    cron_weekday = (now.weekday() + 1) % 7
    dom_match = now.day in month_days
    dow_match = cron_weekday in weekdays
    dom_wildcard = parts[2] == '*'
    dow_wildcard = parts[4] == '*'
    if not dom_wildcard and not dow_wildcard:
        return dom_match or dow_match
    if not dom_wildcard:
        return dom_match
    if not dow_wildcard:
        return dow_match
    return True


# ============================================================
# 调度器核心
# ============================================================

class CronScheduler:
    """轻量级定时任务调度器 (借鉴 Dexter cron)"""

    def __init__(self, base_dir):
        self.base_dir = os.fspath(base_dir)
        self.store_path = os.path.join(self.base_dir, "outputs", "cron_jobs.json")
        self.daemon_state_path = os.path.join(
            self.base_dir, "outputs", "cron_daemon.json"
        )
        self.jobs = self._load_jobs()

    def _load_jobs(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
            if not isinstance(jobs, list):
                raise ValueError("cron_jobs.json must contain a list")
            changed = False
            defaults = {job['id']: job for job in DEFAULT_JOBS}
            existing_ids = {
                job.get('id') for job in jobs if isinstance(job, dict)
            }
            for job in jobs:
                if not isinstance(job, dict) or not job.get('id'):
                    raise ValueError("invalid cron job entry")
                job_id = job['id']
                if job_id in {'weekly_adapt', 'bimonthly_retrain'} and job.get('enabled'):
                    job['enabled'] = False
                    changed = True
                default = defaults.get(job_id)
                if default and job.get('command') != default['command']:
                    job['command'] = default['command']
                    changed = True
            for default in DEFAULT_JOBS:
                if default['id'] not in existing_ids:
                    jobs.append(copy.deepcopy(default))
                    changed = True
            if changed:
                self._save_jobs(jobs)
            return jobs
        # 首次初始化
        jobs = copy.deepcopy(DEFAULT_JOBS)
        self._save_jobs(jobs)
        return jobs

    def _save_jobs(self, jobs=None):
        if jobs is None:
            jobs = self.jobs
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".cron_jobs.", suffix=".tmp", dir=os.path.dirname(self.store_path)
        )
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, ensure_ascii=False, indent=2)
            os.replace(temporary_path, self.store_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

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
                    j['last_error'] = redact_sensitive_text(error)[:200]
                self._save_jobs()
                return

    def _write_daemon_state(self, status):
        os.makedirs(os.path.dirname(self.daemon_state_path), exist_ok=True)
        payload = {
            "status": str(status),
            "pid": os.getpid(),
            "heartbeat_at": datetime.now().isoformat(),
        }
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".cron_daemon.",
            suffix=".tmp",
            dir=os.path.dirname(self.daemon_state_path),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            os.replace(temporary_path, self.daemon_state_path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    def _run_command(self, job):
        """Execute one task and return whether the child process succeeded."""
        src_dir = os.path.join(self.base_dir, "src")
        cmd = job['command']

        print(f"\n{'='*50}")
        print(f"  执行任务: {job['name']} ({job['id']})")
        print(f"  命令: {cmd}")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        start = time.time()

        try:
            if cmd in {'weekly_adapt', 'bimonthly_retrain'}:
                raise RuntimeError(
                    "旧版 AdapterStack 任务已停用；请使用 monthly_retrain "
                    "进入统一候选评估和 ReleaseManager 发布链路"
                )
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
                     "import sys; from memory import TradingMemory; "
                     "m=TradingMemory(sys.argv[1]); m.backfill_from_predictions(); print('OK')",
                     self.base_dir],
                    cwd=src_dir, capture_output=True, text=True, timeout=120
                )
            elif cmd == 'retrain':
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "train"],
                    cwd=src_dir, capture_output=True, text=True, timeout=900
                )
            elif cmd == 'monitor':
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "monitor"],
                    cwd=src_dir, capture_output=True, text=True, timeout=180
                )
            elif cmd == 'iterate':
                result = subprocess.run(
                    [sys.executable, "main.py", "--mode", "iterate"],
                    cwd=src_dir, capture_output=True, text=True, timeout=300
                )
            else:
                raise ValueError(f"未知命令: {cmd}")

            elapsed = time.time() - start

            if result.returncode == 0:
                last_lines = result.stdout.strip().split('\n')[-5:]
                print(f"  ✅ 成功 (耗时 {elapsed:.0f}s)")
                for line in last_lines:
                    print(f"     {redact_sensitive_text(line)[:100]}")
                self._mark_run(job['id'], success=True)
                return True
            else:
                print(f"  ❌ 失败 (exit={result.returncode}, 耗时 {elapsed:.0f}s)")
                print(
                    f"     stderr: "
                    f"{redact_sensitive_text(result.stderr)[:200]}"
                )
                error = result.stderr or result.stdout or f"exit={result.returncode}"
                self._mark_run(job['id'], success=False, error=error)
                return False

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  ⏱️ 超时 (>{elapsed:.0f}s)")
            self._mark_run(job['id'], success=False, error="Timeout")
            return False
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 异常: {redact_sensitive_text(e)}")
            self._mark_run(
                job['id'], success=False, error=redact_sensitive_text(e)
            )
            return False

    def run_job(self, job_id):
        """手动运行指定任务"""
        for j in self.jobs:
            if j['id'] == job_id:
                return self._run_command(j)
        print(f"任务不存在: {job_id}")
        return False

    def run_all(self):
        """运行所有已启用的任务"""
        enabled = [j for j in self.jobs if j.get('enabled', True)]
        print(f"运行 {len(enabled)} 个任务...")
        results = [self._run_command(j) for j in enabled]
        return all(results)

    def daemon(self, check_interval=60):
        """后台守护进程模式 (每60秒检查是否需要执行)"""
        print(f"\n🔄 F_Agent 调度守护进程已启动 (检查间隔: {check_interval}s)")
        print(f"   按 Ctrl+C 停止")
        print(f"   配置文件: {self.store_path}")
        print()

        try:
            self._write_daemon_state("running")
            while True:
                now = datetime.now()
                self._write_daemon_state("running")
                # 简化版: 每分钟检查一次
                # 生产环境建议用 schedule 库或系统 crontab
                for j in self.jobs:
                    if not j.get('enabled', True):
                        continue
                    last_run = j.get('last_run', '')
                    if j['kind'] == 'cron':
                        hour_match = self._cron_hour_match(j['expr'], now)
                        if hour_match:
                            schedule_key = now.strftime('%Y-%m-%d %H:%M')
                            if not last_run.startswith(schedule_key):
                                print(f"\n[{now.strftime('%H:%M:%S')}] 触发: {j['name']}")
                                self._run_command(j)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n守护进程已停止")
        finally:
            self._write_daemon_state("stopped")

    @staticmethod
    def _cron_hour_match(expr, now):
        """Backward-compatible entry point for a complete five-field matcher."""
        return cron_matches(expr, now)


def daemon_is_running(base_dir, *, stale_after_seconds=180):
    """Return a cross-platform best-effort daemon status from its heartbeat."""
    state_path = Path(base_dir) / "outputs" / "cron_daemon.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        heartbeat = datetime.fromisoformat(str(state["heartbeat_at"]))
        pid = int(state["pid"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    if state.get("status") != "running":
        return False
    if (datetime.now() - heartbeat).total_seconds() > stale_after_seconds:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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
        return 0
    elif args.action == 'run':
        if args.job_id:
            return 0 if scheduler.run_job(args.job_id) else 1
        else:
            print("请指定任务ID, 例如: python cron_scheduler.py run daily_update")
            return 2
    elif args.action == 'run-all':
        return 0 if scheduler.run_all() else 1
    elif args.action == 'daemon':
        scheduler.daemon()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
