"""Validated, secret-free subscriptions shared by CLI, dashboard and research."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


SOURCE_NAMES = {"eastmoney_tl", "pbc_news", "pbc_omo", "mof_bonds"}
SEVERITIES = {"low": 0, "medium": 1, "high": 2, "critical": 3}
TOPICS = {
    "rates": ("国债", "债市", "收益率", "利率", "TL", "降息", "加息"),
    "liquidity": ("央行", "人民银行", "流动性", "资金面", "逆回购", "降准", "MLF", "LPR"),
    "supply": ("发行", "供给", "财政", "地方债", "专项债", "特别国债"),
    "credit": ("信用债", "公司债", "企业债", "城投", "违约", "展期", "兑付", "评级", "债务"),
}


@dataclass(frozen=True)
class WatchTask:
    id: str = "bond-market"
    name: str = "债券市场情报"
    topics: tuple[str, ...] = ("rates", "liquidity", "supply", "credit")
    entities: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    min_severity: str = "medium"
    notify_unverified: bool = False

    def matches(self, text: str) -> bool:
        text = text.casefold()
        contains = lambda words: any(word.casefold() in text for word in words)
        topics = tuple(word for topic in self.topics for word in TOPICS[topic])
        return (
            not contains(self.exclude_keywords)
            and (not self.entities or contains(self.entities))
            and (not self.keywords or contains(self.keywords))
            and (not topics or contains(topics))
        )


@dataclass(frozen=True)
class MonitorConfig:
    sources: tuple[str, ...] = ("eastmoney_tl", "pbc_news", "pbc_omo", "mof_bonds")
    tasks: tuple[WatchTask, ...] = field(default_factory=lambda: (WatchTask(),))
    interval_seconds: int = 600
    alert_max_age_hours: int = 24
    semantic_mode: str = "off"
    semantic_max_calls_per_run: int = 10
    notification_channel: str = "local"
    webhook_env: str = "BOND_MONITOR_FEISHU_WEBHOOK"
    signing_secret_env: str = "BOND_MONITOR_FEISHU_SECRET"

    @classmethod
    def load(cls, base_dir, path=None) -> "MonitorConfig":
        base_dir = Path(base_dir).resolve()
        location = Path(path) if path else base_dir / "config" / "bond_monitor.json"
        if not location.is_absolute():
            location = Path(base_dir) / location
        if not location.exists() and path is None:
            return cls()
        if location.stat().st_size > 64 * 1024:
            raise ValueError("monitor config exceeds 64 KiB")
        return cls.from_dict(json.loads(location.read_text(encoding="utf-8")))

    def save(self, base_dir):
        """Save a validated configuration atomically, without storing secrets."""
        import os
        import tempfile

        validated = type(self).from_dict(asdict(self))
        directory = Path(base_dir).resolve() / "config"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "bond_monitor.json"
        descriptor, temporary = tempfile.mkstemp(prefix=".bond_monitor.", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(asdict(validated), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @classmethod
    def from_dict(cls, data: dict) -> "MonitorConfig":
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("unknown monitor configuration field")
        values = dict(data)
        sources = _strings(values.get("sources", cls().sources), "sources")
        if not sources or len(sources) != len(set(sources)) or set(sources) - SOURCE_NAMES:
            raise ValueError("sources must be unique supported provider names")
        values["sources"] = sources
        tasks = []
        entries = values.get("tasks", [asdict(WatchTask())])
        if not isinstance(entries, (list, tuple)) or not 1 <= len(entries) <= 20:
            raise ValueError("configure 1-20 watch tasks")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - set(WatchTask.__dataclass_fields__):
                raise ValueError("unknown watch task field")
            item = dict(entry)
            for key in ("topics", "entities", "keywords", "exclude_keywords"):
                if key in item:
                    item[key] = _strings(item[key], key)
            task = WatchTask(**item)
            if not isinstance(task.id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", task.id):
                raise ValueError("task id must be a short ASCII identifier")
            if not isinstance(task.name, str) or not 1 <= len(task.name) <= 100:
                raise ValueError("invalid task name")
            if set(task.topics) - set(TOPICS) or task.min_severity not in SEVERITIES:
                raise ValueError("unsupported topic or severity")
            if not isinstance(task.notify_unverified, bool):
                raise ValueError("notify_unverified must be boolean")
            tasks.append(task)
        if not 1 <= len(tasks) <= 20 or len({task.id for task in tasks}) != len(tasks):
            raise ValueError("configure 1-20 tasks with unique IDs")
        values["tasks"] = tuple(tasks)
        result = cls(**values)
        for key, low, high in (("interval_seconds", 30, 86400), ("alert_max_age_hours", 1, 168),
                               ("semantic_max_calls_per_run", 0, 50)):
            value = getattr(result, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {key}; expected {low}..{high}")
        if result.semantic_mode not in {"off", "online"}:
            raise ValueError("semantic_mode must be off or online")
        if result.notification_channel not in {"local", "feishu"}:
            raise ValueError("notification_channel must be local or feishu")
        for key in ("webhook_env", "signing_secret_env"):
            if not isinstance(getattr(result, key), str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", getattr(result, key)):
                raise ValueError(f"{key} must name an environment variable, not contain a secret")
        return result


def _strings(value, name):
    if not isinstance(value, (list, tuple)) or len(value) > 100:
        raise ValueError(f"{name} must be a list of at most 100 strings")
    if any(not isinstance(x, str) or not x.strip() or len(x) > 100 for x in value):
        raise ValueError(f"invalid string in {name}")
    return tuple(x.strip() for x in value)
