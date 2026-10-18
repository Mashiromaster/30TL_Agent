"""Durable, deduplicated subscriptions and opt-in Feishu delivery."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timedelta, timezone

from bond_monitor_config import SEVERITIES


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def eligible(record, task, config, now):
    if record["prompt_injection_flag"] or not task.matches(record["title"] + "\n" + record["content"]):
        return False
    if SEVERITIES.get(record["severity"], 0) < SEVERITIES[task.min_severity]:
        return False
    published, available = timestamp(record["published_at"]), timestamp(record["available_at"])
    current = timestamp(now)
    # Missing, future or stale publication time is evidence only, not a live alert.
    if not current or not published or not available or available > current:
        return False
    if not 0 <= (current - published).total_seconds() <= config.alert_max_age_hours * 3600:
        return False
    state = record["raw_response"].get("fact_status", "unknown")
    # A media assertion remains visibly unverified, but is not the same as a
    # rumor. Unconfirmed forecasts/rumors require an explicit subscription opt-in.
    if state not in {"asserted", "denied"} and not task.notify_unverified:
        return state == "conditional" and record["raw_response"].get("verification") == "primary_source"
    return True


def feishu_send(payload, *, webhook, signing_secret="", now, session=None):
    if not re.fullmatch(r"https://open\.feishu\.cn/open-apis/bot/v2/hook/[A-Za-z0-9_-]{10,100}", webhook):
        raise ValueError("invalid Feishu webhook; configure the named environment variable")
    body = {"msg_type": "text", "content": {"text": payload["text"]}}
    if signing_secret:
        seconds = str(int(timestamp(now).timestamp()))
        body.update(timestamp=seconds, sign=base64.b64encode(hmac.new(
            f"{seconds}\n{signing_secret}".encode(), b"", hashlib.sha256
        ).digest()).decode())
    import requests
    client = session or requests.Session()
    if hasattr(client, "trust_env"):
        client.trust_env = False
    try:
        response = client.post(webhook, json=body, timeout=(5, 10), allow_redirects=False)
        try:
            if response.status_code != 200:
                raise ValueError(f"Feishu HTTP status {response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or data.get("code") != 0:
                # Never persist the response body/URL, which may contain secrets.
                raise ValueError("Feishu rejected message")
        finally:
            response.close()
    finally:
        if session is None:
            client.close()


class NotificationService:
    def __init__(self, store, config, *, allow_external=False, sender=None):
        self.store, self.config = store, config
        self.allow_external = allow_external
        self.sender = sender

    def prepare(self, records, now):
        seen_threads = set()
        # Newest assessment first; a correction supersedes the previous wording.
        for record in records:
            cluster = record["event_cluster_id"]
            if cluster in seen_threads:
                continue
            seen_threads.add(cluster)
            for task in self.config.tasks:
                if not eligible(record, task, self.config, now):
                    continue
                raw = record["raw_response"]
                signature = [cluster, raw.get("fact_status"), record["severity"],
                             record["policy_stance"], record["bond_impact"],
                             raw.get("verification"), sorted({fact["text"] for fact in raw.get("facts", [])})]
                delivery_id = hashlib.sha256(json.dumps(
                    [task.id, self.config.notification_channel, signature], ensure_ascii=False, sort_keys=True
                ).encode()).hexdigest()[:40]
                verification = "官方原文（不等同于市场影响已验证）" if raw.get("verification") == "primary_source" else "媒体报道，尚未独立核实"
                evidence = "\n".join(record["evidence"][:3])
                payload = {"text": (
                    f"[{task.name}] {record['title']}\n"
                    f"状态: {raw.get('fact_status', 'unknown')} | 等级: {record['severity']}\n"
                    f"来源: {record['source_name']} | {verification}\n"
                    f"发布时间: {record['published_at']}\n"
                    f"原文证据: {evidence}\n{record['url']}\n"
                    f"跟踪编号: {delivery_id}\n仅供研究，不是交易指令。"
                ), "event_cluster_id": cluster, "published_at": record["published_at"]}
                self.store.enqueue_notification(delivery_id=delivery_id, task_id=task.id,
                                                channel=self.config.notification_channel,
                                                assessment_id=record["assessment_id"], payload=payload, now=now)

    def dispatch(self, now, *, limit=20):
        channel = self.config.notification_channel
        if channel != "local" and not self.allow_external:
            return {"sent": 0, "failed": 0, "external_disabled": True}
        current = timestamp(now)
        counts = {"sent": 0, "failed": 0, "local": 0, "cancelled": 0}
        tasks = {task.id: task for task in self.config.tasks}
        recent = self.store.recent_event_records(limit=1000)
        newest = {}
        for record in recent:
            newest.setdefault(record["event_cluster_id"], record)
        for _ in range(limit):
            item = self.store.claim_delivery(channel=channel, now=now,
                lease_until=(current + timedelta(minutes=5)).isoformat(), active_tasks=list(tasks))
            if item is None:
                break
            record = newest.get(item["payload"].get("event_cluster_id"))
            if (record is None or record["assessment_id"] != item["assessment_id"]
                    or not eligible(record, tasks[item["task_id"]], self.config, now)):
                self.store.finish_delivery(item["delivery_id"], status="cancelled", now=now,
                                           error="expired, superseded or no longer subscribed")
                counts["cancelled"] += 1
                continue
            try:
                if channel == "local":
                    status = "local"
                else:
                    if self.sender:
                        self.sender(item["payload"])
                    else:
                        feishu_send(item["payload"], webhook=os.environ.get(self.config.webhook_env, ""),
                                    signing_secret=os.environ.get(self.config.signing_secret_env, ""), now=now)
                    status = "sent"
                self.store.finish_delivery(item["delivery_id"], status=status, now=now)
                counts[status] += 1
            except Exception as exc:
                # Unknown send outcomes are retried with the same visible ID.
                # Feishu has no idempotency contract: delivery is at-least-once.
                self.store.finish_delivery(item["delivery_id"], status="failed", now=now,
                    next_attempt_at=(current + timedelta(seconds=min(3600, 30 * 2 ** item["attempts"]))).isoformat(),
                    error=type(exc).__name__ + ": notification delivery failed; verify channel credentials/connectivity")
                counts["failed"] += 1
        counts["exhausted"] = sum(item["channel"] == channel and item["status"] == "failed"
                                  and item["attempts"] >= 5 and item["task_id"] in tasks
                                  for item in self.store.list_deliveries(limit=1000))
        return counts
