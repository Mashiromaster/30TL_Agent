# -*- coding: utf-8 -*-
"""Update concrete futures contracts without relabeling continuous quotes.

The legacy training pickle contains contract-level minutes. Continuous daily bars
belong to the separate quote cache and must never be appended to that pickle.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

import pandas as pd

from market_data import (
    VARIETY_CONFIG,
    contract_candidates,
    fetch_bars,
    load_quote_series,
)


DEFAULT_BASE_DIR = str(Path(__file__).resolve().parents[1])


def _contract_symbols(variety, symbols=None, as_of=None):
    """Validate the whole requested batch before contacting any provider."""
    if variety not in VARIETY_CONFIG:
        raise ValueError(f"不支持的合约品种: {variety}")
    if symbols is None:
        symbols = contract_candidates(variety, as_of=as_of)
    if isinstance(symbols, str):
        symbols = [symbols]
    normalized = []
    for value in symbols:
        normalized.extend(part.strip().upper() for part in value.split(",") if part.strip())
    pattern = rf"{re.escape(VARIETY_CONFIG[variety]['prefix'])}\d{{2}}(?:03|06|09|12)"
    if not normalized or any(re.fullmatch(pattern, symbol) is None for symbol in normalized):
        raise ValueError(f"仅允许 {variety} 明确季度合约（如 {variety}2612），不允许主力连续或其它品种")
    return list(dict.fromkeys(normalized))


def fetch_contracts(variety="TL", symbols=None, *, fetcher=None, as_of=None):
    """Fetch concrete contracts independently; report, never mask failures.

    Rolling candidates are requests, not a claim that all are listed or that any
    one is the current main contract. OI is never used as contract identity.
    """
    requested = _contract_symbols(variety, symbols, as_of=as_of)
    fetch = fetcher or fetch_bars
    results = []
    failures = []
    for symbol in requested:
        try:
            frame = fetch(symbol, frequency="1min")
            if frame is None or frame.empty:
                raise ValueError("行情源未返回数据")
            required = {"date", "ticker", "source_symbol", "series_kind", "open", "high", "low", "close", "volume", "oi"}
            if not required.issubset(frame.columns):
                raise ValueError(f"行情字段不完整: {sorted(required - set(frame.columns))}")
            if (
                not frame["ticker"].eq(symbol).all()
                or not frame["source_symbol"].eq(symbol).all()
                or not frame["series_kind"].eq("contract").all()
            ):
                raise ValueError("合约身份与请求不一致，拒绝写入训练数据")
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            if frame["date"].isna().any():
                raise ValueError("行情日期为空")
            frame["trade_dt"] = frame["date"].dt.normalize()
            frame["time"] = frame["date"].dt.strftime("%H:%M")
            # Keep the downstream schema: this is estimated, not reported turnover.
            frame["money"] = frame["close"] * frame["volume"] * 100
            frame["money_estimated"] = True
            frame["money_method"] = "legacy_close_x_volume_x_100"
            results.append(frame)
            print(f"  {symbol}: {len(frame):,} 行, {frame['date'].min()} ~ {frame['date'].max()}")
        except Exception as exc:
            failures.append(symbol)
            print(f"  [ERROR] {symbol}: 获取失败 — {exc}")
    if failures:
        print(f"[WARNING] {len(failures)}/{len(requested)} 个合约更新失败: {', '.join(failures)}；成功合约仍可独立更新")
    return results


def _warn_legacy_provenance(frame):
    """Old labels are not evidence; retain old rows until separately audited."""
    required = {"source_symbol", "series_kind"}
    suspicious = not required.issubset(frame.columns)
    if not suspicious:
        suspicious = bool(
            frame["source_symbol"].isna().any()
            or frame["series_kind"].ne("contract").any()
            or frame["source_symbol"].ne(frame["ticker"]).any()
        )
    if suspicious:
        print("[WARNING] 历史缓存缺少可信合约来源或含可疑标签；保留原记录，不自动重标/删除。请按 docs/market_data_quality.md 审计历史数据。")


def _save_with_backup(frame, target):
    """Back up existing bytes and replace only a fully serialized pickle."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        suffix = datetime.now().strftime("%Y%m%dT%H%M%S%f") + "-" + uuid4().hex[:8]
        backup = target.with_name(target.name + ".bak." + suffix)
        shutil.copy2(target, backup)
        print(f"原缓存备份: {backup}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_pickle(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def update_raw_data(base_dir=DEFAULT_BASE_DIR, variety="TL", symbols=None, *, fetcher=None):
    """Append verified contract minutes, without pretending to backfill history."""
    requested = _contract_symbols(variety, symbols)
    cfg = VARIETY_CONFIG[variety]
    target = Path(base_dir) / "data" / cfg["file_name"]
    print(f"{cfg['label']} 具体合约分钟行情更新（不含主力连续）")
    try:
        if target.exists():
            existing = pd.read_pickle(target)
            if not isinstance(existing, pd.DataFrame) or not {"date", "ticker"}.issubset(existing.columns):
                raise ValueError("历史缓存缺少 date/ticker，拒绝覆盖")
            existing = existing.copy()
            existing["date"] = pd.to_datetime(existing["date"], errors="raise")
            if existing["date"].isna().any() or existing["ticker"].isna().any():
                raise ValueError("历史缓存存在空日期/合约，拒绝覆盖")
            _warn_legacy_provenance(existing)
            print(f"现有数据: {len(existing):,} 行, {existing['date'].min()} ~ {existing['date'].max()}")
        else:
            existing = pd.DataFrame()
            print(f"[WARNING] 未找到历史缓存，将创建: {target}")

        frames = fetch_contracts(variety=variety, symbols=requested, fetcher=fetcher)
        if not frames:
            print("[ERROR] 未获取到任何有效合约数据；原缓存保持不变")
            return False
        fresh = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "ticker"], keep="last")
        if not existing.empty:
            fresh_keys = pd.MultiIndex.from_frame(fresh[["date", "ticker"]])
            old_keys = pd.MultiIndex.from_frame(existing[["date", "ticker"]])
            # Replace only the verified contract/minute keys in this response.
            # Do not incidentally clean or reinterpret unrelated legacy rows.
            existing = existing.loc[~old_keys.isin(fresh_keys)]
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        _save_with_backup(combined, target)
    except Exception as exc:
        print(f"[ERROR] 行情更新未完成，原缓存未替换: {exc}")
        return False

    print(f"已保存: {target}; 共 {len(combined):,} 行; 最新观测: {combined['date'].max()}")
    print("[WARNING] 分钟接口仅提供有限近期窗口；最新时间不代表历史连续完整，不能据此认定中间缺口已补齐。")
    return True


def refresh_main_daily(base_dir=DEFAULT_BASE_DIR, variety="TL"):
    """Refresh the isolated continuous daily cache, never training raw data."""
    symbol = VARIETY_CONFIG[variety]["main_symbol"]
    try:
        snapshot = load_quote_series(base_dir, symbol, "1D", refresh=True)
        if snapshot.error:
            print(f"[ERROR] {symbol} 日线刷新失败: {snapshot.error}；缓存不等于刷新成功")
            return False
        if snapshot.bars.empty:
            print(f"[ERROR] {symbol} 日线为空")
            return False
        print(f"{symbol} 主力连续日线: {len(snapshot.bars):,} 行, {snapshot.bars['date'].min()} ~ {snapshot.bars['date'].max()}；保存于独立行情缓存，未修改训练原始数据")
        return True
    except Exception as exc:
        print(f"[ERROR] {symbol} 日线刷新失败: {exc}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Update explicit treasury-futures contracts or isolated continuous daily quotes")
    parser.add_argument("--variety", default="TL", choices=sorted(VARIETY_CONFIG))
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Project root directory")
    parser.add_argument("--symbols", nargs="+", help="Explicit quarterly contracts, separated by spaces or commas; never TL0/T0/TF0/TS0")
    parser.add_argument("--daily-only", action="store_true", help="Refresh continuous daily quote cache only; do not touch training minutes")
    args = parser.parse_args(argv)
    if args.daily_only and args.symbols:
        parser.error("--daily-only 不可与 --symbols 同时使用")
    if args.daily_only:
        return 0 if refresh_main_daily(args.base_dir, args.variety) else 1
    try:
        symbols = _contract_symbols(args.variety, args.symbols) if args.symbols else None
        succeeded = update_raw_data(args.base_dir, args.variety, symbols=symbols)
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
