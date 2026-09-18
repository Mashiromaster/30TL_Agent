# -*- coding: utf-8 -*-
# data_fetcher.py — AKShare-based real-time market data fetching module

import pandas as pd
import numpy as np
import os
import time
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CachePolicy:
    """Policy for raw macro cache validation and refresh behavior."""

    force_refresh: bool = False
    allow_stale_on_error: bool = True
    coverage_grace_days: int = 7
    schema_version: int = 2


class MacroDataFetcher:
    """
    Fetch macro, funding, yield curve, and cross-market data from AKShare.
    All methods support local .pkl caching for offline/repeated use.
    """

    def __init__(
        self,
        cache_dir,
        start_date="20230421",
        end_date="20251031",
        cache_policy=None,
    ):
        self.cache_dir = os.fspath(cache_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.cache_policy = cache_policy or CachePolicy()
        self._last_request_time = 0
        os.makedirs(self.cache_dir, exist_ok=True)

    def _rate_limit(self):
        """Ensure at least 0.5s between AKShare requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()

    def _manifest_path(self, cache_name):
        return os.path.join(self.cache_dir, f"{cache_name}.meta.json")

    @staticmethod
    def _read_pickle_safely(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
        except Exception as exc:
            print(f"  [DataFetcher] Invalid cache {os.path.basename(cache_path)}: {exc}")
            return None
        return cached if isinstance(cached, pd.DataFrame) else None

    def _load_manifest(self, cache_name):
        try:
            with open(self._manifest_path(cache_name), "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            return None

    def _cache_covers_request(self, cache_name, cached):
        manifest = self._load_manifest(cache_name)
        if not manifest or manifest.get("schema_version") != self.cache_policy.schema_version:
            return False
        if "date" not in cached.columns or cached.empty:
            return False
        dates = pd.to_datetime(cached["date"], errors="coerce").dropna()
        if dates.empty:
            return False
        requested_start = pd.to_datetime(self.start_date)
        requested_end = pd.to_datetime(self.end_date)
        grace = pd.Timedelta(days=self.cache_policy.coverage_grace_days)
        return dates.min().normalize() <= requested_start and dates.max().normalize() >= requested_end - grace

    def _write_cache_atomically(self, cache_name, frame):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        manifest_path = self._manifest_path(cache_name)
        token = f"{os.getpid()}-{time.time_ns()}"
        temp_cache = f"{cache_path}.{token}.tmp"
        temp_manifest = f"{manifest_path}.{token}.tmp"
        dates = (
            pd.to_datetime(frame["date"], errors="coerce").dropna()
            if "date" in frame.columns else pd.Series(dtype="datetime64[ns]")
        )
        manifest = {
            **asdict(self.cache_policy),
            "source": cache_name,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "requested_start": str(self.start_date),
            "requested_end": str(self.end_date),
            "coverage_start": dates.min().isoformat() if not dates.empty else None,
            "coverage_end": dates.max().isoformat() if not dates.empty else None,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }
        try:
            frame.to_pickle(temp_cache)
            with open(temp_manifest, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            os.replace(temp_cache, cache_path)
            os.replace(temp_manifest, manifest_path)
        finally:
            for path in (temp_cache, temp_manifest):
                if os.path.exists(path):
                    os.remove(path)

    def _fetch_with_cache(self, cache_name, fetch_func, force_refresh=None):
        """Return a validated cache or refresh it atomically.

        A forced refresh reaches this raw-cache layer. If the network refresh
        fails, a readable previous cache can still be used when explicitly
        allowed by ``CachePolicy``.
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        refresh = self.cache_policy.force_refresh if force_refresh is None else force_refresh
        stale = self._read_pickle_safely(cache_path) if os.path.exists(cache_path) else None
        if not refresh and stale is not None and self._cache_covers_request(cache_name, stale):
            print(f"  [DataFetcher] Loaded validated cache: {cache_name}")
            return stale

        try:
            self._rate_limit()
            df = fetch_func()
            if df is not None and len(df) > 0:
                self._write_cache_atomically(cache_name, df)
                print(f"  [DataFetcher] Fetched & cached: {cache_name} ({len(df)} rows)")
                return df
            else:
                print(f"  [DataFetcher] Empty result: {cache_name}")
                if (
                    self.cache_policy.allow_stale_on_error
                    and stale is not None
                    and self._cache_covers_request(cache_name, stale)
                ):
                    print(f"  [DataFetcher] Using stale cache after empty refresh: {cache_name}")
                    return stale
                return pd.DataFrame() if df is None else df
        except Exception as e:
            print(f"  [DataFetcher] Failed to fetch {cache_name}: {e}")
            if (
                self.cache_policy.allow_stale_on_error
                and stale is not None
                and self._cache_covers_request(cache_name, stale)
            ):
                print(f"  [DataFetcher] Using stale cache after refresh failure: {cache_name}")
                return stale
            return pd.DataFrame()

    # ============================================================
    # 1. Treasury Yield Curve
    # ============================================================
    def fetch_treasury_yield_curve(self):
        """Fetch China treasury yield curve (3M, 6M, 1Y, 3Y, 5Y, 7Y, 10Y, 30Y)."""
        def _fetch():
            import akshare as ak
            # API may reject large date ranges, batch by 1-year chunks
            s = pd.to_datetime(self.start_date)
            e = pd.to_datetime(self.end_date)
            chunks = []
            cur = s
            while cur < e:
                nxt = min(cur + timedelta(days=365), e)
                s_str = cur.strftime('%Y%m%d')
                e_str = nxt.strftime('%Y%m%d')
                try:
                    batch = ak.bond_china_yield(start_date=s_str, end_date=e_str)
                    chunks.append(batch)
                except Exception:
                    pass
                cur = nxt + timedelta(days=1)
                self._rate_limit()

            if not chunks:
                return pd.DataFrame()

            df = pd.concat(chunks, ignore_index=True)
            # Filter for treasury bond yield curve only
            treasury = df[df['曲线名称'] == '中债国债收益率曲线'].copy()
            if len(treasury) == 0:
                return pd.DataFrame()

            treasury['date'] = pd.to_datetime(treasury['日期'])
            treasury = treasury.rename(columns={
                '3月': 'Yield_3M', '6月': 'Yield_6M', '1年': 'Yield_1Y',
                '3年': 'Yield_3Y', '5年': 'Yield_5Y', '7年': 'Yield_7Y',
                '10年': 'Yield_10Y', '30年': 'Yield_30Y'
            })
            cols = ['date', 'Yield_3M', 'Yield_6M', 'Yield_1Y', 'Yield_3Y',
                    'Yield_5Y', 'Yield_7Y', 'Yield_10Y', 'Yield_30Y']
            return treasury[cols].sort_values('date').reset_index(drop=True)

        return self._fetch_with_cache("yield_curve", _fetch)

    # ============================================================
    # 2. SHIBOR Rates
    # ============================================================
    def fetch_shibor_rates(self):
        """Fetch SHIBOR rates (O/N, 1W, 2W, 1M, 3M, 6M, 9M, 1Y)."""
        def _fetch():
            import akshare as ak
            df = ak.macro_china_shibor_all()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            # Column names: ['日期', 'O/N-定盘', 'O/N-涨跌', '1W-定盘', ...]
            df = df.copy()
            df['date'] = pd.to_datetime(df['日期'])

            # Extract only the fixing rate columns (定盘), not changes (涨跌)
            result = pd.DataFrame()
            result['date'] = df['date']

            tenor_map = {
                'O/N-定盘': 'SHIBOR_ON',
                '1W-定盘': 'SHIBOR_1W',
                '2W-定盘': 'SHIBOR_2W',
                '1M-定盘': 'SHIBOR_1M',
                '3M-定盘': 'SHIBOR_3M',
                '6M-定盘': 'SHIBOR_6M',
                '9M-定盘': 'SHIBOR_9M',
                '1Y-定盘': 'SHIBOR_1Y',
            }
            for src, dst in tenor_map.items():
                if src in df.columns:
                    result[dst] = pd.to_numeric(df[src], errors='coerce')
                else:
                    result[dst] = np.nan

            result = result.dropna(subset=[c for c in tenor_map.values() if c in result.columns],
                                   how='all')
            return result.sort_values('date').reset_index(drop=True)

        return self._fetch_with_cache("shibor_rates", _fetch)

    # ============================================================
    # 3. Repo Rates (DR007, R007, etc.)
    # ============================================================
    def fetch_repo_rates(self):
        """Fetch fixed repo rates from the official AKShare history API."""
        def _fetch():
            import akshare as ak
            frames = []

            # Official repo_rate_hist supports at most one year per request.
            try:
                start = pd.to_datetime(self.start_date)
                end = pd.to_datetime(self.end_date)
                repo_chunks = []
                cursor = start
                while cursor <= end:
                    chunk_end = min(cursor + pd.Timedelta(days=364), end)
                    self._rate_limit()
                    chunk = ak.repo_rate_hist(
                        start_date=cursor.strftime('%Y%m%d'),
                        end_date=chunk_end.strftime('%Y%m%d'),
                    )
                    if chunk is not None and not chunk.empty:
                        repo_chunks.append(chunk)
                    cursor = chunk_end + pd.Timedelta(days=1)
                if repo_chunks:
                    repo = pd.concat(repo_chunks, ignore_index=True)
                    date_col = 'date' if 'date' in repo.columns else repo.columns[0]
                    rename = {
                        col: f"Repo_{str(col).upper()}"
                        for col in repo.columns
                        if str(col).upper() in {
                            'FR001', 'FR007', 'FR014', 'FDR001', 'FDR007', 'FDR014'
                        }
                    }
                    if rename:
                        repo = repo.rename(columns={date_col: 'date', **rename})
                        repo['date'] = pd.to_datetime(repo['date'], errors='coerce')
                        keep = ['date', *rename.values()]
                        frames.append(repo[keep].drop_duplicates('date', keep='last'))
            except Exception as e:
                print(f"  [DataFetcher] Repo rates failed: {e}")

            if not frames:
                return pd.DataFrame()
            result = frames[0]
            for frame in frames[1:]:
                result = result.merge(frame, on='date', how='outer')
            return result.sort_values('date').reset_index(drop=True)

        return self._fetch_with_cache("repo_rates", _fetch)

    # ============================================================
    # 4. PBoC Open Market Operations
    # ============================================================
    def fetch_pboc_omo(self):
        """Fetch PBoC OMO data. Returns empty DataFrame if scraping fails."""
        def _fetch():
            # PBoC OMO has no stable AKShare API; return empty placeholder
            # The factor computer will handle missing data gracefully
            print("  [DataFetcher] PBoC OMO: No stable API, returning empty")
            return pd.DataFrame()

        return self._fetch_with_cache("pboc_omo", _fetch)

    # ============================================================
    # 5. PMI
    # ============================================================
    def fetch_pmi(self):
        """Fetch China manufacturing & non-manufacturing PMI (monthly)."""
        def _fetch():
            import akshare as ak
            df = ak.macro_china_pmi()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.copy()
            # Columns: ['月份', '制造业-指数', '制造业-同比增长', '非制造业-指数', '非制造业-同比增长']
            df['date'] = pd.to_datetime(df['月份'].str.replace('年', '-').str.replace('月份', '-01'))
            df['PMI_Manufacturing'] = pd.to_numeric(df['制造业-指数'], errors='coerce')
            df['PMI_NonManufacturing'] = pd.to_numeric(df['非制造业-指数'], errors='coerce')
            return df[['date', 'PMI_Manufacturing', 'PMI_NonManufacturing']].sort_values('date')

        return self._fetch_with_cache("pmi", _fetch)

    # ============================================================
    # 6. CPI
    # ============================================================
    def fetch_cpi(self):
        """Fetch China CPI (monthly YoY & MoM)."""
        def _fetch():
            import akshare as ak
            df = ak.macro_china_cpi()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.copy()
            # Columns: ['月份', '全国-当月', '全国-同比增长', '全国-环比增长', ...]
            date_col = df.columns[0]
            df['date'] = pd.to_datetime(
                df[date_col].astype(str).str.replace('年', '-').str.replace('月份', '-01')
            )
            # Direct column name matching
            yoy_col = [c for c in df.columns if '同比' in str(c)]
            mom_col = [c for c in df.columns if '环比' in str(c)]
            df['CPI_YoY'] = pd.to_numeric(df[yoy_col[0]], errors='coerce') if yoy_col else np.nan
            df['CPI_MoM'] = pd.to_numeric(df[mom_col[0]], errors='coerce') if mom_col else np.nan
            return df[['date', 'CPI_YoY', 'CPI_MoM']].sort_values('date')

        return self._fetch_with_cache("cpi", _fetch)

    # ============================================================
    # 7. M2 Money Supply
    # ============================================================
    def fetch_m2(self):
        """Fetch China M2 money supply (monthly YoY)."""
        def _fetch():
            import akshare as ak
            df = ak.macro_china_money_supply()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.copy()
            date_col = df.columns[0]
            df['date'] = pd.to_datetime(
                df[date_col].astype(str).str.replace('年', '-').str.replace('月份', '-01')
            )
            # Column pattern: '货币和准货币(M2)-同比增长'
            val_col = [c for c in df.columns if 'M2' in str(c) and '同比' in str(c)]
            if not val_col:
                val_col = [c for c in df.columns[1:] if df[c].dtype in ['float64', 'int64']]
            df['M2_YoY'] = pd.to_numeric(df[val_col[0]], errors='coerce') if val_col else np.nan
            return df[['date', 'M2_YoY']].sort_values('date')

        return self._fetch_with_cache("m2", _fetch)

    # ============================================================
    # 8. Social Financing
    # ============================================================
    def fetch_social_financing(self):
        """Fetch China social financing scale (monthly)."""
        def _fetch():
            import akshare as ak
            df = ak.macro_china_shrzgm()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.copy()
            date_col = df.columns[0]
            df['date'] = pd.to_datetime(
                df[date_col].astype(str).str.replace('年', '-').str.replace('月份', '-01')
            )
            # Find the flow amount column
            val_cols = [c for c in df.columns if '量' in c or '规模' in c or '融资' in c]
            if not val_cols:
                val_cols = [c for c in df.columns[1:] if df[c].dtype in ['float64', 'int64']]
            df['SocialFin_Flow'] = pd.to_numeric(df[val_cols[0]], errors='coerce') if val_cols else np.nan
            return df[['date', 'SocialFin_Flow']].sort_values('date')

        return self._fetch_with_cache("social_financing", _fetch)

    # ============================================================
    # 9. Treasury Futures Daily Data
    # ============================================================
    def fetch_treasury_futures_daily(self):
        """Fetch TL, T, TF, TS main-contract daily data from Sina."""
        def _fetch():
            import akshare as ak
            symbols = {
                'TL': 'TL0',   # 30Y Treasury Bond Futures
                'T': 'T0',     # 10Y
                'TF': 'TF0',   # 5Y
                'TS': 'TS0',   # 2Y
            }
            results = []
            for name, code in symbols.items():
                try:
                    self._rate_limit()
                    df = ak.futures_main_sina(
                        symbol=code,
                        start_date=self.start_date,
                        end_date=self.end_date
                    )
                    if df is not None and len(df) > 0:
                        df = df.copy()
                        df['date'] = pd.to_datetime(df['日期'])
                        df['symbol'] = name
                        df = df.rename(columns={
                            '开盘价': 'open', '最高价': 'high', '最低价': 'low',
                            '收盘价': 'close', '成交量': 'volume', '持仓量': 'open_interest'
                        })
                        results.append(df[['date', 'symbol', 'open', 'high', 'low',
                                          'close', 'volume', 'open_interest']])
                    print(f"  [DataFetcher] {name} futures: {len(df) if df is not None else 0} rows")
                except Exception as e:
                    print(f"  [DataFetcher] {name} futures failed: {e}")

            if not results:
                return pd.DataFrame()
            return pd.concat(results, ignore_index=True).sort_values(['symbol', 'date'])

        return self._fetch_with_cache("treasury_futures_daily", _fetch)

    # ============================================================
    # 10. Cross-Market Data (CSI 300, China-US Spread)
    # ============================================================
    def fetch_cross_market(self):
        """Fetch CSI 300 index and China-US yield spread."""
        def _fetch():
            import akshare as ak

            result = pd.DataFrame()

            # China-US yield spread
            try:
                self._rate_limit()
                df_spread = ak.bond_zh_us_rate(start_date=self.start_date)
                if df_spread is not None and len(df_spread) > 0:
                    df_spread = df_spread.copy()
                    result['date'] = pd.to_datetime(df_spread['日期'])
                    # Column names vary; try common patterns
                    for col in df_spread.columns:
                        if '中国' in col and '10年' in col:
                            result['CN_10Y'] = pd.to_numeric(df_spread[col], errors='coerce')
                        if '美国' in col and '10年' in col:
                            result['US_10Y'] = pd.to_numeric(df_spread[col], errors='coerce')
                    if 'CN_10Y' in result.columns and 'US_10Y' in result.columns:
                        result['CN_US_10Y_Spread'] = result['CN_10Y'] - result['US_10Y']
            except Exception as e:
                print(f"  [DataFetcher] CN-US spread failed: {e}")

            # LPR (Loan Prime Rate) as policy rate reference
            try:
                self._rate_limit()
                df_lpr = ak.macro_china_lpr()
                if df_lpr is not None and len(df_lpr) > 0:
                    df_lpr = df_lpr.copy()
                    # AKShare has exposed both English and Chinese schemas.
                    lpr_date_col = next(
                        (col for col in ('TRADE_DATE', '日期', 'date') if col in df_lpr),
                        df_lpr.columns[0],
                    )
                    lpr_date = pd.to_datetime(df_lpr[lpr_date_col])
                    result_lpr = pd.DataFrame({'date': lpr_date})
                    for col in df_lpr.columns:
                        normalized = str(col).upper().replace('_', '')
                        if '1年' in str(col) or normalized == 'LPR1Y':
                            result_lpr['LPR_1Y'] = pd.to_numeric(df_lpr[col], errors='coerce')
                        elif '5年' in str(col) or normalized == 'LPR5Y':
                            result_lpr['LPR_5Y'] = pd.to_numeric(df_lpr[col], errors='coerce')
                    if len(result_lpr.columns) > 1:
                        result = (
                            result_lpr if result.empty
                            else result.merge(result_lpr, on='date', how='outer')
                        )
            except Exception as e:
                print(f"  [DataFetcher] LPR failed: {e}")

            # Swap rate (IRS)
            try:
                self._rate_limit()
                df_swap = ak.macro_china_swap_rate(
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                if df_swap is not None and len(df_swap) > 0 and 'date' not in result.columns:
                    result['date'] = pd.to_datetime(df_swap.iloc[:, 0])
            except Exception as e:
                print(f"  [DataFetcher] IRS metadata failed: {e}")

            if len(result) == 0:
                return pd.DataFrame()
            return result.sort_values('date').reset_index(drop=True)

        return self._fetch_with_cache("cross_market", _fetch)

    # ============================================================
    # Fetch All
    # ============================================================
    def fetch_all(self):
        """
        Fetch all available data sources.
        Returns dict mapping source name -> DataFrame.
        Each DataFrame has a 'date' column (datetime64).
        """
        print(f"[DataFetcher] Fetching all data {self.start_date} ~ {self.end_date}...")
        print(f"[DataFetcher] Cache directory: {self.cache_dir}")

        data = {}

        sources = [
            ('yield_curve', self.fetch_treasury_yield_curve),
            ('shibor', self.fetch_shibor_rates),
            ('repo', self.fetch_repo_rates),
            ('pboc_omo', self.fetch_pboc_omo),
            ('pmi', self.fetch_pmi),
            ('cpi', self.fetch_cpi),
            ('m2', self.fetch_m2),
            ('social_financing', self.fetch_social_financing),
            ('treasury_futures', self.fetch_treasury_futures_daily),
            ('cross_market', self.fetch_cross_market),
        ]

        success_count = 0
        for name, func in sources:
            print(f"\n[DataFetcher] === {name} ===")
            df = func()
            if df is not None and len(df) > 0:
                data[name] = df
                success_count += 1
            else:
                data[name] = pd.DataFrame()
                print(f"  [DataFetcher] {name}: no data")

        print(f"\n[DataFetcher] Complete: {success_count}/{len(sources)} sources fetched")
        return data


if __name__ == '__main__':
    # Quick test
    fetcher = MacroDataFetcher(
        cache_dir="./test_macro_cache",
        start_date="20230421",
        end_date="20230630"
    )
    results = fetcher.fetch_all()
    for k, v in results.items():
        if len(v) > 0:
            print(f"\n{k}: {len(v)} rows, columns: {list(v.columns)[:8]}...")
            print(v.head(2))
