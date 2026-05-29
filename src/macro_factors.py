# -*- coding: utf-8 -*-
# macro_factors.py — Macro, funding, sentiment factor computation

import pandas as pd
import numpy as np


class MacroFactorComputer:
    """
    Compute macro/funding/sentiment factors from raw market data.

    Input: daily-frequency DataFrames (passed from MacroDataFetcher or any source)
    Output: daily-frequency DataFrame with factor columns, shifted +1 day for look-ahead safety.

    Factor categories (22 total):
      FUNDING     — money market tightness (4)
      YIELD_CURVE — Treasury curve shape & shift (5)
      LIQUIDITY   — PBoC operations (4)
      CROSS_MKT   — cross-asset relationships (4)
      MACRO       — macro momentum / surprises (5)
    """

    def __init__(self):
        self.factor_names = []

    # ============================================================
    # Main Entry Point
    # ============================================================
    def compute_all(self, raw_data_dict):
        """
        Compute all macro factors.

        Parameters:
            raw_data_dict: dict with keys:
                yield_curve  — DataFrame with date + Yield_3M..Yield_30Y
                shibor       — DataFrame with date + SHIBOR_ON..SHIBOR_1Y
                repo         — DataFrame (optional) with repo rates
                pmi          — DataFrame with date + PMI_Manufacturing/NonManufacturing
                cpi          — DataFrame with date + CPI_YoY/CPI_MoM
                m2           — DataFrame with date + M2_YoY
                social_financing — DataFrame with date + SocialFin_Flow
                cross_market — DataFrame with CN_US_10Y_Spread etc.
                pboc_omo     — DataFrame (optional)
        Returns:
            pd.DataFrame: daily frequency, date-column index, all factor columns.
        """
        print("[MacroFactors] Computing macro factors...")

        factors_list = []

        # --- Funding Factors ---
        df_shibor = raw_data_dict.get('shibor', pd.DataFrame())
        df_repo = raw_data_dict.get('repo', pd.DataFrame())
        if len(df_shibor) > 0:
            factors_list.append(self._compute_funding_factors(df_shibor, df_repo))

        # --- Yield Curve Factors ---
        df_yield = raw_data_dict.get('yield_curve', pd.DataFrame())
        if len(df_yield) > 0:
            factors_list.append(self._compute_yield_curve_factors(df_yield))

        # --- Liquidity Factors ---
        df_omo = raw_data_dict.get('pboc_omo', pd.DataFrame())
        factors_list.append(self._compute_liquidity_factors(df_omo, df_shibor))

        # --- Cross-Market Factors ---
        df_cross = raw_data_dict.get('cross_market', pd.DataFrame())
        if len(df_cross) > 0:
            factors_list.append(self._compute_cross_market_factors(df_cross, df_yield))

        # --- Macro Momentum Factors ---
        df_pmi = raw_data_dict.get('pmi', pd.DataFrame())
        df_cpi = raw_data_dict.get('cpi', pd.DataFrame())
        df_m2 = raw_data_dict.get('m2', pd.DataFrame())
        df_sf = raw_data_dict.get('social_financing', pd.DataFrame())
        factors_list.append(self._compute_macro_momentum_factors(df_pmi, df_cpi, df_m2, df_sf))

        # Merge all factor DataFrames by date
        df_all = None
        for fdf in factors_list:
            if fdf is None or len(fdf) == 0:
                continue
            fdf = fdf.copy()
            if 'date' not in fdf.columns:
                continue
            fdf['date'] = pd.to_datetime(fdf['date'])
            if df_all is None:
                df_all = fdf
            else:
                df_all = pd.merge(df_all, fdf, on='date', how='outer')

        if df_all is None:
            print("[MacroFactors] WARNING: No factors computed!")
            return pd.DataFrame()

        df_all = df_all.sort_values('date').reset_index(drop=True)

        # Finalize: forward-fill, shift +1 day, winsorize
        df_all = self._finalize(df_all)

        return df_all

    # ============================================================
    # Category 1: Funding Tightness (4 factors)
    # ============================================================
    def _compute_funding_factors(self, df_shibor, df_repo):
        """Compute money-market funding tightness factors."""
        df_shibor = df_shibor.copy()
        df_shibor['date'] = pd.to_datetime(df_shibor['date'])
        df_shibor = df_shibor.sort_values('date')

        # Spread factors: measure curve steepness in short-end money market
        if 'SHIBOR_3M' in df_shibor.columns and 'SHIBOR_1M' in df_shibor.columns:
            df_shibor['SHIBOR_3M_1M_Spread'] = df_shibor['SHIBOR_3M'] - df_shibor['SHIBOR_1M']
        if 'SHIBOR_1Y' in df_shibor.columns and 'SHIBOR_3M' in df_shibor.columns:
            df_shibor['SHIBOR_1Y_3M_Spread'] = df_shibor['SHIBOR_1Y'] - df_shibor['SHIBOR_3M']

        # Funding volatility: rolling std of O/N rate
        if 'SHIBOR_ON' in df_shibor.columns:
            on_mean = df_shibor['SHIBOR_ON'].rolling(20, min_periods=5).mean()
            on_std = df_shibor['SHIBOR_ON'].rolling(20, min_periods=5).std()
            df_shibor['SHIBOR_ON_Deviation'] = (df_shibor['SHIBOR_ON'] - on_mean) / (on_std + 0.01)
            df_shibor['Repo_Volatility'] = on_std / (on_mean + 0.01)

        # Repo rate factors if available
        if len(df_repo) > 0 and 'date' in df_repo.columns:
            df_repo = df_repo.copy()
            df_repo['date'] = pd.to_datetime(df_repo['date'])
            # Check for DR007-like column (any column that might be a repo rate)
            numeric_cols = df_repo.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == 'date':
                    continue
                # Compute deviation from 20-day mean as a tightness proxy
                roll_mean = df_repo[col].rolling(20, min_periods=5).mean()
                roll_std = df_repo[col].rolling(20, min_periods=5).std()
                df_repo[f'{col}_Deviation'] = (df_repo[col] - roll_mean) / (roll_std + 0.01)

            if len(df_shibor) > 0:
                df_shibor = pd.merge(df_shibor, df_repo, on='date', how='left')

        # Select output columns
        out_cols = ['date']
        for c in ['SHIBOR_3M_1M_Spread', 'SHIBOR_1Y_3M_Spread',
                   'SHIBOR_ON_Deviation', 'Repo_Volatility']:
            if c in df_shibor.columns:
                out_cols.append(c)

        return df_shibor[out_cols] if len(out_cols) > 1 else pd.DataFrame()

    # ============================================================
    # Category 2: Yield Curve Shape (5 factors)
    # ============================================================
    def _compute_yield_curve_factors(self, df_yield):
        """Compute Treasury yield curve shape factors."""
        df = df_yield.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Slope: 10Y - 2Y (classic), 30Y - 10Y (ultra-long, TL-relevant)
        if 'Yield_10Y' in df.columns and 'Yield_1Y' in df.columns:
            df['YC_Slope_10Y_1Y'] = df['Yield_10Y'] - df['Yield_1Y']
        if 'Yield_30Y' in df.columns and 'Yield_10Y' in df.columns:
            df['YC_Slope_30Y_10Y'] = df['Yield_30Y'] - df['Yield_10Y']

        # Curvature: 2*5Y - 2Y - 10Y (butterfly)
        if all(c in df.columns for c in ['Yield_5Y', 'Yield_1Y', 'Yield_10Y']):
            df['YC_Curvature'] = 2 * df['Yield_5Y'] - df['Yield_1Y'] - df['Yield_10Y']

        # Level shift: deviation of 10Y from 60-day MA
        if 'Yield_10Y' in df.columns:
            y10_ma60 = df['Yield_10Y'].rolling(60, min_periods=10).mean()
            df['YC_Level_Shift'] = df['Yield_10Y'] - y10_ma60

            y10_ma252 = df['Yield_10Y'].rolling(252, min_periods=60).mean()
            y10_std252 = df['Yield_10Y'].rolling(252, min_periods=60).std()
            df['YC_Level_ZScore'] = (df['Yield_10Y'] - y10_ma252) / (y10_std252 + 0.01)

        out_cols = ['date'] + [c for c in [
            'YC_Slope_10Y_1Y', 'YC_Slope_30Y_10Y', 'YC_Curvature',
            'YC_Level_Shift', 'YC_Level_ZScore'
        ] if c in df.columns]

        return df[out_cols] if len(out_cols) > 1 else pd.DataFrame()

    # ============================================================
    # Category 3: Liquidity / Monetary Policy (4 factors)
    # ============================================================
    def _compute_liquidity_factors(self, df_omo, df_shibor):
        """
        Compute liquidity conditions factors.
        If OMO data is unavailable, derive from SHIBOR dynamics.
        """
        if df_omo is None or len(df_omo) == 0:
            # Derive liquidity proxy from SHIBOR spread dynamics
            if len(df_shibor) > 0 and 'date' in df_shibor.columns:
                df = df_shibor[['date']].copy()
                df['date'] = pd.to_datetime(df['date'])

                # Liquidity stress: widen of short-end spread
                if 'SHIBOR_3M_1M_Spread' in df_shibor.columns:
                    s = df_shibor['SHIBOR_3M_1M_Spread']
                    df['Liquidity_Stress'] = (
                        (s > s.rolling(60, min_periods=10).quantile(0.8))
                    ).astype(int)
                else:
                    df['Liquidity_Stress'] = 0

                df['Net_Injection_5D'] = 0
                df['Net_Injection_20D'] = 0
                df['Injection_Momentum'] = 0
                return df

            return pd.DataFrame()

        # If OMO data exists, compute proper factors
        df = df_omo.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            for col in ['reverse_repo_amount', 'mlf_amount']:
                if col not in df.columns:
                    df[col] = 0

            df['net_injection'] = df.get('reverse_repo_amount', 0) + df.get('mlf_amount', 0)
            df['Net_Injection_5D'] = df['net_injection'].rolling(5, min_periods=1).sum()
            df['Net_Injection_20D'] = df['net_injection'].rolling(20, min_periods=1).sum()
            df['Injection_Momentum'] = df['Net_Injection_5D'] - df['Net_Injection_20D']

            if 'reverse_repo_rate' in df.columns:
                rate_change = df['reverse_repo_rate'].diff(60)
                rate_std = df['reverse_repo_rate'].rolling(60).std()
                df['OMO_Rate_Signal'] = rate_change / (rate_std + 0.01)

            if 'Liquidity_Stress' not in df.columns:
                df['Liquidity_Stress'] = 0

            out_cols = ['date', 'Net_Injection_5D', 'Net_Injection_20D',
                       'Injection_Momentum', 'Liquidity_Stress']
            if 'OMO_Rate_Signal' in df.columns:
                out_cols.append('OMO_Rate_Signal')
            return df[out_cols]

        return pd.DataFrame()

    # ============================================================
    # Category 4: Cross-Market (4 factors)
    # ============================================================
    def _compute_cross_market_factors(self, df_cross, df_yield):
        """Compute cross-asset relationship factors."""
        df = df_cross.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif len(df) > 0:
            df['date'] = pd.to_datetime(df.iloc[:, 0])

        df = df.sort_values('date')

        # CN-US 10Y spread
        if 'CN_US_10Y_Spread' in df.columns:
            df['CN_US_10Y_Spread_Z'] = (
                (df['CN_US_10Y_Spread'] - df['CN_US_10Y_Spread'].rolling(60, min_periods=10).mean())
                / (df['CN_US_10Y_Spread'].rolling(60, min_periods=10).std() + 0.01)
            )

        # Stock-bond correlation proxy via yield change autocorrelation
        if len(df_yield) > 0 and 'Yield_10Y' in df_yield.columns:
            df_y = df_yield[['date', 'Yield_10Y']].copy()
            df_y['date'] = pd.to_datetime(df_y['date'])
            df_y['yield_change'] = df_y['Yield_10Y'].diff()
            df_y['yield_change_5d'] = df_y['Yield_10Y'].diff(5)

            df = pd.merge(df, df_y[['date', 'yield_change', 'yield_change_5d']],
                         on='date', how='left')

            # Yield momentum as risk proxy
            df['YC_Momentum_5D'] = df.get('yield_change_5d', 0)

        # Risk-On/Off indicator: yield rising + no liquidity stress
        if 'YC_Momentum_5D' in df.columns:
            df['Risk_On_Off'] = np.sign(df['YC_Momentum_5D'])
        else:
            df['Risk_On_Off'] = 0

        # Credit spread proxy: use yield curve curvature as credit spread approximation
        df['Credit_Spread_Proxy'] = 0  # placeholder

        out_cols = ['date']
        for c in ['CN_US_10Y_Spread', 'CN_US_10Y_Spread_Z',
                   'YC_Momentum_5D', 'Risk_On_Off', 'Credit_Spread_Proxy']:
            if c in df.columns:
                out_cols.append(c)

        return df[out_cols] if len(out_cols) > 1 else pd.DataFrame()

    # ============================================================
    # Category 5: Macro Momentum (5 factors)
    # ============================================================
    def _compute_macro_momentum_factors(self, df_pmi, df_cpi, df_m2, df_sf):
        """
        Compute macro momentum / surprise factors.
        Monthly data is forward-filled before z-score computation.
        """
        # Build a unified monthly date index
        dfs = {'pmi': df_pmi, 'cpi': df_cpi, 'm2': df_m2, 'sf': df_sf}
        result = None

        for name, raw_df in dfs.items():
            if raw_df is None or len(raw_df) == 0:
                continue
            df = raw_df.copy()
            if 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            if result is None:
                result = df
            else:
                result = pd.merge(result, df, on='date', how='outer')

        if result is None:
            return pd.DataFrame()

        result = result.sort_values('date')

        # Create a complete daily date range and forward-fill monthly data
        if len(result) > 0:
            date_range = pd.date_range(
                start=result['date'].min(),
                end=result['date'].max(),
                freq='D'
            )
            result = result.set_index('date').reindex(date_range)
            result = result.ffill().reset_index()
            result = result.rename(columns={'index': 'date'})

        # --- PMI Z-Score ---
        if 'PMI_Manufacturing' in result.columns:
            pmi = result['PMI_Manufacturing'].astype(float)
            pmi_mean = pmi.rolling(24 * 30, min_periods=60).mean()  # ~24 months
            pmi_std = pmi.rolling(24 * 30, min_periods=60).std()
            result['PMI_ZScore'] = ((pmi - pmi_mean) / (pmi_std + 0.01)).clip(-4, 4)

        # --- CPI Momentum ---
        if 'CPI_YoY' in result.columns:
            cpi = result['CPI_YoY'].astype(float)
            # 3-month acceleration (using daily index, ~90 days lag)
            result['CPI_Momentum'] = (cpi - cpi.shift(90)).clip(-5, 5)

        # --- M2 Surprise ---
        if 'M2_YoY' in result.columns:
            m2 = result['M2_YoY'].astype(float)
            m2_mean = m2.rolling(12 * 30, min_periods=60).mean()
            m2_std = m2.rolling(12 * 30, min_periods=60).std()
            result['M2_Surprise'] = ((m2 - m2_mean) / (m2_std + 0.01)).clip(-4, 4)

        # --- Social Financing Z-Score ---
        if 'SocialFin_Flow' in result.columns:
            sf = result['SocialFin_Flow'].astype(float)
            # Log-transform for normalization (social financing is highly seasonal)
            sf_log = np.log(sf.clip(lower=1))
            sf_mean = sf_log.rolling(12 * 30, min_periods=60).mean()
            sf_std = sf_log.rolling(12 * 30, min_periods=60).std()
            result['SocialFin_ZScore'] = ((sf_log - sf_mean) / (sf_std + 0.01)).clip(-4, 4)

        # --- Composite macro surprise ---
        surprise_cols = [c for c in ['PMI_ZScore', 'M2_Surprise', 'SocialFin_ZScore']
                        if c in result.columns]
        if surprise_cols:
            result['Macro_Surprise_Composite'] = result[surprise_cols].mean(axis=1)

        out_cols = ['date'] + [c for c in [
            'PMI_ZScore', 'CPI_Momentum', 'M2_Surprise',
            'SocialFin_ZScore', 'Macro_Surprise_Composite'
        ] if c in result.columns]

        return result[out_cols] if len(out_cols) > 1 else pd.DataFrame()

    # ============================================================
    # Finalize: forward-fill, shift, winsorize
    # ============================================================
    def _finalize(self, df_factors):
        """
        1. Forward-fill all NaN (monthly data becomes daily)
        2. Shift ALL factor columns by 1 day (no look-ahead bias)
        3. Winsorize at 1st/99th percentile
        4. Store factor names
        """
        df = df_factors.sort_values('date').reset_index(drop=True)

        # Identify factor columns (everything except 'date')
        factor_cols = [c for c in df.columns if c != 'date']

        # Forward fill
        df[factor_cols] = df[factor_cols].ffill()

        # Shift by 1 day
        df[factor_cols] = df[factor_cols].shift(1)

        # Winsorize
        for col in factor_cols:
            if df[col].notna().sum() > 10:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                if upper > lower:
                    df[col] = df[col].clip(lower, upper)

        # Fill remaining NaN with 0
        df[factor_cols] = df[factor_cols].fillna(0)

        self.factor_names = factor_cols
        print(f"[MacroFactors] Computed {len(self.factor_names)} factors: {self.factor_names}")
        return df


if __name__ == '__main__':
    # Quick test with synthetic data
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')

    synthetic = {
        'yield_curve': pd.DataFrame({
            'date': dates,
            'Yield_3M': 2.0 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.2,
            'Yield_6M': 2.2 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.15,
            'Yield_1Y': 2.3 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.15,
            'Yield_3Y': 2.5 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.1,
            'Yield_5Y': 2.7 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.1,
            'Yield_7Y': 2.8 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.1,
            'Yield_10Y': 2.85 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.1,
            'Yield_30Y': 3.2 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.1,
        }),
        'shibor': pd.DataFrame({
            'date': dates,
            'SHIBOR_ON': 1.5 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_1W': 1.8 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_2W': 1.9 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_1M': 2.0 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_3M': 2.2 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_6M': 2.3 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_9M': 2.4 + np.random.randn(len(dates)) * 0.1,
            'SHIBOR_1Y': 2.5 + np.random.randn(len(dates)) * 0.1,
        }),
        'pmi': pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                    '2023-04-01', '2023-05-01', '2023-06-01']),
            'PMI_Manufacturing': [50.1, 50.2, 51.9, 49.2, 48.8, 49.0],
            'PMI_NonManufacturing': [54.0, 55.0, 58.2, 56.4, 54.5, 53.2],
        }),
        'cpi': pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                    '2023-04-01', '2023-05-01', '2023-06-01']),
            'CPI_YoY': [2.1, 1.0, 0.7, 0.1, 0.2, 0.0],
            'CPI_MoM': [0.8, -0.5, -0.3, -0.1, -0.2, -0.1],
        }),
        'm2': pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                    '2023-04-01', '2023-05-01', '2023-06-01']),
            'M2_YoY': [12.6, 12.9, 12.7, 12.4, 11.6, 11.3],
        }),
        'social_financing': pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                    '2023-04-01', '2023-05-01', '2023-06-01']),
            'SocialFin_Flow': [59800, 31560, 53800, 12200, 15600, 42200],
        }),
        'cross_market': pd.DataFrame({
            'date': dates,
            'CN_US_10Y_Spread': -0.8 + np.sin(np.linspace(0, np.pi, len(dates))) * 0.3,
        }),
        'pboc_omo': pd.DataFrame(),
    }

    computer = MacroFactorComputer()
    df_factors = computer.compute_all(synthetic)
    print(f"\nFinal factor DataFrame: {len(df_factors)} rows x {len(df_factors.columns)} cols")
    print(f"Columns: {list(df_factors.columns)}")
    print(f"\nTail (last 5 rows):")
    print(df_factors.tail())
