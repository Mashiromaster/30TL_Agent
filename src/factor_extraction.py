# -*- coding: utf-8 -*-
# factor_extraction.py (V9.7 - 修复基差数据处理)

import pandas as pd
import numpy as np
import os
import mc_ex
from tick_data_processor import calculate_microstructure_factors, run_tick_processing
from data_fetcher import MacroDataFetcher
from macro_factors import MacroFactorComputer

def load_data(data_path, output_filename):
    if os.path.exists(output_filename):
        print(f"[Factor] 读取中间文件: {os.path.basename(output_filename)}")
        return pd.read_pickle(output_filename)
    try:
        df_main = mc_ex.get_main_contract_data(data_path)
        df_main.to_pickle(output_filename)
        return df_main
    except Exception as e:
        print(f"[Factor ERROR] 数据读取失败: {e}")
        raise


def detect_bar_format(df_main):
    """自动检测分钟数据的时间戳格式"""
    df_main['date'] = pd.to_datetime(df_main['date'])
    first_date = df_main['date'].dt.date.iloc[0]
    df_first_day = df_main[df_main['date'].dt.date == first_date]
    first_bar_time = df_first_day['date'].iloc[0]
    
    if first_bar_time.hour == 9 and first_bar_time.minute == 30:
        print("[Factor] 检测到时间戳格式: Start-of-Bar")
        return 'start'
    else:
        print("[Factor] 检测到时间戳格式: End-of-Bar")
        return 'end'


# ============================================================
# 日频基差数据处理
# ============================================================

def load_and_process_basis_data(basis_path):
    """
    加载并处理日频基差数据
    """
    if not os.path.exists(basis_path):
        print(f"[Factor WARNING] 未找到基差数据: {basis_path}")
        return None
    
    print(f"[Factor] 加载基差数据: {basis_path}")
    df_basis = pd.read_pickle(basis_path)
    df_basis['date'] = pd.to_datetime(df_basis['date'])
    
    print(f"[Factor] 原始基差数据: {len(df_basis)} 行")
    
    # 清洗数据：去除 NaN
    df_basis = df_basis.dropna(subset=['net_basis']).copy()
    df_basis = df_basis.reset_index(drop=True)
    
    print(f"[Factor] 清洗后基差数据: {len(df_basis)} 行")
    
    # === 1. ✅ 修复：使用 transform + 布尔索引选择 CTD ===
    # 每日每合约取净基差最小的债券（CTD券）
    df_basis['min_net_basis'] = df_basis.groupby(['date', 'ticker'])['net_basis'].transform('min')
    df_ctd = df_basis[df_basis['net_basis'] == df_basis['min_net_basis']].copy()
    
    # 如果同一天有多个最小值，取第一个
    df_ctd = df_ctd.drop_duplicates(subset=['date', 'ticker'], keep='first')
    
    # 只保留需要的列
    df_ctd = df_ctd[['date', 'ticker', 'basis', 'net_basis', 'irr']].copy()
    df_ctd = df_ctd.rename(columns={
        'basis': 'Basis_CTD',
        'net_basis': 'NetBasis_CTD', 
        'irr': 'IRR_CTD'
    })
    
    print(f"[Factor] CTD筛选后: {len(df_ctd)} 行")
    
    # === 2. 计算核心因子（只保留2个）===
    df_ctd = df_ctd.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # 基差 Z-Score_20 和 Z-Score_10
    def calc_zscore(series, window=20):
        mean = series.rolling(window, min_periods=5).mean()
        std = series.rolling(window, min_periods=5).std()
        return ((series - mean) / (std + 1e-6)).clip(-3, 3)
    
    df_ctd['Basis_ZScore_20'] = df_ctd.groupby('ticker')['NetBasis_CTD'].transform(
        lambda x: calc_zscore(x, 20)
    )
    
    df_ctd['Basis_ZScore_10'] = df_ctd.groupby('ticker')['NetBasis_CTD'].transform(
        lambda x: calc_zscore(x, 10)
    )
    # 基差变化方向（3日）
    df_ctd['Basis_Trend'] = df_ctd.groupby('ticker')['NetBasis_CTD'].transform(
        lambda x: np.sign(x.diff(3))
    )
    
    # === 3. 延迟一天使用===
    df_ctd['available_date'] = df_ctd['date'] + pd.Timedelta(days=1)
    
    # 只保留需要的列
    result = df_ctd[['available_date', 'ticker', 'Basis_ZScore_20', 'Basis_ZScore_10','Basis_Trend']].copy()
    result = result.fillna(0)
    
    print(f"[Factor] 基差因子处理完成: {len(result)} 行，3 个因子")
    
    return result


def merge_basis_to_minute(df_main, df_basis):
    """将日频基差因子合并到分钟数据"""
    if df_basis is None:
        return df_main
    
    print("[Factor] 合并基差因子...")
    
    df_main = df_main.copy()
    df_main['trade_date'] = pd.to_datetime(df_main['date']).dt.date
    
    df_basis = df_basis.copy()
    df_basis['available_date'] = pd.to_datetime(df_basis['available_date']).dt.date
    
    df_merged = pd.merge(
        df_main,
        df_basis,
        left_on=['trade_date', 'ticker'],
        right_on=['available_date', 'ticker'],
        how='left'
    )
    
    if 'available_date' in df_merged.columns:
        df_merged.drop('available_date', axis=1, inplace=True)
    if 'trade_date' in df_merged.columns:
        df_merged.drop('trade_date', axis=1, inplace=True)
    
    # 填充因子
    for col in ['Basis_ZScore_20', 'Basis_ZScore_10', 'Basis_Trend']:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill().fillna(0)
    
    non_zero = 0
    for col in ['Basis_ZScore_20', 'Basis_ZScore_10']:
        if col in df_merged.columns:
            non_zero += (df_merged[col] != 0).sum()
    print(f"[Factor] 基差因子合并成功率: {non_zero/(2*len(df_merged)):.1%}")
    
    return df_merged


# ============================================================
# 微观结构数据处理
# ============================================================

def merge_tick_features(df_main, df_tick, bar_format='end'):
    """将快照数据特征合并到主力合约数据"""
    print("[Factor] 合并微观结构特征...")
    
    if df_tick is None or len(df_tick) == 0:
        print("[Factor WARNING] 无快照数据，跳过合并")
        return df_main
    
    df_main = df_main.copy()
    df_tick = df_tick.copy()
    
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_tick['date'] = pd.to_datetime(df_tick['date'])
    
    if bar_format == 'end':
        df_main['merge_minute'] = df_main['date'].dt.floor('min') - pd.Timedelta(minutes=1)
    else:
        df_main['merge_minute'] = df_main['date'].dt.floor('min')
    
    df_tick['merge_minute'] = df_tick['date'].dt.floor('min') - pd.Timedelta(minutes=1)
    
    base_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'money', 'oi', 
                 'merge_minute', 'high_tick', 'low_tick'}
    tick_feature_cols = [col for col in df_tick.columns if col not in base_cols]
    
    print(f"[Factor] 待合并的微观结构特征: {len(tick_feature_cols)} 个")
    
    df_tick_merge = df_tick[['merge_minute', 'ticker'] + tick_feature_cols].copy()
    df_tick_merge = df_tick_merge.drop_duplicates(subset=['merge_minute', 'ticker'], keep='last')
    
    df_merged = pd.merge(df_main, df_tick_merge, on=['merge_minute', 'ticker'], how='left')
    
    for col in tick_feature_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
    
    df_merged.drop('merge_minute', axis=1, inplace=True)
    
    non_zero_count = (df_merged[tick_feature_cols[0]] != 0).sum() if tick_feature_cols else 0
    print(f"[Factor] Tick特征合并成功率: {non_zero_count/len(df_merged):.1%}")
    
    return df_merged


# ============================================================
# 宏观因子数据处理
# ============================================================

def load_and_process_macro_data(base_dir, start_date="20230421", end_date="20251031", force_refresh=False):
    """
    Load or fetch macro data via AKShare and compute daily-frequency macro factors.

    Parameters:
        base_dir: project root directory
        start_date, end_date: YYYYMMDD date strings
        force_refresh: if True, bypass cache and re-fetch

    Returns:
        pd.DataFrame: daily macro factors with 'available_date' column,
                      or None if data fetching failed.
    """
    cache_dir = os.path.join(base_dir, "data/macro")
    os.makedirs(cache_dir, exist_ok=True)

    MACRO_FACTOR_CACHE = os.path.join(base_dir, "outputs/macro_factors.pkl")

    if os.path.exists(MACRO_FACTOR_CACHE) and not force_refresh:
        print("[Factor] 加载已缓存的宏观因子")
        return pd.read_pickle(MACRO_FACTOR_CACHE)

    try:
        # Step 1: Fetch raw data
        print("[Factor] 开始从 AKShare 获取宏观数据...")
        fetcher = MacroDataFetcher(cache_dir=cache_dir, start_date=start_date, end_date=end_date)
        raw_data = fetcher.fetch_all()

        if not raw_data:
            print("[Factor WARNING] 未能获取任何宏观数据")
            return None

        # Step 2: Compute macro factors
        computer = MacroFactorComputer()
        df_macro = computer.compute_all(raw_data)

        if df_macro is None or len(df_macro) == 0:
            print("[Factor WARNING] 宏观因子计算为空")
            return None

        # Step 3: Prepare for merge — shift by 1 day as available_date key
        df_macro['available_date'] = df_macro['date'] + pd.Timedelta(days=1)
        df_macro = df_macro.drop(columns=['date'])

        # Step 4: Cache
        df_macro.to_pickle(MACRO_FACTOR_CACHE)
        print(f"[Factor] 宏观因子已缓存 ({len(df_macro)} 行, {len(df_macro.columns)-1} 个因子)")

        return df_macro

    except Exception as e:
        import traceback
        print(f"[Factor WARNING] 宏观数据加载失败: {e}")
        traceback.print_exc()
        return None


def merge_macro_to_minute(df_main, df_macro):
    """
    Merge daily-frequency macro factors into minute-level DataFrame.
    Macro factors are contract-independent — merge on date only.

    Parameters:
        df_main: minute-level DataFrame with 'date' column
        df_macro: daily DataFrame with 'available_date' column

    Returns:
        pd.DataFrame: df_main with macro factor columns added
    """
    if df_macro is None or len(df_macro) == 0:
        return df_main

    print("[Factor] 合并宏观因子...")

    df_main = df_main.copy()
    df_main['trade_date'] = pd.to_datetime(df_main['date']).dt.date

    df_macro = df_macro.copy()
    df_macro['available_date'] = pd.to_datetime(df_macro['available_date']).dt.date

    # Left join on date only (macro factors apply to all contracts equally)
    df_merged = pd.merge(
        df_main,
        df_macro,
        left_on=['trade_date'],
        right_on=['available_date'],
        how='left'
    )

    if 'available_date' in df_merged.columns:
        df_merged.drop('available_date', axis=1, inplace=True)
    if 'trade_date' in df_merged.columns:
        df_merged.drop('trade_date', axis=1, inplace=True)

    # Forward fill and fill remaining NaN with 0
    macro_cols = [c for c in df_macro.columns if c != 'available_date']
    for col in macro_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill().fillna(0)

    non_zero = sum((df_merged[c] != 0).sum() for c in macro_cols if c in df_merged.columns)
    total_cells = len(macro_cols) * len(df_merged) if macro_cols else 1
    print(f"[Factor] 宏观因子合并成功率: {non_zero / total_cells:.1%}")

    return df_merged


# ============================================================
# 核心因子计算
# ============================================================

def calculate_core_factors(df):
    """V9.7 核心因子"""
    print("[Factor] 计算核心因子...")
    
    close_lagged = df['close'].shift(1)
    ret_lagged = close_lagged.pct_change()
    
    # === 1. 动量因子 ===
    df['Short_Momentum_1D'] = close_lagged.pct_change(240)
    df['Short_Momentum_3D'] = close_lagged.pct_change(720)
    df['Short_Momentum_5D'] = close_lagged.pct_change(1200)
    df['Mid_Momentum_1M'] = close_lagged.pct_change(5000)
    df['Mid_Momentum_2M'] = close_lagged.pct_change(10000)
    
    weights = np.exp(-np.arange(60) / 20)[::-1]
    weights = weights / weights.sum()
    df['TSMOM'] = ret_lagged.rolling(60).apply(
        lambda x: np.dot(x, weights) if len(x) == 60 else 0, raw=False
    )
    
    mom_1d = np.sign(df['Short_Momentum_1D'])
    mom_3d = np.sign(df['Short_Momentum_3D'])
    mom_5d = np.sign(df['Short_Momentum_5D'])
    df['Momentum_Alignment'] = (mom_1d + mom_3d + mom_5d) / 3
    
    # === 2. 波动率因子 ===
    df['RV_30'] = ret_lagged.rolling(30).std() * np.sqrt(240) * 100
    df['RV_120'] = ret_lagged.rolling(120).std() * np.sqrt(240) * 100
    
    rv_5 = ret_lagged.rolling(5).std() * np.sqrt(240)
    rv_60 = ret_lagged.rolling(60).std() * np.sqrt(240)
    df['Vol_Surge'] = (rv_5 / (rv_60 + 1e-9)).clip(0, 10)
    
    high_lag = df['high'].shift(1)
    low_lag = df['low'].shift(1)
    close_lag2 = df['close'].shift(2)
    
    tr = pd.DataFrame()
    tr['hl'] = high_lag - low_lag
    tr['hc'] = abs(high_lag - close_lag2)
    tr['lc'] = abs(low_lag - close_lag2)
    df['TR'] = tr.max(axis=1)
    df['ATR_14'] = df['TR'].rolling(14).mean()
    
    # === 3. 市场状态识别 ===
    rv_for_percentile = df['RV_30'].shift(1)
    df['RV_Percentile'] = rv_for_percentile.rolling(480, min_periods=60).rank(pct=True)
    df['Is_High_Vol'] = (df['RV_Percentile'] > 0.85).astype(int)
    
    price_direction_lagged = np.sign(ret_lagged)
    df['Trend_Consistency'] = price_direction_lagged.rolling(5).apply(
        lambda x: 1 if len(x) == 5 and abs(x.sum()) == 5 else 0, raw=False
    )
    
    rv_lagged = df['RV_30'].shift(1)
    vol_median = rv_lagged.rolling(480, min_periods=60).median()
    df['Vol_Regime'] = (rv_lagged / (vol_median + 0.01) - 1).clip(-2, 2)
    
    df['Market_Regime'] = 0
    df.loc[df['Is_High_Vol'] == 1, 'Market_Regime'] = 1
    df.loc[df['Trend_Consistency'] == 1, 'Market_Regime'] = 2
    
    # === 4. 量价因子 ===
    oi_lagged = df['oi'].shift(1)
    volume_lagged = df['volume'].shift(1)
    
    oi_pct = oi_lagged.pct_change(30)
    vol_pct = volume_lagged.pct_change(30)
    
    df['OI_Volume_Flow'] = (oi_pct / (vol_pct + 1e-9)).clip(-5, 5)
    df['Smart_Money'] = close_lagged.pct_change(30) * oi_pct
    
    vol_ma = volume_lagged.rolling(60).mean()
    df['Large_Trade_Ratio'] = (volume_lagged / (vol_ma + 1)).clip(0, 5)
    df['Large_Trade_Direction'] = df['Large_Trade_Ratio'] * np.sign(close_lagged.pct_change())
    
    # === 5. 技术因子 ===
    ema_12 = close_lagged.ewm(span=12, adjust=False).mean()
    ema_26 = close_lagged.ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema_12 - ema_26) / (close_lagged + 1e-9) * 100
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    delta = close_lagged.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    bb_ma = close_lagged.rolling(20).mean()
    bb_std = close_lagged.rolling(20).std()
    df['BB_Position'] = (close_lagged - (bb_ma - 2*bb_std)) / (4*bb_std + 1e-6)
    
    return df


def calculate_combined_factors(df):
    """计算组合因子"""
    print("[Factor] 计算组合因子...")
    
    # === 微观结构 + 动量交互 ===
    if 'Cum_Imbalance_5' in df.columns:
        imb_lagged = df['Cum_Imbalance_5'].shift(1)
        df['Imbalance_Momentum_Confirm'] = imb_lagged * np.sign(df['Short_Momentum_1D'])
        
        if 'Imbalance_ZScore' in df.columns:
            imb_zscore_lagged = df['Imbalance_ZScore'].shift(1)
            imb_strong = abs(imb_zscore_lagged) > 1.5
            mom_std = df['Short_Momentum_1D'].rolling(60).std()
            mom_strong = abs(df['Short_Momentum_1D']) > mom_std
            df['Strong_Signal'] = (imb_strong & mom_strong).astype(int)
    
    if 'HF_RV_5' in df.columns:
        hf_rv_lagged = df['HF_RV_5'].shift(1)
        df['Vol_Disconnect'] = hf_rv_lagged / (df['RV_30'].shift(1) + 0.01)
    
    if 'Signed_Vol_5' in df.columns:
        signed_vol_lagged = df['Signed_Vol_5'].shift(1)
        df['Informed_Trading'] = signed_vol_lagged * df['OI_Volume_Flow']
    
    if 'Cum_Net_Open_15' in df.columns:
        net_open_lagged = df['Cum_Net_Open_15'].shift(1)
        df['Open_Price_Push'] = net_open_lagged * df['Short_Momentum_1D']
        
        if 'Close_Ratio' in df.columns:
            close_ratio_lagged = df['Close_Ratio'].shift(1)
            df['Close_Pressure'] = -close_ratio_lagged * abs(df['Short_Momentum_1D'])
    
    return df


def calculate_enhanced_factors(df, df_tick=None, df_basis=None, df_macro=None, bar_format='end'):
    df = df.copy()
    print("[Factor] 开始计算因子 ...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['Hour'] = df['date'].dt.hour
    df['Minute'] = df['date'].dt.minute
    df['Minute_of_Day'] = df['Hour'] * 60 + df['Minute']
    
    # 1. 合并微观结构特征
    if df_tick is not None and len(df_tick) > 0:
        df = merge_tick_features(df, df_tick, bar_format=bar_format)
    
    # 2. 合并基差因子
    if df_basis is not None:
        df = merge_basis_to_minute(df, df_basis)
    # 3. 合并宏观因子
    if df_macro is not None:
        df = merge_macro_to_minute(df, df_macro)

    # 4. 计算核心因子
    df = calculate_core_factors(df)
    
    # 5. 计算组合因子
    df = calculate_combined_factors(df)
    
    # 清洗
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    min_valid_rows = 500
    df_clean = df.iloc[min_valid_rows:].copy()
    df_clean = df_clean.ffill().fillna(0)
    
    # 统计
    micro_cols = [col for col in df_clean.columns if any(
        keyword in col for keyword in ['Spread', 'Depth', 'Imbalance', 'VPIN', 'HF_', 'Signed', 'Trade_']
    )]
    basis_cols = [col for col in df_clean.columns if 'Basis' in col]
    
    macro_cols = [col for col in df_clean.columns if any(
        keyword in col for keyword in ['SHIBOR', 'Repo', 'YC_', 'PMI', 'CPI', 'M2_',
                                        'SocialFin', 'Injection', 'OMO', 'Stock_Bond',
                                        'CN_US', 'Credit', 'Risk_On', 'Liquidity', 'Macro_Surprise']
    )]

    print(f"[Factor] 有效数据: {len(df_clean)} 行")
    print(f"[Factor] 总特征数: {len(df_clean.columns)}")
    print(f"[Factor] 微观结构特征数: {len(micro_cols)}")
    print(f"[Factor] 基差特征数: {len(basis_cols)}")
    print(f"[Factor] 宏观特征数: {len(macro_cols)}")
    
    if 'Market_Regime' in df_clean.columns:
        print("\n[Factor] 市场状态分布:")
        for regime_id in [0, 1, 2]:
            count = (df_clean['Market_Regime'] == regime_id).sum()
            pct = count / len(df_clean)
            regime_name = ['正常', '高波动', '趋势'][regime_id]
            print(f"  - {regime_name}市: {count} 样本 ({pct:.1%})")
    
    return df_clean


def run_process(base_dir, tick_subdir="data/tick", basis_file="data/TL合约价差日频数据.pkl"):
    """因子构建主流程"""
    print("\n" + "="*50)
    print("STEP 2: 因子构建")
    print("="*50)
    
    RAW_FILE = os.path.join(base_dir, "data/TL分钟级量价数据.pkl")
    BASIS_FILE = os.path.join(base_dir, basis_file)
    TICK_FEATURE_FILE = os.path.join(base_dir, "outputs/tick_minute_features.pkl")
    OUTPUT_FILE = os.path.join(base_dir, "outputs/df_factors.pkl")
    INTERMEDIATE_FILE = os.path.join(base_dir, "data/main_contract_spliced.pkl")
    
    if not os.path.exists(RAW_FILE):
        print(f"[Factor ERROR] 找不到源文件: {RAW_FILE}")
        return False

    try:
        # 1. 加载主力合约数据
        df_main = load_data(RAW_FILE, INTERMEDIATE_FILE)
        
        # 2. 检测时间戳格式
        bar_format = detect_bar_format(df_main)
        
        # 3. 处理快照数据
        df_tick = None
        TICK_DIR = os.path.join(base_dir, tick_subdir)
        
        if os.path.exists(TICK_DIR) or os.path.exists(TICK_FEATURE_FILE):
            if os.path.exists(TICK_FEATURE_FILE):
                print(f"[Factor] 加载已处理的快照特征: {TICK_FEATURE_FILE}")
                df_tick = pd.read_pickle(TICK_FEATURE_FILE)
            else:
                df_tick = run_tick_processing(base_dir, tick_subdir)
            
            if df_tick is not None:
                df_tick = calculate_microstructure_factors(df_tick)
        
        # 4. 处理基差数据
        df_basis = None
        if os.path.exists(BASIS_FILE):
            df_basis = load_and_process_basis_data(BASIS_FILE)
        else:
            print(f"[Factor WARNING] 未找到基差数据: {BASIS_FILE}")

        # === 4.5. 处理宏观数据 ===
        date_min = pd.to_datetime(df_main['date']).min().strftime('%Y%m%d')
        date_max = pd.to_datetime(df_main['date']).max().strftime('%Y%m%d')

        df_macro = load_and_process_macro_data(
            base_dir,
            start_date=date_min,
            end_date=date_max,
            force_refresh=False
        )

        # 5. 计算所有因子
        df_factors = calculate_enhanced_factors(df_main, df_tick, df_basis, df_macro=df_macro, bar_format=bar_format)

        # 6. 保存
        df_factors.to_pickle(OUTPUT_FILE)
        print(f"[Factor SUCCESS] 因子已保存: {OUTPUT_FILE}")

        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False