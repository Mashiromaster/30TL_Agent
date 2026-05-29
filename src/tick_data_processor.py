# -*- coding: utf-8 -*-
# tick_data_processor.py
# 半秒快照数据处理模块 (修复版)

import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

class TickDataProcessor:
    """
    处理半秒快照数据，聚合为分钟级微观结构因子
    """
    
    def __init__(self, tick_data_dir):
        """
        参数:
            tick_data_dir: 存放每日pkl文件的目录
        """
        self.tick_data_dir = tick_data_dir
        self.tick_files = sorted(glob(os.path.join(tick_data_dir, "*.pkl")))
        print(f"[TickProcessor] 发现 {len(self.tick_files)} 个快照数据文件")
    
    def load_single_day(self, file_path):
        """加载单日快照数据"""
        try:
            df = pd.read_pickle(file_path)
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            return df
        except Exception as e:
            print(f"[TickProcessor] 加载失败: {file_path}, 错误: {e}")
            return None
    
    def calculate_tick_features(self, df_tick):
        """
        计算单个合约的半秒级特征
        """
        df = df_tick.copy()
        
        # === 1. 基础盘口特征 ===
        df['Spread'] = df['卖一价格'] - df['买一价格']
        df['Spread_Pct'] = df['Spread'] / (df['last'] + 1e-9) * 10000
        df['Mid_Price'] = (df['卖一价格'] + df['买一价格']) / 2
        df['Effective_Spread'] = 2 * abs(df['last'] - df['Mid_Price'])
        df['Effective_Spread_Pct'] = df['Effective_Spread'] / (df['last'] + 1e-9) * 10000
        
        # === 2. 深度不平衡 ===
        df['Depth_Imbalance'] = (df['买一手数'] - df['卖一手数']) / (df['买一手数'] + df['卖一手数'] + 1)
        df['Depth_Ratio'] = df['买一手数'] / (df['卖一手数'] + 1)
        df['Depth_Ratio_Log'] = np.log(df['Depth_Ratio'] + 0.01)
        df['Total_Depth'] = df['买一手数'] + df['卖一手数']
        
        # === 3. 价格动态 ===
        df['Price_Change'] = df['last'].diff()
        df['Mid_Price_Change'] = df['Mid_Price'].diff()
        df['Price_Direction'] = np.sign(df['Price_Change'])
        df['HF_Return'] = df['last'].pct_change() * 10000
        
        # === 4. 成交量动态 ===
        df['Volume_Change'] = df['volume'].diff().clip(lower=0)
        df['Money_Change'] = df['money'].diff().clip(lower=0)
        df['Trade_Price'] = np.where(
            df['Volume_Change'] > 0,
            df['Money_Change'] / (df['Volume_Change'] * 10000 + 1),
            df['last']
        )
        
        # === 5. 订单流毒性 ===
        df['Trade_Sign'] = np.where(
            df['last'] > df['Mid_Price'], 1,
            np.where(df['last'] < df['Mid_Price'], -1, 0)
        )
        df['Signed_Volume'] = df['Trade_Sign'] * df['Volume_Change']
        
        # === 6. 持仓量动态 ===
        df['OI_Change'] = df['oi'].diff()
        df['Is_Open'] = ((df['Volume_Change'] > 0) & (df['OI_Change'] > 0)).astype(int)
        df['Is_Close'] = ((df['Volume_Change'] > 0) & (df['OI_Change'] < 0)).astype(int)
        
        return df
    
    def aggregate_to_minute(self, df_tick, ticker):
        """
        将半秒快照聚合为分钟级特征
        """
        df = df_tick[df_tick['ticker'] == ticker].copy()
        
        if len(df) == 0:
            return None
        
        # 计算半秒级特征
        df = self.calculate_tick_features(df)
        
        # 创建分钟时间戳
        df['minute'] = df['datetime'].dt.floor('min')
        
        # 按分钟聚合
        agg_funcs = {
            # OHLC
            'last': ['first', 'max', 'min', 'last'],
            'high': 'max',
            'low': 'min',
            'volume': 'last',
            'money': 'last',
            'oi': 'last',
            
            # 微观结构因子
            'Spread_Pct': ['mean', 'max', 'std'],
            'Depth_Imbalance': ['mean', 'std', 'last'],
            'Total_Depth': ['mean', 'min'],
            'HF_Return': ['sum', 'std'],
            'Signed_Volume': 'sum',
            'Volume_Change': lambda x: (x > 0).sum(),
            'Is_Open': 'sum',
            'Is_Close': 'sum',
            'Price_Direction': lambda x: (x > 0).mean() if len(x) > 0 else 0.5,
            'Effective_Spread_Pct': 'mean',
        }
        
        df_minute = df.groupby('minute').agg(agg_funcs)
        
        # 展平多级列名
        df_minute.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                            for col in df_minute.columns.values]
        
        # 重命名列
        rename_dict = {
            'last_first': 'open',
            'last_max': 'high_tick',
            'last_min': 'low_tick', 
            'last_last': 'close',
            'high_max': 'high',
            'low_min': 'low',
            'volume_last': 'volume',
            'money_last': 'money',
            'oi_last': 'oi',
            'Spread_Pct_mean': 'Spread_Mean',
            'Spread_Pct_max': 'Spread_Max',
            'Spread_Pct_std': 'Spread_Std',
            'Depth_Imbalance_mean': 'Depth_Imbalance_Mean',
            'Depth_Imbalance_std': 'Depth_Imbalance_Std',
            'Depth_Imbalance_last': 'Depth_Imbalance_Last',
            'Total_Depth_mean': 'Total_Depth_Mean',
            'Total_Depth_min': 'Total_Depth_Min',
            'HF_Return_sum': 'HF_Return_Sum',
            'HF_Return_std': 'HF_Return_Std',
            'Signed_Volume_sum': 'Signed_Volume_Sum',
            'Volume_Change_<lambda>': 'Trade_Count',
            'Is_Open_sum': 'Open_Count',
            'Is_Close_sum': 'Close_Count',
            'Price_Direction_<lambda>': 'Up_Tick_Ratio',
            'Effective_Spread_Pct_mean': 'Effective_Spread_Mean',
        }
        
        df_minute = df_minute.rename(columns=rename_dict)
        df_minute = df_minute.reset_index()
        df_minute = df_minute.rename(columns={'minute': 'date'})
        
        # 添加合约代码
        df_minute['ticker'] = ticker
        
        # 处理 NaN
        df_minute = df_minute.fillna(0)
        
        # 删除不需要的列
        cols_to_drop = [col for col in df_minute.columns if '<lambda>' in col]
        df_minute = df_minute.drop(columns=cols_to_drop, errors='ignore')
        
        return df_minute
    
    def process_single_day(self, file_path):
        """处理单日数据"""
        df_tick = self.load_single_day(file_path)
        
        if df_tick is None:
            return None
        
        tickers = df_tick['ticker'].unique()
        
        results = []
        for ticker in tickers:
            df_minute = self.aggregate_to_minute(df_tick, ticker)
            if df_minute is not None:
                results.append(df_minute)
        
        if len(results) == 0:
            return None
        
        return pd.concat(results, ignore_index=True)
    
    def process_all_days(self, output_path):
        """处理所有日期的快照数据"""
        print(f"[TickProcessor] 开始处理 {len(self.tick_files)} 个文件...")
        
        all_results = []
        
        for file_path in tqdm(self.tick_files, desc="处理快照数据"):
            df_day = self.process_single_day(file_path)
            if df_day is not None:
                all_results.append(df_day)
        
        if len(all_results) == 0:
            print("[TickProcessor] 没有成功处理任何文件")
            return None
        
        df_all = pd.concat(all_results, ignore_index=True)
        df_all = df_all.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        df_all.to_pickle(output_path)
        print(f"[TickProcessor] 处理完成，共 {len(df_all)} 行，保存至: {output_path}")
        
        return df_all


def calculate_microstructure_factors(df):
    """
    基于聚合后的分钟数据计算更高级的微观结构因子
    """
    print("[Microstructure] 计算高级微观结构因子...")
    
    df = df.copy()
    
    # === 1. 滚动价差因子 ===
    if 'Spread_Mean' in df.columns:
        df['Spread_MA_5'] = df['Spread_Mean'].rolling(5, min_periods=1).mean()
        df['Spread_MA_30'] = df['Spread_Mean'].rolling(30, min_periods=1).mean()
        df['Spread_Ratio'] = df['Spread_Mean'] / (df['Spread_MA_30'] + 0.01)
        
        spread_q95 = df['Spread_Mean'].shift(1).rolling(60, min_periods=10).quantile(0.95)
        df['Spread_Shock'] = (df['Spread_Mean'] > spread_q95).astype(int)
    
    # === 2. 订单不平衡因子 ===
    if 'Depth_Imbalance_Mean' in df.columns:
        df['Cum_Imbalance_5'] = df['Depth_Imbalance_Mean'].rolling(5, min_periods=1).sum()
        df['Cum_Imbalance_15'] = df['Depth_Imbalance_Mean'].rolling(15, min_periods=1).sum()
        df['Cum_Imbalance_30'] = df['Depth_Imbalance_Mean'].rolling(30, min_periods=1).sum()
        df['Imbalance_Momentum'] = df['Cum_Imbalance_5'] - df['Cum_Imbalance_15']
        
        imb_std = df['Depth_Imbalance_Mean'].rolling(60, min_periods=10).std()
        df['Imbalance_ZScore'] = df['Depth_Imbalance_Mean'] / (imb_std + 0.01)
    
    # === 3. 订单流毒性因子 ===
    if 'Signed_Volume_Sum' in df.columns:
        df['Signed_Vol_5'] = df['Signed_Volume_Sum'].rolling(5, min_periods=1).sum()
        df['Signed_Vol_15'] = df['Signed_Volume_Sum'].rolling(15, min_periods=1).sum()
        
        vol_diff = df['volume'].diff().abs()
        df['Total_Vol_5'] = vol_diff.rolling(5, min_periods=1).sum()
        df['Total_Vol_15'] = vol_diff.rolling(15, min_periods=1).sum()
        
        df['VPIN_5'] = abs(df['Signed_Vol_5']) / (df['Total_Vol_5'] + 1)
        df['VPIN_15'] = abs(df['Signed_Vol_15']) / (df['Total_Vol_15'] + 1)
    
    # === 4. 成交活跃度因子 ===
    if 'Trade_Count' in df.columns:
        trade_ma = df['Trade_Count'].rolling(30, min_periods=1).mean()
        df['Trade_Intensity'] = df['Trade_Count'] / (trade_ma + 1)
        df['Trade_Intensity_Shock'] = (df['Trade_Intensity'] > 2).astype(int)
    
    # === 5. 开平仓比例 ===
    if 'Open_Count' in df.columns and 'Close_Count' in df.columns:
        total_trades = df['Open_Count'] + df['Close_Count'] + 1
        df['Open_Ratio'] = df['Open_Count'] / total_trades
        df['Close_Ratio'] = df['Close_Count'] / total_trades
        df['Net_Open'] = df['Open_Count'] - df['Close_Count']
        df['Cum_Net_Open_15'] = df['Net_Open'].rolling(15, min_periods=1).sum()
        df['Cum_Net_Open_30'] = df['Net_Open'].rolling(30, min_periods=1).sum()
    
    # === 6. 高频动率因子 ===
    if 'HF_Return_Std' in df.columns:
        df['HF_RV_5'] = df['HF_Return_Std'].rolling(5, min_periods=1).mean() * np.sqrt(120)
        df['HF_RV_30'] = df['HF_Return_Std'].rolling(30, min_periods=1).mean() * np.sqrt(120)
        df['HF_Vol_Ratio'] = df['HF_RV_5'] / (df['HF_RV_30'] + 0.01)
    
    # === 7. 价格压力因子 ===
    if 'Up_Tick_Ratio' in df.columns:
        df['Up_Tick_MA_5'] = df['Up_Tick_Ratio'].rolling(5, min_periods=1).mean()
        df['Up_Tick_MA_15'] = df['Up_Tick_Ratio'].rolling(15, min_periods=1).mean()
        df['Price_Pressure'] = df['Up_Tick_MA_5'] - 0.5
        df['Price_Pressure_Momentum'] = df['Price_Pressure'].diff(5)
    
    # === 8. 流动性因子 ===
    if 'Total_Depth_Min' in df.columns and 'Total_Depth_Mean' in df.columns:
        df['Depth_Decay'] = df['Total_Depth_Min'] / (df['Total_Depth_Mean'] + 1)
    
    if 'HF_Return_Sum' in df.columns and 'Trade_Count' in df.columns:
        df['HF_Illiquidity'] = df['HF_Return_Sum'].abs() / (df['Trade_Count'] + 1)
        df['HF_Illiquidity_MA'] = df['HF_Illiquidity'].rolling(30, min_periods=1).mean()
        df['Liquidity_Risk'] = df['HF_Illiquidity'] / (df['HF_Illiquidity_MA'] + 0.01)
    
    # 填充 NaN
    df = df.fillna(0)
    
    return df


def run_tick_processing(base_dir, tick_subdir="data/tick"):
    """运行快照数据处理流程"""
    print("\n" + "="*50)
    print("STEP 1.5: 快照数据处理")
    print("="*50)
    
    TICK_DIR = os.path.join(base_dir, tick_subdir)
    OUTPUT_FILE = os.path.join(base_dir, "outputs/tick_minute_features.pkl")
    
    if not os.path.exists(TICK_DIR):
        print(f"[TickProcessor ERROR] 找不到快照数据目录: {TICK_DIR}")
        return None
    
    if os.path.exists(OUTPUT_FILE):
        print(f"[TickProcessor] 发现已处理的数据: {OUTPUT_FILE}")
        df = pd.read_pickle(OUTPUT_FILE)
        print(f"[TickProcessor] 加载完成，共 {len(df)} 行")
        return df
    
    processor = TickDataProcessor(TICK_DIR)
    df = processor.process_all_days(OUTPUT_FILE)
    
    return df