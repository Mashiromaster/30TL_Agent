# -*- coding: utf-8 -*-
# 文件名: mc_ex.py
import pandas as pd
import numpy as np
import os

def get_main_contract_data(min_file_path):
    """
    读取分钟数据，拼接主力合约，并计算复权价格
    """
    if not os.path.exists(min_file_path):
        raise FileNotFoundError(f"未找到文件: {min_file_path}")

    print(f"[mc_ex] 正在读取文件: {min_file_path} ...")
    df = pd.read_pickle(min_file_path)
    
    # 1. 确定每日主力合约 
    daily_oi = df.sort_values('date').groupby(['trade_dt', 'ticker'])['oi'].last().unstack(fill_value=0)
    target_ticker_daily = daily_oi.idxmax(axis=1)
    
    main_contract_list = []
    prev_ticker = None
    
    print("[mc_ex] 正在拼接主力合约并处理换月缺口...")
    
    # 2. 循环提取并标记换月
    for trade_date, ticker in target_ticker_daily.items():
        # 提取当日该合约数据
        day_data = df[(df['trade_dt'] == trade_date) & (df['ticker'] == ticker)].copy()
        
        # 标记是否发生主力切换
        if prev_ticker is not None and ticker != prev_ticker:
            day_data['is_switch'] = 1
        else:
            day_data['is_switch'] = 0
            
        main_contract_list.append(day_data)
        prev_ticker = ticker
        
    main_df = pd.concat(main_contract_list).sort_values('date').reset_index(drop=True)
    
    # 3. 计算后复权价格
    # 构造一个连续的收益率序列，然后累乘得到 adj_close
    # 计算原始分钟收益率
    main_df['raw_ret'] = main_df['close'].pct_change().fillna(0)
    
    # 识别换月时刻：当合约代码发生变化时，pct_change 计算出的收益是虚假的（新合约/旧合约 - 1）
    # 利用 ticker 变化来识别这一时刻，强制将收益归零
    main_df['ticker_shift'] = main_df['ticker'].shift(1)
    mask_switch = (main_df['ticker'] != main_df['ticker_shift']) & (main_df['ticker_shift'].notnull())
    
    # 将换月瞬间的收益抹平
    main_df.loc[mask_switch, 'raw_ret'] = 0
    
    # 重建复权净值
    main_df['adj_close'] = 100 * (1 + main_df['raw_ret']).cumprod()
    
    print(f"[mc_ex] 处理完成，数据行数: {len(main_df)}")
    return main_df

if __name__ == '__main__':
    print("Please run main.py")