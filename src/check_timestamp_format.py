# -*- coding: utf-8 -*-
# check_timestamp_format.py
# 检查分钟数据的时间戳格式

import pandas as pd
import os

def check_bar_timestamp_format(base_dir):
    """
    检查分钟K线数据的时间戳格式
    判断是 Start-of-Bar 还是 End-of-Bar
    """
    print("\n" + "="*60)
    print("检查分钟数据时间戳格式")
    print("="*60)
    
    # 加载分钟数据
    MIN_FILE = os.path.join(base_dir, "data/TL分钟级量价数据.pkl")
    TICK_FEATURE_FILE = os.path.join(base_dir, "outputs/tick_minute_features.pkl")
    
    if not os.path.exists(MIN_FILE):
        print(f"[ERROR] 找不到分钟数据: {MIN_FILE}")
        return None
    
    df_min = pd.read_pickle(MIN_FILE)
    df_min['date'] = pd.to_datetime(df_min['date'])
    
    # 查看第一个交易日的数据
    first_date = df_min['date'].dt.date.iloc[0]
    df_first_day = df_min[df_min['date'].dt.date == first_date].head(20)
    
    print("\n【分钟数据】前 20 行:")
    print(df_first_day[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']].to_string())
    
    # 检查第一根K线的时间
    first_bar_time = df_first_day['date'].iloc[0]
    print(f"\n第一根K线时间戳: {first_bar_time}")
    
    # 判断格式
    first_minute = first_bar_time.minute
    first_hour = first_bar_time.hour
    
    print(f"\n【诊断结果】")
    
    # 国债期货 TL 的交易时间：09:30 - 11:30, 13:00 - 15:15
    if first_hour == 9 and first_minute == 30:
        print("✅ 时间戳格式: Start-of-Bar（09:30:00）")
        print("   含义: 09:30:00 代表 09:30:00 ~ 09:31:00 这一分钟的K线")
        print("   结论: 当前合并逻辑【安全】")
        bar_format = "start"
    elif first_hour == 9 and first_minute == 31:
        print("⚠️  时间戳格式: End-of-Bar（09:31:00）")
        print("   含义: 09:31:00 代表 09:30:00 ~ 09:31:00 这一分钟的K线")
        print("   结论: 当前合并逻辑【存在时间泄露风险】")
        bar_format = "end"
    else:
        print(f"⚠️  无法确定格式，第一根K线时间: {first_bar_time}")
        print("   请手动检查数据含义")
        bar_format = "unknown"
    
    # 如果有 tick 数据，也检查一下
    if os.path.exists(TICK_FEATURE_FILE):
        df_tick = pd.read_pickle(TICK_FEATURE_FILE)
        df_tick['date'] = pd.to_datetime(df_tick['date'])
        
        # 找到同一天的 tick 数据
        df_tick_first_day = df_tick[df_tick['date'].dt.date == first_date].head(20)
        
        if len(df_tick_first_day) > 0:
            print("\n【Tick 聚合数据】前 20 行:")
            cols = ['date', 'ticker'] + [c for c in df_tick_first_day.columns if c not in ['date', 'ticker']][:5]
            print(df_tick_first_day[cols].to_string())
            
            first_tick_time = df_tick_first_day['date'].iloc[0]
            print(f"\n第一条 Tick 聚合数据时间戳: {first_tick_time}")
    
    return bar_format


def check_tick_raw_data(base_dir, tick_subdir="data/tick"):
    """
    检查原始 Tick 数据的时间戳
    """
    print("\n" + "="*60)
    print("检查原始 Tick 数据时间戳")
    print("="*60)
    
    TICK_DIR = os.path.join(base_dir, tick_subdir)
    
    if not os.path.exists(TICK_DIR):
        print(f"[WARNING] Tick 目录不存在: {TICK_DIR}")
        return
    
    # 读取第一个 tick 文件
    import glob
    tick_files = sorted(glob.glob(os.path.join(TICK_DIR, "*.pkl")))
    
    if len(tick_files) == 0:
        print("[WARNING] 没有找到 tick 文件")
        return
    
    df_tick = pd.read_pickle(tick_files[0])
    print(f"\n读取 Tick 文件: {os.path.basename(tick_files[0])}")
    print(f"\n【原始 Tick 数据】前 20 行:")
    print(df_tick.head(20).to_string())
    
    # 检查 time 列格式
    if 'time' in df_tick.columns:
        print(f"\n第一条 Tick 时间: {df_tick['time'].iloc[0]}")
        print(f"最后一条 Tick 时间: {df_tick['time'].iloc[-1]}")


if __name__ == '__main__':
    BASE_DIR = r"D:\桌面\蒋政_固收课题"
    
    bar_format = check_bar_timestamp_format(BASE_DIR)
    check_tick_raw_data(BASE_DIR)
    
    print("\n" + "="*60)
    if bar_format == "end":
        print("【重要】需要修复合并逻辑！")
        print("修复方案：将分钟数据时间戳向前偏移1分钟，或将Tick数据向后偏移1分钟")
    elif bar_format == "start":
        print("【安全】当前合并逻辑正确")
    print("="*60)