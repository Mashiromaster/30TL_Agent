import pandas as pd
import os

def process_all_pkl_files():
    # 1. 获取当前脚本所在的目录
    if '__file__' in globals():
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        current_dir = os.getcwd()

    print(f"当前扫描目录: {current_dir}\n")
    print("="*50)

    # 2. 获取目录下所有文件并筛选 .pkl
    all_files = os.listdir(current_dir)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]

    if not pkl_files:
        print("未找到任何 .pkl 文件。")
        return

    # 3. 循环处理每个 pkl 文件
    for file_name in pkl_files:
        file_path = os.path.join(current_dir, file_name)
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(current_dir, txt_name)

        print(f"正在处理文件: {file_name} ...")

        try:
            # 读取 pickle 文件
            data = pd.read_pickle(file_path)

            # 判断数据类型
            if isinstance(data, (pd.DataFrame, pd.Series)):
                # --- 修改核心：不再使用 .head(1000) ---
                # 使用 to_string() 时，如果不设置限制，默认会转换所有数据
                # 但对于超大型 DataFrame，直接 to_string 可能会非常耗内存
                output_content = data.to_string()
                row_count = len(data)
                info_msg = f"  -> 类型: {type(data).__name__}, 已提取全部 {row_count} 行"
            
            else:
                # 其他类型直接转字符串
                output_content = str(data)
                info_msg = f"  -> 类型: {type(data).__name__}, 已全部转换为文本"

            # --- 1. 打印到控制台 (仅预览前5行，防止刷屏) ---
            print(info_msg)
            print("  -> 内容预览 (前5行):")
            print("-" * 30)
            print('\n'.join(str(output_content).split('\n')[:5])) 
            print("-" * 30)

            # --- 2. 保存为 TXT 文件 (保存全部数据) ---
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            print(f"  -> [成功] 全量数据已保存至: {txt_name}\n")

        except Exception as e:
            print(f"  -> [错误] 无法读取或处理 {file_name}: {e}\n")

    print("="*50)
    print("所有文件全量转换完毕。")

if __name__ == '__main__':
    process_all_pkl_files()