# analyze_batch_result.py
"""
批量回测结果分析脚本
读取 batch_batch_summary_xxx.csv，生成 Markdown 对比表格
"""
import os
import glob
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "./output"
IMAGES_DIR = "./images"

def load_latest_summary():
    """读取最新一份批量汇总csv"""
    pattern = os.path.join(OUTPUT_DIR, "batch_batch_summary_*.csv")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise FileNotFoundError("未找到 batch_batch_summary_*.csv，请先运行批量回测！")
    latest_file = max(files, key=os.path.getmtime)
    print(f"读取文件：{latest_file}")
    df = pd.read_csv(latest_file)
    return df

def df_to_markdown_table(df: pd.DataFrame, columns: list):
    """挑选指定列，转markdown表格"""
    sub_df = df[columns].copy()
    # 保留4位小数美化
    for col in sub_df.columns:
        if sub_df[col].dtype in ["float64", "float32"]:
            sub_df[col] = sub_df[col].round(4)
    md = sub_df.to_markdown(index=False)
    return md

if __name__ == "__main__":
    df = load_latest_summary()
    # 你想要展示的核心指标，按需增删
    show_cols = [
        "stock_code",
        "model_type",
        "cluster_id",
        "cv_mean_acc",
        "test_acc",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "trade_cnt",
        "fail_reason"
    ]
    # 过滤掉不存在的列，防止报错
    show_cols = [c for c in show_cols if c in df.columns]
    md_table = df_to_markdown_table(df, show_cols)

    save_md_path = os.path.join(IMAGES_DIR, "batch_compare_table.md")
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(save_md_path, "w", encoding="utf-8") as f:
        f.write(f"# 批量多股票回测结果对比\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md_table)
    print(f"✅ Markdown表格已保存到 {save_md_path}")
    print("="*80)
    print(md_table)
