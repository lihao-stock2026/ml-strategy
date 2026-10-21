"""
网格搜索调参实验：遍历max_depth / learning_rate，观察过拟合现象
复用 batch_gbdt_event_strategy 的完整流水线
输出：grid_search_result.csv，记录每组参数交叉验证、测试集、回测指标
"""
import os
import pandas as pd
from batch_gbdt_event_strategy import (
    run_single_stock,
    DEFAULT_PARAMS
)

# ======================调参网格定义======================
# 树深度：越大模型容量越高，容易过拟合
max_depth_list = [3,5,7,9]
# 学习率：大学习率收敛快，容易震荡过拟合；小学习率更稳但迭代慢
lr_list = [0.02, 0.05, 0.10]

# 回测公共配置
STOCK_CODE = "600519.SH"
START = "20200101"
END = "20241231"
INIT_CAPITAL = 100000000
MODEL_TYPE = "lgb"
PROB_THRESHOLD = 0.35
SELL_THRESHOLD = 0.45
DATA_SOURCE = "akshare"
REFRESH_CACHE = False
NO_PLOT = True  # 网格搜索关闭绘图，避免生成大量图片
TOKEN = ""
CLUSTER_ID = -1
# ======================================================

def main():
    out_rows = []
    base_params = DEFAULT_PARAMS.copy()

    for max_depth in max_depth_list:
        for lr in lr_list:
            print(f"\n==========实验 max_depth={max_depth}, lr={lr} ==========")
            cur_params = base_params.copy()
            cur_params["max_depth"] = max_depth
            cur_params["learning_rate"] = lr

            try:
                res_dict = run_single_stock(
                    token=TOKEN,
                    stock_code=STOCK_CODE,
                    start=START,
                    end=END,
                    initial_capital=INIT_CAPITAL,
                    no_plot=NO_PLOT,
                    model_type=MODEL_TYPE,
                    model_params=cur_params,
                    prob_threshold=PROB_THRESHOLD,
                    sell_threshold=SELL_THRESHOLD,
                    cluster_id=CLUSTER_ID,
                    data_source=DATA_SOURCE,
                    refresh_cache=REFRESH_CACHE
                )
                row = {
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "cv_mean_acc": res_dict["cv_mean_acc"],
                    "test_acc": res_dict["test_acc"],
                    "total_return": res_dict["total_return"],
                    "sharpe": res_dict["sharpe"],
                    "max_drawdown": res_dict["max_drawdown"],
                    "trade_cnt": res_dict["trade_cnt"]
                }
                out_rows.append(row)
                print(row)
            except Exception as e:
                print(f"该组参数执行失败：{str(e)}")

    df_out = pd.DataFrame(out_rows)
    save_path = os.path.join("./output", "grid_search_result.csv")
    df_out.to_csv(save_path, index=False, encoding="utf‑8‑sig")
    print(f"\n✅网格搜索结果保存到 {save_path}")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    main()
