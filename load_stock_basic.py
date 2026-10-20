# load_stock_basic.py 修复版本
import pandas as pd
from db_utils import StockDB

if __name__ == "__main__":
    target_code = "600519.SH"
    db = StockDB()
    if not db.connect():
        print("数据库连接失败，程序退出")
        exit(1)

    print(f"手动构造 {target_code} 股票基础信息，绕过Tushare stock_basic接口限流")
    df_basic = pd.DataFrame([
        {
            "ts_code": "600519.SH",
            "symbol": "600519",
            "name": "贵州茅台",
            "industry": "白酒",
            "list_date": "2001-08-27"
        }
    ])
    print("\n待写入数据：")
    print(df_basic)

    try:
        db.insert_stock_basic(df_basic)
        print("✅ insert + commit已经执行完成")
    except Exception as e:
        print(f"stock_basic写入异常 {e}")
    finally:
        # 写入完成立刻关闭，结束会话，强制把commit的数据刷入MySQL磁盘
        db.close()

    # ✅校验：必须【新建全新数据库连接】，不能复用刚才写入的连接！
    print("\n====新建连接校验数据库真实磁盘数据====")
    db_check = StockDB()
    check_df = pd.DataFrame()
    if db_check.connect():
        db_check.cursor.execute("SELECT * FROM stock_basic;")
        res = db_check.cursor.fetchall()
        cols = ["ts_code", "symbol", "name", "industry", "list_date"]
        check_df = pd.DataFrame(res, columns=cols)
        print("stock_basic磁盘真实数据：")
        print(check_df)
    db_check.close()

    print("\n程序执行完毕")
