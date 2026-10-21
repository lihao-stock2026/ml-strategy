# load_stock_basic.py 修复&优化版本
"""
初始化stock_basic表基础股票信息
绕过tushare stock_basic高频限流接口，手动写入标的基础信息
写入完成后新建数据库连接校验磁盘真实数据，规避MVCC快照看不到已提交数据问题
"""
import logging
import pandas as pd
from db_utils import StockDB

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

logger = setup_logger()

def insert_target_stock_basic():
    """
    手动构造单只股票基础信息写入stock_basic表，完成后新建连接校验数据持久化结果
    """
    target_code = "600519.SH"
    db = StockDB()
    if not db.connect():
        logger.error("数据库连接失败，程序退出")
        return

    logger.info(f"手动构造 {target_code} 股票基础信息，绕过Tushare stock_basic接口限流")
    df_basic = pd.DataFrame([
        {
            "ts_code": "600519.SH",
            "symbol": "600519",
            "name": "贵州茅台",
            "industry": "白酒",
            "list_date": "2001-08-27"
        }
    ])
    logger.info(f"待写入数据：\n{df_basic}")
    try:
        db.insert_stock_basic(df_basic)
        logger.info("✅ insert + commit已经执行完成")
    except Exception as e:
        logger.error(f"stock_basic写入异常 {e}")
    finally:
        # 写入完成立刻关闭，结束会话，强制把commit的数据刷入MySQL磁盘
        db.close()

    # ✅校验：必须【新建全新数据库连接】，不能复用刚才写入的连接！
    logger.info("\n====新建连接校验数据库真实磁盘数据====")
    db_check = StockDB()
    check_df = pd.DataFrame()
    if db_check.connect():
        db_check.cursor.execute("SELECT * FROM stock_basic;")
        res = db_check.cursor.fetchall()
        cols = ["ts_code", "symbol", "name", "industry", "list_date"]
        check_df = pd.DataFrame(res, columns=cols)
        logger.info(f"stock_basic磁盘真实数据：\n{check_df}")
    db_check.close()
    logger.info("\n程序执行完毕")

if __name__ == "__main__":
    insert_target_stock_basic()
