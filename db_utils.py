"""
db_utils.py
MySQL数据库工具类，用于量化项目读写数据
彻底修复np.nan/numpy类型批量插入回滚；连接泄露；空闲超时gone‑away问题
"""
import os
import logging
import pymysql
import pandas as pd
from dotenv import load_dotenv
from pymysql.err import OperationalError, MySQLError

load_dotenv()
# 模块独立日志器，不和外部logger耦合
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)


class StockDB:
    def __init__(self):
        """初始化数据库配置，从.env读取连接参数，不建立实际连接"""
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.db_name = os.getenv("DB_NAME")
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        建立数据库连接，使用普通cursor，适配executemany批量插入
        :return: bool True连接成功，False连接失败
        """
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=int(self.port),
                user=self.user,
                password=self.password,
                database=self.db_name,
                charset="utf8mb4"
            )
            self.cursor = self.conn.cursor()
            logger.info("✅ MySQL连接成功")
            return True
        except OperationalError as e:
            logger.error(f"❌ MySQL连接失败: {e}")
            return False

    def close(self):
        """安全关闭游标与数据库连接，避免数据库连接泄露"""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("🔌 数据库连接已关闭")

    @staticmethod
    def _convert_datetime_to_str(df: pd.DataFrame):
        """
        数据预处理：datetime转字符串，np.nan/pd.NA转换为MySQL可识别的None
        :param df: 待写入DataFrame
        :return: pd.DataFrame 处理完成副本
        """
        df = df.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        # 真正处理：np.nan、pd.NA、numpy数值 → Python原生None/原生类型
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) else (x.item() if hasattr(x, "item") else x)
            )
        return df

    def insert_stock_basic(self, df: pd.DataFrame):
        """
        批量写入股票基础信息表，主键冲突执行ON DUPLICATE KEY UPDATE更新
        :param df: 股票基础信息DataFrame
        :raises RuntimeError: mysql写入异常，已执行回滚
        """
        if df.empty:
            logger.info("ℹ️ stock_basic 数据集为空，跳过写入")
            return
        df = self._convert_datetime_to_str(df)
        sql = """
        INSERT INTO stock_basic (ts_code, symbol, name, industry, list_date)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            symbol=VALUES(symbol), name=VALUES(name), industry=VALUES(industry), list_date=VALUES(list_date)
        """
        data_list = df[["ts_code", "symbol", "name", "industry", "list_date"]].values.tolist()
        try:
            self.cursor.executemany(sql, data_list)
            self.conn.commit()
            logger.info(f"📥 写入 {len(data_list)} 条股票基础信息")
        except MySQLError as e:
            self.conn.rollback()
            err_msg = f"❌写入stock_basic异常，已回滚: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def insert_stock_daily(self, df: pd.DataFrame):
        """
        批量写入日线行情表，主键冲突执行更新
        :param df: 日线行情DataFrame
        :raises RuntimeError: mysql写入异常，已执行回滚
        """
        if df.empty:
            logger.info("ℹ️ stock_daily 数据集为空，跳过写入")
            return
        df = self._convert_datetime_to_str(df)
        sql = """
        INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, vol)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open), high=VALUES(high), low=VALUES(low),
            close=VALUES(close), vol=VALUES(vol)
        """
        data_list = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol"]].values.tolist()
        try:
            self.cursor.executemany(sql, data_list)
            self.conn.commit()
            logger.info(f"📥 写入 {len(data_list)} 条日线行情")
        except MySQLError as e:
            self.conn.rollback()
            err_msg = f"❌写入stock_daily异常，已回滚: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def insert_stock_factor(self, df: pd.DataFrame):
        """
        批量写入技术因子表；自动把np.nan转为None，防止入库报错回滚
        :param df: 技术因子DataFrame
        :raises RuntimeError: mysql写入异常，已执行回滚
        """
        if df.empty:
            logger.info("📢 stock_factor 数据集为空，跳过写入")
            return
        df = self._convert_datetime_to_str(df)
        sql = """
        INSERT INTO stock_factor (
            ts_code, trade_date, ma5, ma20, close_ma5_ratio, close_ma20_ratio,
            rsi14, macd_dif, macd_dea, macd_hist, vol_change, vol_ratio,
            ret_1d, ret_5d, ret_10d, volatility, label
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            ma5=VALUES(ma5), ma20=VALUES(ma20),
            close_ma5_ratio=VALUES(close_ma5_ratio), close_ma20_ratio=VALUES(close_ma20_ratio),
            rsi14=VALUES(rsi14), macd_dif=VALUES(macd_dif), macd_dea=VALUES(macd_dea), macd_hist=VALUES(macd_hist),
            vol_change=VALUES(vol_change), vol_ratio=VALUES(vol_ratio),
            ret_1d=VALUES(ret_1d), ret_5d=VALUES(ret_5d), ret_10d=VALUES(ret_10d),
            volatility=VALUES(volatility), label=VALUES(label)
        """
        cols = [
            "ts_code", "trade_date", "ma5", "ma20", "close_ma5_ratio", "close_ma20_ratio",
            "rsi14", "macd_dif", "macd_dea", "macd_hist", "vol_change", "vol_ratio",
            "ret_1d", "ret_5d", "ret_10d", "volatility", "label"
        ]
        data_list = []
        for _, row in df[cols].iterrows():
            row_data = []
            for x in row.tolist():
                if pd.isna(x):
                    row_data.append(None)
                else:
                    row_data.append(x)
            data_list.append(row_data)
        try:
            self.cursor.executemany(sql, data_list)
            self.conn.commit()
            logger.info(f"📥 写入 {len(data_list)} 条因子数据")
        except MySQLError as e:
            self.conn.rollback()
            err_msg = f"❌写入stock_factor异常，已回滚: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def insert_backtest_trade(self, df: pd.DataFrame):
        """
        批量写入回测交易记录表
        :param df: 交易记录DataFrame
        :raises RuntimeError: mysql写入异常，已执行回滚
        """
        if df.empty:
            logger.info("ℹ️ backtest_trade 交易记录数据集为空，跳过写入")
            return
        df = self._convert_datetime_to_str(df)
        sql = """
        INSERT INTO backtest_trade (strategy_name, ts_code, trade_date, direction, price, volume, commission, tax)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            direction=VALUES(direction), price=VALUES(price), volume=VALUES(volume),
            commission=VALUES(commission), tax=VALUES(tax)
        """
        data_list = df[["strategy_name", "ts_code", "trade_date", "direction", "price", "volume", "commission", "tax"]].values.tolist()
        try:
            self.cursor.executemany(sql, data_list)
            self.conn.commit()
            logger.info(f"📥 写入 {len(data_list)} 条交易记录")
        except MySQLError as e:
            self.conn.rollback()
            err_msg = f"❌写入backtest_trade异常，已回滚: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def insert_backtest_summary(self, data_dict: dict):
        """
        写入回测汇总指标单条记录
        :param data_dict: 回测汇总指标字典
        :raises RuntimeError: mysql写入异常，已执行回滚
        """
        sql = """
        INSERT INTO backtest_summary
        (strategy_name, start_date, end_date, initial_capital, final_capital, total_return, sharpe_ratio, max_drawdown, win_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            final_capital=VALUES(final_capital), total_return=VALUES(total_return),
            sharpe_ratio=VALUES(sharpe_ratio), max_drawdown=VALUES(max_drawdown), win_rate=VALUES(win_rate)
        """
        try:
            self.cursor.execute(sql, (
                data_dict["strategy_name"],
                data_dict["start_date"],
                data_dict["end_date"],
                data_dict["initial_capital"],
                data_dict["final_capital"],
                data_dict["total_return"],
                data_dict["sharpe_ratio"],
                data_dict["max_drawdown"],
                data_dict["win_rate"]
            ))
            self.conn.commit()
            logger.info("📥 写入回测汇总指标")
        except MySQLError as e:
            self.conn.rollback()
            err_msg = f"❌写入backtest_summary异常，已回滚: {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from e

    def query_daily_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        根据股票代码+时间范围，查询日线行情
        :param ts_code: 股票ts代码
        :param start_date: 开始日期 YYYY‑MM‑DD
        :param end_date: 结束日期 YYYY‑MM‑DD
        :return: pd.DataFrame 查询结果，无数据返回空DataFrame
        """
        sql = """
        SELECT trade_date, open, high, low, close, vol
        FROM stock_daily
        WHERE ts_code = %s AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date ASC
        """
        cols = ["trade_date", "open", "high", "low", "close", "vol"]
        try:
            self.cursor.execute(sql, (ts_code, start_date, end_date))
            res = self.cursor.fetchall()
            df = pd.DataFrame(res, columns=cols)
            if df.empty:
                logger.info(f"ℹ️ query_daily_data: {ts_code} 在[{start_date}~{end_date}]未查询到行情数据")
            return df
        except MySQLError as e:
            logger.error(f"❌查询日线数据异常:{e}")
            return pd.DataFrame(columns=cols)


if __name__ == "__main__":
    db = StockDB()
    if db.connect():
        logger.info("数据库工具测试成功！")
    db.close()
