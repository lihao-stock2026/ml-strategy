"""
事件驱动双均线回测框架 v1.0.6【完善最终版】
金叉买入，死叉卖出；模拟A股现金、持仓、100股最小交易单位、手续费、印花税、滑点
交互式输入股票代码与初始资金，也支持命令行传参
v1.0.6更新：卖出订单增加价格NaN/0防护，规避异常行情触发异常
⚠️风险免责：本代码仅用于学习演示，不构成任何投资建议。历史回测结果不能代表未来实际收益。
【运行示例】
python event_backtest.py --code 600519.SH --start 20200101 --end 20241231
"""
import os
import glob
import logging
import argparse
import time
import shlex
import random
from datetime import datetime
from dotenv import load_dotenv
import tushare as ts
import pandas as pd
import numpy as np

__version__ = "1.0.6"
RISK_FREE_RATE = 0.02
COMMISSION_RATE = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.0005
MAX_RETRY = 3
ADJ_MODE = 'hfq'
API_MIN_INTERVAL = 0.4
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_shell_join(args_list):
    """
    跨平台安全拼接shell命令参数
    :param args_list: list[str] 命令参数列表
    :return: str 拼接完成命令字符串
    """
    try:
        return shlex.join(args_list)
    except AttributeError:
        return " ".join(shlex.quote(x) for x in args_list)


try:
    import importlib.metadata
except ImportError:
    importlib = None

try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv()
np.random.seed(42)
random.seed(42)

if importlib is not None:
    try:
        tushare_ver = importlib.metadata.version("tushare")
        ver_major, ver_minor, ver_patch = map(int, tushare_ver.split(".")[:3])
        if (ver_major, ver_minor, ver_patch) < (1, 2, 89):
            print(f"⚠️警告：检测到tushare版本 {tushare_ver} 过低，建议执行：pip install tushare --upgrade")
    except Exception:
        print("⚠️警告：无法获取tushare版本，请确认tushare>=1.2.89已正确安装")
else:
    print("⚠️当前Python版本较低，跳过tushare版本检测，请手动确认tushare>=1.2.89")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def clear_all_cache():
    """清除cache_*.csv行情缓存文件"""
    cache_files = glob.glob("cache_*.csv")
    for f in cache_files:
        os.remove(f)
        logging.info(f"✅已清理行情缓存，共删除 {len(cache_files)} 个")


def clean_all_output():
    """清理事件驱动回测输出产物，保留行情cache缓存"""
    patterns = [
        os.path.join(OUTPUT_DIR, "event_result_*.png"),
        os.path.join(OUTPUT_DIR, "event_backtest_report_*.csv"),
        os.path.join(OUTPUT_DIR, "event_trade_records_*.csv"),
        os.path.join(OUTPUT_DIR, "event_summary_*.txt"),
    ]
    cnt = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            cnt += 1
    print(f"🧹已清理事件驱动输出报告/图片，共删除 {cnt} 个；行情cache缓存已保留")


def setup_logger():
    """
    初始化日志，同时输出控制台与日志文件event_backtest.log
    :return: logging.Logger 日志实例
    """
    log_path = os.path.join(OUTPUT_DIR, "event_backtest.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def ts_pro_bar_with_retry(code, adj, start_date, end_date, max_retry=MAX_RETRY):
    """
    tushare pro_bar接口带重试，处理限流、积分异常
    :param code: str ts股票代码
    :param adj: str 复权模式 hfq/qfq
    :param start_date: str YYYYMMDD
    :param end_date: str YYYYMMDD
    :param max_retry: int 最大重试次数
    :return: pd.DataFrame|None 行情数据，失败返回None
    """
    for attempt in range(max_retry):
        try:
            time.sleep(API_MIN_INTERVAL)
            df = ts.pro_bar(ts_code=code, adj=adj, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                logger.warning(f"Tushare返回空数据，code={code}")
                return None
            return df
        except Exception as e:
            err_msg = str(e)
            if "积分" in err_msg or "permission" in err_msg.lower():
                raise RuntimeError("❌Tushare积分不足，无法调取数据，请提升账号积分！")
            if ("访问频率" in err_msg.lower()) or ("adj_factor" in err_msg):
                if attempt + 1 < max_retry:
                    sleep_sec = 60 + random.uniform(2, 8)
                    logger.warning(f"Tushare adj_factor限流，随机休眠 {sleep_sec:.1f}s 后重试 {attempt+1}/{max_retry}")
                    time.sleep(sleep_sec)
            else:
                logger.warning(f"Tushare请求失败，重试 {attempt+1}/{max_retry}, err:{err_msg}")
    raise RuntimeError("Tushare接口多次请求失败，请检查网络、token、账号积分")


class Event:
    """事件基类，所有事件继承此类"""
    pass


class MarketEvent(Event):
    """
    行情K线事件，保存单根K线信息
    :param date: datetime 交易日期
    :param open_: float 开盘价
    :param high: float 最高价
    :param low: float 最低价
    :param close: float 收盘价
    :param volume: float 成交量
    """
    def __init__(self, date, open_, high, low, close, volume):
        self.type = "MARKET"
        self.date = date
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class SignalEvent(Event):
    """
    策略信号事件，策略产生买卖信号
    :param date: datetime 信号产生日期
    :param signal: int 1买入，-1卖出
    """
    def __init__(self, date, signal):
        self.type = "SIGNAL"
        self.date = date
        self.signal = signal


class OrderEvent(Event):
    """
    订单事件，待执行的委托订单
    :param date: datetime 订单日期
    :param direction: str BUY / SELL
    :param quantity: int 委托股数
    """
    def __init__(self, date, direction, quantity):
        self.type = "ORDER"
        self.date = date
        self.direction = direction
        self.quantity = quantity


class FillEvent(Event):
    """
    成交事件，订单撮合完成之后生成
    :param fill_date: datetime 成交日期
    :param direction: str BUY / SELL
    :param quantity: int 成交股数
    :param price: float 成交价格
    :param commission: float 手续费率
    :param stamp_tax: float 印花税率
    """
    def __init__(self, fill_date, direction, quantity, price, commission, stamp_tax):
        self.type = "FILL"
        self.fill_date = fill_date
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.stamp_tax = stamp_tax


class DataHandler:
    """
    行情迭代器，逐根K线输出MarketEvent
    :param df: pd.DataFrame 已经排好序的日线DataFrame
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.idx = 0

    def next_bar(self):
        """
        获取下一根K线行情事件
        :return: MarketEvent|None 无更多数据返回None
        """
        if self.idx >= len(self.df):
            return None
        row = self.df.iloc[self.idx]
        bar = MarketEvent(
            date=row["trade_date"],
            open_=row["open"], high=row["high"],
            low=row["low"], close=row["close"],
            volume=row["vol"]
        )
        self.idx += 1
        return bar


class MAStrategy:
    """
    双均线金叉死叉策略，根据短、长期均线生成SignalEvent
    :param short_window: int 短期均线周期
    :param long_window: int 长期均线周期
    """
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window
        self.price_series = pd.Series(dtype=float)
        self.buy_points = []
        self.sell_points = []

    def calculate_signal(self, bar: MarketEvent):
        """
        接收单根行情K线，计算交易信号
        :param bar: MarketEvent
        :return: SignalEvent|None 有信号返回事件，无信号返回None
        """
        self.price_series.loc[bar.date] = bar.close
        if len(self.price_series) < self.long_window:
            return None
        ma_short = self.price_series.rolling(window=self.short_window).mean()
        ma_long = self.price_series.rolling(window=self.long_window).mean()
        ma_short_now = ma_short.iloc[-1]
        ma_long_now = ma_long.iloc[-1]
        ma_short_prev = ma_short.iloc[-2]
        ma_long_prev = ma_long.iloc[-2]
        if ma_short_prev <= ma_long_prev and ma_short_now > ma_long_now:
            logger.info(f"✅检测到金叉,日期:{bar.date.date()}, MA{self.short_window}_prev:{ma_short_prev:.2f}, MA{self.long_window}_prev:{ma_long_prev:.2f}")
            self.buy_points.append(bar.date)
            return SignalEvent(bar.date, signal=1)
        elif ma_short_prev >= ma_long_prev and ma_short_now < ma_long_now:
            logger.info(f"❌检测到死叉,日期:{bar.date.date()}, MA{self.short_window}_prev:{ma_short_prev:.2f}, MA{self.long_window}_prev:{ma_long_prev:.2f}")
            self.sell_points.append(bar.date)
            return SignalEvent(bar.date, signal=-1)
        return None


class Portfolio:
    """
    投资组合账户类：管理现金、持仓、每日净值曲线、成交记录，模拟A股最小1手交易
    :param initial_capital: float 初始账户资金
    """
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.current_price = 0.0
        self.last_buy_price = 0.0
        self.equity_curve = []
        self.trade_records = []

    def update_market(self, bar: MarketEvent):
        """
        接收最新行情，更新账户市值，写入净值曲线
        :param bar: MarketEvent
        """
        self.current_price = bar.close
        total_value = self.cash + self.position * self.current_price
        self.equity_curve.append({
            "date": bar.date,
            "cash": self.cash,
            "position": self.position,
            "equity": total_value
        })

    def generate_order(self, signal_event: SignalEvent, estimate_open_price: float):
        """
        根据信号事件生成委托OrderEvent，处理100股最小交易单位、资金不足判断
        :param signal_event: SignalEvent 买卖信号
        :param estimate_open_price: float 预估开盘成交价格
        :return: OrderEvent|None 无法下单返回None
        """
        if np.isnan(estimate_open_price) or estimate_open_price <= 1e-6:
            logger.warning(f"⚠️价格异常 estimate_open_price={estimate_open_price},跳过生成订单")
            return None
        if signal_event.signal == 1 and self.position == 0:
            per_lot_cost = estimate_open_price * 100 * (1 + COMMISSION_RATE + SLIPPAGE)
            max_lots = int(self.cash / per_lot_cost)
            quantity = max_lots * 100
            if quantity >= 100:
                logger.info(f"生成买入订单，拟下单数量:{quantity}股 (股价{estimate_open_price:.2f})")
                return OrderEvent(signal_event.date, "BUY", quantity)
            else:
                logger.warning(
                    f"⚠️ 资金不足，无法买入！当前股价≈{estimate_open_price:.2f}元，"
                    f"买1手(100股)至少需要≈{per_lot_cost:,.2f}元，"
                    f"但账户仅剩现金{self.cash:,.2f}元，缺口≈{per_lot_cost - self.cash:,.2f}元"
                )
        elif signal_event.signal == -1 and self.position > 0:
            if np.isnan(estimate_open_price) or estimate_open_price <= 1e-6:
                logger.warning(f"⚠️卖出价格异常 estimate_open_price={estimate_open_price}，跳过生成卖出订单")
                return None
            logger.info(f"生成卖出订单，拟卖出全部持仓:{self.position}股")
            return OrderEvent(signal_event.date, "SELL", self.position)
        return None

    def update_fill(self, fill_event: FillEvent):
        """
        接收成交FillEvent，更新现金、持仓、成交记录
        :param fill_event: FillEvent 撮合完成的成交事件
        """
        cost = fill_event.quantity * fill_event.price
        commission = cost * fill_event.commission
        stamp_tax_amount = cost * fill_event.stamp_tax
        if fill_event.direction == "BUY":
            self.cash -= (cost + commission)
            self.position += fill_event.quantity
            self.last_buy_price = fill_event.price
            logger.info(
                f"【成交-买入】日期:{fill_event.fill_date.date()}, "
                f"价格:{fill_event.price:.2f}, 股数:{fill_event.quantity}, "
                f"手续费:{commission:.2f}, 剩余现金:{self.cash:,.2f}"
            )
            self.trade_records.append({
                "trade_date": fill_event.fill_date,
                "direction":"BUY",
                "price":fill_event.price,
                "quantity":fill_event.quantity,
                "commission":commission,
                "stamp_tax":0.0,
                "trade_profit":0.0
            })
        else:
            profit = fill_event.price * fill_event.quantity - self.last_buy_price * fill_event.quantity - commission - stamp_tax_amount
            self.cash += (cost - commission - stamp_tax_amount)
            self.position -= fill_event.quantity
            logger.info(
                f"【成交-卖出】日期:{fill_event.fill_date.date()}, "
                f"价格:{fill_event.price:.2f}, 股数:{fill_event.quantity}, "
                f"手续费:{commission:.2f}, 印花税:{stamp_tax_amount:.2f}, "
                f"剩余现金:{self.cash:,.2f}"
            )
            self.trade_records.append({
                "trade_date": fill_event.fill_date,
                "direction":"SELL",
                "price":fill_event.price,
                "quantity":fill_event.quantity,
                "commission":commission,
                "stamp_tax":stamp_tax_amount,
                "trade_profit": profit
            })


class ExecutionHandler:
    """
    订单执行器：模拟滑点，把OrderEvent撮合转换为FillEvent成交事件
    :param slippage: float 滑点系数
    """
    def __init__(self, slippage: float):
        self.slippage = slippage

    def execute(self, order: OrderEvent, next_bar: MarketEvent) -> FillEvent:
        """
        执行委托订单，应用滑点规则返回成交事件
        :param order: 待执行订单OrderEvent
        :param next_bar: 下一根K线，使用开盘价撮合
        :return: FillEvent 成交事件
        """
        if np.isnan(next_bar.open_) or next_bar.open_ <= 1e-6:
            raise RuntimeError(f"next_bar.open_价格异常 {next_bar.open_}")
        if order.direction == "BUY":
            exec_price = next_bar.open_ * (1 + self.slippage)
        else:
            exec_price = next_bar.open_ * (1 - self.slippage)
        tax = STAMP_TAX if order.direction == "SELL" else 0.0
        return FillEvent(
            fill_date=next_bar.date,
            direction=order.direction,
            quantity=order.quantity,
            price=exec_price,
            commission=COMMISSION_RATE,
            stamp_tax=tax
        )


class BacktestEngine:
    """
    事件驱动回测引擎，串联数据、策略、账户、执行器完整回测流程
    :param df: pd.DataFrame 已清洗日线数据
    :param initial_capital: float 初始资金
    :param short_window: int 短期均线
    :param long_window: int 长期均线
    """
    def __init__(self, df: pd.DataFrame, initial_capital=100000.0, short_window=10, long_window=30):
        self.df = df
        self.initial_capital = initial_capital
        self.short_window = short_window
        self.long_window = long_window

    def run(self):
        """
        启动回测主循环，逐事件驱动运行
        :return: equity_df,buy_points,sell_points,trade_records
        """
        data_handler = DataHandler(self.df)
        strategy = MAStrategy(self.short_window, self.long_window)
        portfolio = Portfolio(self.initial_capital)
        executor = ExecutionHandler(SLIPPAGE)
        pending_order = None
        while True:
            bar = data_handler.next_bar()
            if bar is None:
                break
            if pending_order is not None:
                fill = executor.execute(pending_order, bar)
                portfolio.update_fill(fill)
                pending_order = None
            portfolio.update_market(bar)
            sig = strategy.calculate_signal(bar)
            if sig is not None:
                order = portfolio.generate_order(sig, bar.open_)
                if order is not None:
                    pending_order = order
        if len(portfolio.trade_records) == 0:
            logger.warning("⚠️本次回测没有产生任何一笔实际成交，请检查均线参数或者行情数据")
        return pd.DataFrame(portfolio.equity_curve), strategy.buy_points, strategy.sell_points, portfolio.trade_records


def calc_metrics(equity_df: pd.DataFrame, rf: float):
    """
    回测指标计算：总收益、年化收益、夏普比率、最大回撤
    :param equity_df: pd.DataFrame 账户净值曲线，包含date、equity字段
    :param rf: float 无风险年利率
    :return: total_return, annual_return, sharpe, max_dd, net_series
    """
    net = equity_df["equity"] / equity_df["equity"].iloc[0]
    net = net.dropna()
    if len(net) <= 1:
        logger.warning("⚠️净值序列有效数据过少，指标全部置空")
        return np.nan, np.nan, np.nan, np.nan, net
    ret = net.pct_change().dropna()
    total_return = net.iloc[-1] - 1
    years = len(ret) / 252
    if years <= 0:
        logger.warning("⚠️交易年数计算为0，指标全部置空")
        return np.nan, np.nan, np.nan, np.nan, net
    annual_return = (1 + total_return) ** (1 / years) - 1
    sharpe = np.nan
    if ret.std() > 1e-8:
        sharpe = (ret.mean() - rf / 252) / ret.std() * np.sqrt(252)
    cummax = net.cummax()
    max_dd = ((net - cummax) / cummax).min()
    return total_return, annual_return, sharpe, max_dd, net


def auto_calc_capital(df: pd.DataFrame) -> float:
    """
    根据行情最高价自动估算建议初始资金
    :param df: pd.DataFrame 日线行情
    :return: float 建议资金
    """
    max_price = df["close"].max()
    suggested_capital = max_price * 1000 * 1.15
    suggested_capital = np.ceil(suggested_capital / 10000) * 10000
    return float(suggested_capital)


def plot_chart(equity_df: pd.DataFrame, net_series, bh_df, stock_code, buy_points, sell_points, short_window, long_window):
    """
    绘制回测净值对比图：策略净值 vs 买入持有，标记买卖点
    :param equity_df: 账户净值DataFrame
    :param net_series: 策略归一化净值
    :param bh_df: 买入持有数据
    :param stock_code: str 股票代码
    :param buy_points: list 买入信号日期列表
    :param sell_points: list 卖出信号日期列表
    :param short_window: int 短期均线
    :param long_window: int 长期均线
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(equity_df["date"], net_series, label="事件驱动双均线策略", linewidth=1.5, color="#1f77b4")
    axes[0].plot(bh_df["trade_date"], bh_df["bh_net"], label="买入持有", alpha=0.7, color="#ff7f0e")
    for d in buy_points:
        row = equity_df[equity_df["date"] == d]
        if not row.empty:
            val = row["equity"].values[0] / equity_df["equity"].iloc[0]
            axes[0].scatter(d, val, marker="^", color="green", s=80, zorder=5)
    for d in sell_points:
        row = equity_df[equity_df["date"] == d]
        if not row.empty:
            val = row["equity"].values[0] / equity_df["equity"].iloc[0]
            axes[0].scatter(d, val, marker="v", color="red", s=80, zorder=5)
    axes[0].set_title(f"{stock_code}：事件驱动MA{short_window}-MA{long_window}回测净值")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(equity_df["date"], equity_df["cash"], label="账户现金余额", color="orange")
    axes[1].set_title("账户现金变化曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f"event_result_{stock_code.replace('.','_')}_{timestamp}.png")
    try:
        plt.savefig(filename, dpi=150)
        logger.info(f"回测净值图表保存至：{filename}")
    except Exception as e:
        logger.warning(f"⚠️净值图表保存失败，跳过绘图，err:{str(e)}")
    plt.close(fig)


def fetch_data(token: str, code: str, start: str, end: str) -> pd.DataFrame:
    """
    获取股票日线行情，优先读取本地缓存，无缓存调用tushare
    :param token: tushare token字符串
    :param code: ts股票代码
    :param start: YYYYMMDD
    :param end: YYYYMMDD
    :return: pd.DataFrame 清洗、排序之后日线数据
    """
    if not token:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            token = input('粘贴你的tushare token：').strip()
            if not token:
                raise RuntimeError("token不能为空，请配置.env文件或者输入有效的Tushare token")
    ts.set_token(token)
    cache_file = f"cache_{code.replace('.','_')}_{ADJ_MODE}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        logger.info(f"读取本地行情缓存：{cache_file}")
        try:
            df_raw = pd.read_csv(cache_file, encoding="utf-8-sig").reset_index(drop=True)
        except pd.errors.EmptyDataError:
            logger.warning(f"⚠️缓存文件损坏 {cache_file}，删除损坏缓存，重新拉取")
            os.remove(cache_file)
            df_raw = None
        if df_raw is not None:
            df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
    else:
        df_raw = None
    if df_raw is None:
        logger.info("本地缓存不存在，从Tushare拉取后复权数据...")
        df_raw = ts_pro_bar_with_retry(code, adj=ADJ_MODE, start_date=start, end_date=end)
        if df_raw is None or df_raw.empty:
            raise RuntimeError(f"股票 {code} 获取到空数据集，请检查参数")
        required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        miss_cols = [c for c in required_cols if c not in df_raw.columns]
        if miss_cols:
            raise RuntimeError(f"Tushare返回数据缺少必要字段：{','.join(miss_cols)}，接口返回异常")
        valid_ratio = (~df_raw[["open","high","low","close","vol"]].isna()).all(axis=1).mean()
        if valid_ratio < 0.3:
            raise RuntimeError(f"行情数据大量缺失，有效行占比仅{valid_ratio:.1%}，无法继续回测")
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
        df_raw_save = df_raw.copy()
        df_raw_save["trade_date"] = df_raw_save["trade_date"].dt.strftime("%Y%m%d")
        df_raw_save.to_csv(cache_file, index=False, encoding="utf-8-sig")
        logger.info(f"行情已写入缓存文件：{cache_file}")
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    df = df.sort_values("trade_date").reset_index(drop=True)
    logger.info(f"成功加载 {code} 日线数据，共 {len(df)} 条")
    return df


def main():
    """程序入口，解析命令行参数，执行双均线事件驱动回测"""
    parser = argparse.ArgumentParser(
        description="事件驱动双均线回测框架，本代码仅学习演示，不构成投资建议"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--code", type=str, default=None, help="股票ts代码，不传则控制台手动输入")
    parser.add_argument("--start", type=str, default="20190601", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--token", type=str, default="", help="Tushare api令牌")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金(元)，不传则进入控制台交互式输入")
    parser.add_argument("--short", type=int, default=10, help="短期均线周期，默认10")
    parser.add_argument("--long", type=int, default=30, help="长期均线周期，默认30")
    parser.add_argument("--clear_cache", action="store_true", help="清除cache_*.csv行情缓存")
    parser.add_argument("--clean_output", action="store_true", help="清理事件驱动输出图片/报告日志，保留缓存")
    args = parser.parse_args()
    if args.clean_output:
        clean_all_output()
    if args.clear_cache:
        clear_all_cache()
    start_dt = pd.to_datetime(args.start, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end, format="%Y%m%d")
    delta_years = (end_dt - start_dt).days / 365.0
    if delta_years > 8:
        logger.warning(f"⚠️回测时间跨度 {delta_years:.1f} 年，区间较大，容易触发接口限流与内存不足，建议适当缩短区间")
    for arg_name, val in [("start", args.start), ("end", args.end)]:
        if not (val.isdigit() and len(val) == 8):
            raise RuntimeError(f"参数 --{arg_name}={val} 格式错误！必须为8位数字 YYYYMMDD，示例：20200101")
    if args.short < 1 or args.long <1:
        raise RuntimeError("均线周期不能小于1")
    if args.short >= args.long:
        logger.warning("提示：短期均线大于等于长期均线，策略逻辑反常，请确认参数")
    logger.info("========== 程序启动参数 ==========")
    logger.info(f"脚本版本: {__version__}")
    for k, v in vars(args).items():
        logger.info(f"{k:20s}: {v}")
    logger.info("==================================")
    shell_cmd_parts = ["python", "event_backtest.py"]
    if args.code:
        shell_cmd_parts.extend(["--code", args.code])
    shell_cmd_parts.extend(["--start", args.start, "--end", args.end])
    if args.token:
        shell_parts.extend(["--token", args.token])
    shell_cmd_parts.extend(["--short", str(args.short), "--long", str(args.long)])
    if args.capital is not None:
        shell_cmd_parts.extend(["--capital", str(args.capital)])
    run_shell_cmd = safe_shell_join(shell_cmd_parts)
    default_code = "600519.SH"
    if args.code is not None:
        stock_code = args.code.upper()
        logger.info(f"💻 使用命令行指定股票：{stock_code}")
    else:
        print("-" * 60)
        user_input = input(f"请输入股票代码(回车使用默认 {default_code})：").strip()
        stock_code = user_input.upper() if user_input else default_code
    logger.info(f"回测标的 {stock_code}，时间区间 {args.start} ~ {args.end}")
    try:
        df_raw = fetch_data(token=args.token, code=stock_code, start=args.start, end=args.end)
        suggest_cap = auto_calc_capital(df_raw)
        max_price = df_raw["close"].max()
        min_one_lot_cost = max_price * 100 * (1 + COMMISSION_RATE + SLIPPAGE)
        logger.info(f"📌回测区间最高股价 {max_price:.2f}，买入1手预估最低需要资金：{min_one_lot_cost:,.2f} 元")
        if args.capital is not None:
            initial_capital = args.capital
            logger.info(f"💰 使用命令行指定初始资金：{initial_capital:,.2f}元")
            if initial_capital < min_one_lot_cost:
                logger.warning(f"⚠️你设置的初始资金小于1手最低需要资金，后续触发买入信号会无法成交！")
        else:
            print("-"*60)
            print(f"📊 回测区间股票最高价:{max_price:.2f}元，系统推荐资金: {suggest_cap:,.0f} 元")
            user_input = input(f"请输入初始资金(直接回车使用推荐值): ").strip()
            if not user_input:
                initial_capital = suggest_cap
                logger.info(f"💰 用户直接回车，使用系统推荐资金 {initial_capital:,.0f} 元")
            else:
                try:
                    input_val = float(user_input)
                    if input_val <= 0:
                        logger.warning("输入资金不能小于等于0，使用系统推荐资金")
                        initial_capital = suggest_cap
                    else:
                        initial_capital = input_val
                        logger.info(f"💰 用户手动输入初始资金：{initial_capital:,.2f}元")
                        if initial_capital < min_one_lot_cost:
                            logger.warning(f"⚠️你输入的初始资金小于1手最低需要资金，后续触发买入信号会无法成交！")
                except ValueError:
                    logger.warning("输入不是有效数字！使用系统推荐资金")
                    initial_capital = suggest_cap
        engine = BacktestEngine(df_raw, initial_capital=initial_capital, short_window=args.short, long_window=args.long)
        equity_df, buy_points, sell_points, trade_records = engine.run()
        total_ret, annual_ret, sharpe, max_dd, net_series = calc_metrics(equity_df, RISK_FREE_RATE)
        bh_df = df_raw[["trade_date", "close"]].copy()
        bh_df["bh_net"] = bh_df["close"] / bh_df["close"].iloc[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_name = os.path.join(OUTPUT_DIR, f"event_backtest_report_{stock_code.replace('.','_')}_{timestamp}.csv")
        equity_df.to_csv(csv_name, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅每日账户明细导出 {csv_name}")
        trade_df = pd.DataFrame(trade_records)
        trade_csv = os.path.join(OUTPUT_DIR, f"event_trade_records_{stock_code.replace('.','_')}_{timestamp}.csv")
        trade_df.to_csv(trade_csv, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅每笔成交记录导出 {trade_csv}")
        txt_name = os.path.join(OUTPUT_DIR, f"event_summary_{stock_code.replace('.','_')}_{timestamp}.txt")
        with open(txt_name,"w",encoding="utf‑8") as f:
            f.write("⚠️本结果仅用于学习演示，不构成任何投资建议。历史回测不能代表未来收益。\n")
            f.write(f"脚本名称:event_backtest.py【事件驱动双均线回测】\n")
            f.write(f"脚本版本: {__version__}\n")
            f.write(f"本地运行时间: {local_run_time}\n")
            f.write(f"股票代码:{stock_code}\n")
            f.write(f"复权模式:{ADJ_MODE}\n")
            f.write(f"均线周期:MA{args.short}-MA{args.long}\n")
            f.write(f"初始资金:{initial_capital:.2f}\n")
            f.write(f"时间区间:{args.start}-{args.end}\n")
            f.write(f"手续费率:{COMMISSION_RATE},滑点:{SLIPPAGE},印花税:{STAMP_TAX}\n")
            f.write(f"总收益率:{total_ret:.2%} 年化:{annual_ret:.2%} 夏普:{sharpe:.3f} 最大回撤:{max_dd:.2%}\n")
            f.write(f"成交笔数:{len(trade_records)}\n")
            f.write(f"完整启动参数: {vars(args)}\n")
            f.write(f"可复现运行命令:\n{run_shell_cmd}\n")
            f.write("# End of report\n")
        logger.info(f"✅汇总报告txt导出 {txt_name}")
        logger.info("\n" + "=" * 60)
        bh_total = float(bh_df["bh_net"].iloc[-1]) - 1
        bh_years = (bh_df["trade_date"].iloc[-1] - bh_df["trade_date"].iloc[0]).days / 365.0
        bh_annual = (1 + bh_total) ** (1 / bh_years) - 1 if bh_years > 0 else np.nan
        logger.info(f"{'指标':<16}{'双均线策略':>14}{'买入持有':>14}")
        logger.info("-" * 60)
        logger.info(f"总收益率    {total_ret:>13.2%}{bh_total:>13.2%}")
        logger.info(f"年化收益率   {annual_ret:>13.2%}{bh_annual:>13.2%}")
        logger.info(f"夏普比率    {sharpe:>13.3f}{'—':>13}")
        logger.info(f"最大回撤    {max_dd:>13.2%}{'—':>13}")
        logger.info(f"成交笔数    {len(trade_records):>13d}{'—':>13}")
        logger.info("=" * 60)
        plot_chart(equity_df, net_series, bh_df, stock_code, buy_points, sell_points, args.short, args.long)
        logger.info("✅事件驱动回测全部流程执行完成")
    except KeyboardInterrupt:
        logger.info("\n👋 用户手动 Ctrl+C 终止程序，正常退出")
    except MemoryError:
        logger.error("❌内存不足！请缩小回测时间区间或者减少股票数量")
    except Exception as e:
        logger.error(f"程序异常终止：{str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 用户手动Ctrl+C终止程序，正常退出")
    except MemoryError:
        logger.error("❌内存不足，请缩小回测区间")
    except Exception as e:
        logger.error(f"顶层程序异常终止 {str(e)}", exc_info=True)
