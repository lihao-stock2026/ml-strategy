"""
事件驱动双均线回测框架
金叉买入，死叉卖出；模拟A股现金、持仓、100股最小交易单位、手续费、印花税、滑点
交互式输入股票代码与初始资金，也支持命令行传参

工程化功能：
1. argparse命令行传参；交互式控制台输入股票、初始资金
2. logging日志，控制台+本地event_backtest.log保存
3. 本地CSV缓存，避免重复请求Tushare
4. 绘图标记绿色买入▲、红色卖出▼
5. 自动计算推荐初始资金，防止股价过高资金不足无法买入
6. ✨增强：--short --long 设置均线周期；--clear_cache清理缓存；导出账户明细+交易明细csv；接口简易重试；.env读取token；输出txt报告；参数合法性校验
7. ✨小优化：报告文件名带时间戳；tushare区分积分/限流报错；回测前预判打印1手最低需要资金；成交记录增加单笔盈亏；报告保存全部入参便于复现

风险免责：
    本代码仅用于学习演示，不构成任何投资建议。
    历史回测结果不能代表未来实际收益。
"""
import os
import glob
import logging
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 加载.env环境变量
load_dotenv()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局配置参数 =====================
RISK_FREE_RATE = 0.02
COMMISSION_RATE = 0.0003    # 单边手续费万三
STAMP_TAX = 0.001           # A股卖出印花税千1
SLIPPAGE = 0.0005           # 模拟滑点
MAX_RETRY = 3

def clear_all_cache():
    """删除全部cache_*.csv缓存文件"""
    cache_files = glob.glob("cache_*.csv")
    for f in cache_files:
        os.remove(f)
    logging.info(f"✅已清理缓存文件，共删除 {len(cache_files)} 个")

def setup_logger():
    """初始化日志对象，控制台输出 + 写入event_backtest.log"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("event_backtest.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def ts_pro_bar_with_retry(pro, code, adj, start_date, end_date, max_retry=MAX_RETRY):
    """tushare pro_bar简易重试封装，区分积分、限流错误"""
    for attempt in range(max_retry):
        try:
            df = pro.pro_bar(ts_code=code, adj=adj, start_date=start_date, end_date=end_date)
            return df
        except Exception as e:
            err_msg = str(e)
            if "积分" in err_msg or "permission" in err_msg.lower():
                raise ConnectionError("❌Tushare积分不足，无法调取数据，请提升积分！")
            if "访问频率" in err_msg or "limit" in err_msg.lower():
                logger.warning(f"Tushare接口限流，等待重试 {attempt+1}/{max_retry}")
            else:
                logger.warning(f"Tushare请求失败，重试 {attempt+1}/{max_retry}, err:{err_msg}")
            time.sleep(1.2)
    raise ConnectionError("Tushare接口多次请求失败，请检查网络、token、账号积分")

class Event:
    """事件基类，所有事件都继承该类"""
    pass

class MarketEvent(Event):
    """行情事件：当新一根K线数据到达时触发"""
    def __init__(self, date, open_, high, low, close, volume):
        self.type = "MARKET"
        self.date = date
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class SignalEvent(Event):
    """策略信号事件：策略计算后产生交易信号 1=买入，‑1=卖出"""
    def __init__(self, date, signal):
        self.type = "SIGNAL"
        self.date = date
        self.signal = signal

class OrderEvent(Event):
    """订单事件：投资组合根据信号生成待执行订单"""
    def __init__(self, date, direction, quantity):
        self.type = "ORDER"
        self.date = date
        self.direction = direction
        self.quantity = quantity

class FillEvent(Event):
    """成交回报事件：订单模拟成交之后，返回成交结果"""
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
    数据处理器
    迭代历史DataFrame，逐根K线生成MarketEvent行情事件
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.idx = 0

    def next_bar(self):
        """获取下一根K线，遍历结束返回None"""
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
    双均线策略
    金叉：短期均线上穿长期均线 → 买入信号
    死叉：短期均线下穿长期均线 → 卖出信号
    """
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window
        self.price_series = pd.Series(dtype=float)
        self.buy_points = []
        self.sell_points = []

    def calculate_signal(self, bar: MarketEvent):
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
    投资组合模块
    管理现金、持仓数量；接收信号生成订单；接收成交更新账户状态；记录每日权益曲线
    A股规则：股票必须100股整数手交易
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
        self.current_price = bar.close
        total_value = self.cash + self.position * self.current_price
        self.equity_curve.append({
            "date": bar.date,
            "cash": self.cash,
            "position": self.position,
            "equity": total_value
        })

    def generate_order(self, signal_event: SignalEvent, estimate_open_price: float):
        """将策略信号转换为可执行订单，控制只能满仓开仓/全部清仓"""
        if signal_event.signal == 1 and self.position == 0:
            min_cost_per_lot = estimate_open_price * 100 * (1 + COMMISSION_RATE + SLIPPAGE)
            quantity = int((self.cash / estimate_open_price / 100)) * 100
            if quantity >= 100:
                logger.info(f"生成买入订单，拟下单数量:{quantity}股 (股价{estimate_open_price:.2f})")
                return OrderEvent(signal_event.date, "BUY", quantity)
            else:
                logger.warning(
                    f"⚠️ 资金不足，无法买入！当前股价≈{estimate_open_price:.2f}元，"
                    f"买1手(100股)至少需要≈{min_cost_per_lot:,.2f}元，"
                    f"但账户仅剩现金{self.cash:,.2f}元，缺口≈{min_cost_per_lot - self.cash:,.2f}元"
                )
        elif signal_event.signal == -1 and self.position > 0:
            logger.info(f"生成卖出订单，拟卖出全部持仓:{self.position}股")
            return OrderEvent(signal_event.date, "SELL", self.position)
        return None

    def update_fill(self, fill_event: FillEvent):
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
    模拟订单执行器
    订单在下一根K线开盘价成交，加入滑点模拟真实市场冲击
    """
    def __init__(self, slippage: float):
        self.slippage = slippage

    def execute(self, order: OrderEvent, next_bar: MarketEvent) -> FillEvent:
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
    """回测主引擎：事件驱动主循环，串联全部组件"""
    def __init__(self, df: pd.DataFrame, initial_capital=100000.0, short_window=10, long_window=30):
        self.df = df
        self.initial_capital = initial_capital
        self.short_window = short_window
        self.long_window = long_window

    def run(self):
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

        return pd.DataFrame(portfolio.equity_curve), strategy.buy_points, strategy.sell_points, portfolio.trade_records

def calc_metrics(equity_df: pd.DataFrame, rf: float):
    """计算回测指标：总收益、年化收益、夏普比率、最大回撤"""
    net = equity_df["equity"] / equity_df["equity"].iloc[0]
    ret = net.pct_change().dropna()
    total_return = net.iloc[-1] - 1
    years = len(ret) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe = 0.0
    if ret.std() > 1e-8:
        sharpe = (ret.mean() - rf / 252) / ret.std() * np.sqrt(252)

    cummax = net.cummax()
    max_dd = ((net - cummax) / cummax).min()
    return total_return, annual_return, sharpe, max_dd, net

def auto_calc_capital(df: pd.DataFrame) -> float:
    """自动计算足够的初始资金"""
    max_price = df["close"].max()
    suggested_capital = max_price * 1000 * 1.15
    suggested_capital = np.ceil(suggested_capital / 10000) * 10000
    return float(suggested_capital)

def plot_chart(equity_df: pd.DataFrame, net_series, bh_df, stock_code, buy_points, sell_points, short_window, long_window):
    """绘制净值曲线，标记买卖点三角"""
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

    axes[0].set_title(f"{stock_code}：事件驱动MA{short_window}‑MA{long_window}回测净值")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(equity_df["date"], equity_df["cash"], label="账户现金余额", color="orange")
    axes[1].set_title("账户现金变化曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"event_result_{stock_code.replace('.','_')}_{timestamp}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    logger.info(f"回测图表已保存至：{filename}")

def fetch_data(token: str, code: str, start: str, end: str) -> pd.DataFrame:
    """获取Tushare日线行情，本地csv缓存"""
    if not token:
        token = os.getenv("TUSHARE_TOKEN", "")
        if not token:
            token = input('粘贴你的tushare token：').strip()

    ts.set_token(token)
    pro = ts.pro_api(token)

    cache_file = f"cache_{code.replace('.','_')}_{start}_{end}.csv"

    if os.path.exists(cache_file):
        logger.info(f"读取本地缓存数据：{cache_file}")
        df_raw = pd.read_csv(cache_file, encoding="utf-8-sig").reset_index(drop=True)
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
    else:
        logger.info("本地缓存不存在，从Tushare拉取数据...")
        df_raw = ts_pro_bar_with_retry(pro, code, adj=None, start_date=start, end_date=end)
        if df_raw is None or df_raw.empty:
            raise ValueError(f"股票 {code} 获取到空数据集，请检查参数")
        df_raw["trade_date"] = df_raw["trade_date"].astype(str)
        df_raw.to_csv(cache_file, index=False, encoding="utf-8-sig")
        logger.info(f"数据已写入本地缓存：{cache_file}")

    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    df = df.dropna(subset=["open","high","low","close","vol"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    logger.info(f"成功加载 {code} 日线数据，共 {len(df)} 条")
    return df

def main():
    """程序入口，支持交互式输入股票代码与初始资金"""
    parser = argparse.ArgumentParser(description="事件驱动双均线回测框架")
    parser.add_argument("--code", type=str, default=None, help="股票ts代码，不传则控制台手动输入")
    parser.add_argument("--start", type=str, default="20190601", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--token", type=str, default="", help="Tushare api令牌")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金(元)，不传则进入控制台交互式输入")
    parser.add_argument("--short", type=int, default=10, help="短期均线周期，默认10")
    parser.add_argument("--long", type=int, default=30, help="长期均线周期，默认30")
    parser.add_argument("--clear_cache", action="store_true", help="运行前清除全部cache_*.csv缓存")
    args = parser.parse_args()

    # 参数合法性校验
    if args.short < 1 or args.long <1:
        raise ValueError("均线周期不能小于1")
    if args.short >= args.long:
        logger.warning("提示：短期均线大于等于长期均线，策略逻辑反常，请确认参数")

    if args.clear_cache:
        clear_all_cache()

    # =========交互式输入股票代码=========
    default_code = "600519.SH"
    if args.code is not None:
        stock_code = args.code
        logger.info(f"💻 使用命令行指定股票代码: {stock_code}")
    else:
        print("-"*60)
        user_input_code = input(f"请输入股票代码(例如000001.SZ，直接回车使用默认{default_code}): ").strip()
        if not user_input_code:
            stock_code = default_code
            logger.info(f"📌 用户直接回车，使用默认股票代码 {stock_code}")
        else:
            stock_code = user_input_code

    logger.info(f"启动事件驱动回测，股票:{stock_code}，MA{args.short}-MA{args.long}，数据区间 {args.start} ~ {args.end}")
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
        report_csv = f"event_backtest_report_{stock_code.replace('.','_')}_{timestamp}.csv"
        equity_df.to_csv(report_csv, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅每日账户明细导出：{report_csv}")

        trade_df = pd.DataFrame(trade_records)
        trade_csv = f"event_trade_records_{stock_code.replace('.','_')}_{timestamp}.csv"
        trade_df.to_csv(trade_csv, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅每笔成交记录导出：{trade_csv}")

        report_txt = f"event_summary_{stock_code.replace('.','_')}_{timestamp}.txt"
        with open(report_txt,"w",encoding="utf‑8") as f:
            f.write(f"股票代码:{stock_code}\n")
            f.write(f"均线周期:MA{args.short}‑MA{args.long}\n")
            f.write(f"时间区间:{args.start}‑{args.end}\n")
            f.write(f"初始资金:{initial_capital:.2f}\n")
            f.write(f"完整命令行参数：{vars(args)}\n")
            f.write(f"期末权益:{equity_df['equity'].iloc[-1]:.2f}\n")
            f.write(f"总收益率:{total_ret:.2%}\n")
            f.write(f"年化收益率:{annual_ret:.2%}\n")
            f.write(f"夏普比率:{sharpe:.3f}\n")
            f.write(f"最大回撤:{max_dd:.2%}\n")
            f.write(f"买入信号次数:{len(buy_points)},卖出信号次数:{len(sell_points)},实际成交笔数:{len(trade_records)}\n")
        logger.info(f"✅文本回测报告已保存 {report_txt}")

        logger.info("\n" + "="*55)
        logger.info(f"股票代码: {stock_code}")
        logger.info(f"均线周期: MA{args.short}‑MA{args.long}")
        logger.info(f"初始资金: {initial_capital:,.2f}")
        logger.info(f"期末权益: {equity_df['equity'].iloc[-1]:,.2f}")
        logger.info(f"总收益率: {total_ret:.2%}")
        logger.info(f"年化收益率: {annual_ret:.2%}")
        logger.info(f"夏普比率: {sharpe:.3f}")
        logger.info(f"最大回撤: {max_dd:.2%}")
        logger.info(f"买入信号次数: {len(buy_points)}")
        logger.info(f"卖出信号次数: {len(sell_points)}")
        logger.info(f"实际成交笔数: {len(trade_records)}")
        logger.info("="*55)

        plot_chart(equity_df, net_series, bh_df, stock_code, buy_points, sell_points, args.short, args.long)
        logger.info("事件驱动回测执行完毕")

    except Exception as e:
        logger.error(f"回测程序异常终止：{str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
