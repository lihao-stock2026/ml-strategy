"""
机器学习股价预测策略
利用历史K线、均线、MACD、RSI、成交量等技术特征
使用随机森林分类器预测股票次日涨跌，生成机器学习交易策略
与买入持有策略做回测对比

工程化进阶功能：
1. argparse命令行传参：运行直接指定股票代码、起止时间、Tushare令牌，无需修改源码
2. logging日志模块：同时输出控制台，并且保存本地日志文件 ml_backtest.log
3. 完整异常捕获：网络请求失败、Tushare积分不足、数据为空、数据量不足友好提示，程序不直接崩溃
4. 本地CSV缓存：相同股票时间区间不再重复请求接口，节约积分，防止限流
5. 使用后复权行情数据，模拟A股手续费、滑点
6. 输出回测净值对比图表，保存为png图片；--no_plot可关闭绘图适配无图形服务器
7. ✨新增交互式输入股票代码，命令行不传参数可控制台手动输入
8. ✨增强：--clear_cache一键清理缓存；导出回测明细csv；接口简易重试；读取.env加载token；输出summary报告txt
9. ✨小优化：报告文件名带时间戳；区分tushare积分/限流错误；命令行可调随机森林超参；输出混淆矩阵图片；报告保存全部入参便于复现

⚠️重要说明：本脚本采用简化数学收益回测，没有模拟真实现金与股票持仓；
event_backtest.py为完整事件驱动框架，模拟真实账户现金、股数、100手交易限制。

风险免责：
    本代码仅用于学习演示，不构成任何投资建议。
    历史回测结果不能代表未来实际收益。

依赖库：
    tushare, pandas, numpy, scikit‑learn, matplotlib, python‑dotenv
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
# 自动适配绘图后端：本地弹窗，服务器无界面自动切Agg
try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 加载.env环境变量
load_dotenv()

# 绘图中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局回测参数 =====================
RISK_FREE_RATE = 0.02          # 年化无风险利率
COST_RATE = 0.0003             # 单边交易手续费 万三
SLIPPAGE = 0.0005              # 模拟滑点 0.05%
PROB_THRESHOLD = 0.50          # 预测上涨概率阈值
TRAIN_RATIO = 0.8              # 训练集占比
MAX_RETRY = 3                   # tushare最大重试次数

# 模型使用的特征列表
FEATURE_COLS = [
    'close_ma5_ratio', 'close_ma20_ratio',
    'rsi14', 'macd_dif', 'macd_dea', 'macd_hist',
    'vol_change', 'vol_ratio',
    'ret_1d', 'ret_5d', 'ret_10d', 'volatility'
]

def clear_all_cache():
    """删除全部cache_*.csv缓存文件"""
    cache_files = glob.glob("cache_*.csv")
    for f in cache_files:
        os.remove(f)
    logging.info(f"✅已清理缓存文件，共删除 {len(cache_files)} 个")

def setup_logger():
    """
    初始化日志对象
    同时输出控制台 + 写入本地ml_backtest.log日志文件
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("ml_backtest.log", encoding="utf-8"),
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

def fetch_data(token: str, code: str, start: str, end: str) -> pd.DataFrame:
    """
    从Tushare接口获取后复权日线行情数据，增加本地csv缓存
    ✅修复BUG：读完缓存/接口后强制按start、end过滤日期，防止缓存带入多余时间
    :param token: 外部传入的token（命令行--token参数），为空则尝试.env、环境变量、控制台输入
    :param code: 股票ts代码，例如 600519.SH
    :param start: 开始日期，格式 YYYYMMDD
    :param end: 结束日期，格式 YYYYMMDD
    :return: 处理完成的DataFrame，时间升序排列
    """
    # token优先级：命令行 > .env > 系统环境变量 > 控制台输入
    if not token:
        token = os.getenv("TUSHARE_TOKEN", "")
        if not token:
            token = input('粘贴你的tushare token：').strip()
    ts.set_token(token)
    pro = ts.pro_api(token)
    # 构造缓存文件名，不同股票时间不会覆盖
    cache_file = f"cache_{code.replace('.','_')}_{start}_{end}.csv"
    if os.path.exists(cache_file):
        logger.info(f"读取本地缓存数据：{cache_file}")
        df_raw = pd.read_csv(cache_file, encoding="utf-8-sig").reset_index(drop=True)
        # 【重点修复】先转字符串，再按%Y%m%d格式解析日期，防止数字解析成时间戳
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
    else:
        logger.info("本地缓存不存在，从Tushare拉取数据...")
        df_raw = ts_pro_bar_with_retry(pro, code, adj='hfq', start_date=start, end_date=end)
        if df_raw is None or df_raw.empty:
            logger.warning(f"股票 {code} 返回的数据集为空")
            raise ValueError("数据集为空，请检查股票代码、时间范围或者Tushare积分")
        # 【重点修复】保存csv前，trade_date强制转字符串，不要保存数字！
        df_raw["trade_date"] = df_raw["trade_date"].astype(str)
        df_raw.to_csv(cache_file, index=False, encoding="utf-8-sig")
        logger.info(f"数据已写入本地缓存：{cache_file}")

    # =========关键修复：强制时间过滤，不管缓存还是接口返回，只保留 [start, end] 区间
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    # 新增：剔除停牌、价格成交量为空的行
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 80:
        raise ValueError(f"数据行数不足，当前仅{len(df)}条，至少需要80条以上日线数据")
    logger.info(f"成功加载 {code} 日线数据，共 {len(df)} 条")
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造全部技术指标特征 + 监督学习标签
    标签规则：T日的特征，预测T+1日收盘价是否上涨，上涨=1，下跌=0
    :param df: 原始后复权行情DataFrame
    :return: 增加特征与label列后的DataFrame
    """
    df = df.copy()
    # 移动平均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['close_ma5_ratio'] = df['close'] / df['ma5']
    df['close_ma20_ratio'] = df['close'] / df['ma20']
    # RSI‑14 相对强弱指标（Wilder平滑算法）【增加除零保护】
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    df['rsi14'] = 100 - (100 / (1 + rs))
    # MACD指标
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_dif'] - df['macd_dea']
    # 成交量特征
    df['vol_change'] = df['vol'].pct_change()
    df['vol_ma5'] = df['vol'].rolling(5).mean()
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    # 收益率与波动率特征
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['volatility'] = df['ret_1d'].rolling(14).std()
    # 构建标签：预测次日涨跌
    df['label'] = (df['close'].shift(-1) > df['close']).astype(np.int64)
    up_rate = df['label'].mean()
    logger.info(f"样本标签分布：上涨 {up_rate:.2%}，下跌 {1-up_rate:.2%}")
    return df

def calc_metrics(net_value: pd.Series, rf: float):
    """
    计算回测评价指标：总收益率、年化收益率、夏普比率、最大回撤
    :param net_value: 策略累计净值序列
    :param rf: 年化无风险利率
    :return: 指标字典
    """
    ret = net_value.pct_change().dropna()
    total_return = net_value.iloc[-1] / net_value.iloc[0] - 1
    years = len(ret) / 252
    if years <= 0:
        return {"total_return":0, "annual_return":0, "sharpe":0, "max_drawdown":0}
    annual_return = (1 + total_return) ** (1 / years) - 1
    sharpe = 0.0
    if ret.std() > 1e-8:
        sharpe = (ret.mean() - rf / 252) / ret.std() * np.sqrt(252)
    cummax = net_value.cummax()
    max_drawdown = ((net_value - cummax) / cummax).min()
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

def ml_backtest(df: pd.DataFrame, train_ratio=0.8, prob_threshold=0.5, n_splits=5, n_estimators=200, max_depth=5):
    """
    随机森林模型训练 + 样本外回测
    使用时间序列交叉验证，每折内部做标准化，避免数据泄露
    根据预测上涨概率生成交易信号，模拟交易成本滑点
    """
    feature_cols = FEATURE_COLS
    split_idx = int(len(df) * train_ratio)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()

    logger.info(f"训练集时间范围：{train_raw['trade_date'].min().date()} ~ {train_raw['trade_date'].max().date()}")
    logger.info(f"样本外回测测试集时间范围：{test_raw['trade_date'].min().date()} ~ {test_raw['trade_date'].max().date()}")

    # 删除存在空值的行
    train = train_raw.dropna(subset=feature_cols + ["label"])
    test = test_raw.dropna(subset=feature_cols + ["label"])
    if len(train) < 20 or len(test) < 10:
        raise ValueError("去除空值后训练集或测试集样本数量过少，无法训练模型")
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]
    logger.info("===== 时间序列交叉验证 =====")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_train.iloc[tr_idx])
        X_va = scaler_fold.transform(X_train.iloc[va_idx])
        fold_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        fold_model.fit(X_tr, y_train.iloc[tr_idx])
        score = fold_model.score(X_va, y_train.iloc[va_idx])
        cv_scores.append(score)
        logger.info(f"第{fold}折 验证集准确率: {score:.4f}")
    logger.info(f"交叉验证平均准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    # 在全部训练集上训练最终模型
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_s, y_train)
    # 使用预测概率而不是直接0/1硬分类
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n测试集准确率: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\n特征重要性Top5:\n{importance.head()}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["下跌","上涨"])
    fig_cm, ax_cm = plt.subplots(figsize=(6,5))
    disp.plot(ax=ax_cm)
    cm_filename = f"ml_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(cm_filename, dpi=120)
    plt.close(fig_cm)
    logger.info(f"✅混淆矩阵图片已保存 {cm_filename}")

    test = test.copy()
    test["prob_up"] = y_pred_proba
    test["signal"] = y_pred
    # 计算策略收益：信号翻转一次代表一次完整买卖，扣除双向手续费+滑点
    test["trade"] = test["signal"].diff().abs()
    cost_per_switch = 2 * COST_RATE + SLIPPAGE
    test["strategy_ret"] = test["signal"].shift(1) * test["ret_1d"] - test["trade"].shift(1) * cost_per_switch
    test["strategy_net"] = (1 + test["strategy_ret"]).cumprod()
    test["buyhold_net"] = (1 + test["ret_1d"]).cumprod()
    # 新增：统计总交易次数
    trade_cnt = test["trade"].sum()
    logger.info(f"样本外区间总共交易次数：{int(trade_cnt)}")
    return test, importance

def plot_result(test_df: pd.DataFrame, importance: pd.Series, stock_code: str, no_plot=False):
    if no_plot:
        logger.info("📌 --no_plot已开启，跳过绘图")
        return
    test_df = test_df.copy()
    date_list = test_df["trade_date"].dt.to_pydatetime()
    fig, axes = plt.subplots(3, 1, figsize=(14,12))
    axes[0].plot(date_list, test_df["strategy_net"], label="机器学习策略", linewidth=1.5)
    axes[0].plot(date_list, test_df["buyhold_net"], label="买入持有", alpha=0.7)
    axes[0].set_title(f"{stock_code}：机器学习策略 VS 买入持有（样本外测试集）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].plot(date_list, test_df["close"], color="gray", alpha=0.5, label="后复权收盘价")
    long = test_df[test_df["signal"] == 1]
    flat = test_df[test_df["signal"] == 0]
    long_dates = long["trade_date"].dt.to_pydatetime()
    flat_dates = flat["trade_date"].dt.to_pydatetime()
    axes[1].scatter(long_dates, long["close"], marker="^", c="red", s=12, label="持仓信号")
    axes[1].scatter(flat_dates, flat["close"], marker="v", c="green", s=12, label="空仓信号")
    axes[1].set_title("价格与模型生成交易信号")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    importance.plot(kind="bar", ax=axes[2])
    axes[2].set_title("模型特征重要性")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    filename = f"ml_result_{stock_code.replace('.','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150)
    logger.info(f"回测图表已保存至：{filename}")
    plt.show()

def main():
    """程序入口，支持交互式输入股票代码，支持命令行调参训练集比例、概率阈值"""
    parser = argparse.ArgumentParser(description="机器学习股票回测程序")
    parser.add_argument("--code", type=str, default=None, help="股票ts代码，不传则控制台手动输入")
    parser.add_argument("--start", type=str, default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--token", type=str, default="", help="Tushare api令牌")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="训练集占总数据比例，默认0.8")
    parser.add_argument("--prob_threshold", type=float, default=PROB_THRESHOLD, help="持仓概率阈值，默认0.5")
    parser.add_argument("--n_estimators", type=int, default=200, help="随机森林树数量，默认200")
    parser.add_argument("--max_depth", type=int, default=5, help="随机森林树最大深度，默认5")
    parser.add_argument("--clear_cache", action="store_true", help="运行前清除全部cache_*.csv缓存")
    parser.add_argument("--no_plot", action="store_true", help="设置后不生成图片，适合服务器无图形环境")
    args = parser.parse_args()

    # 参数校验
    if args.train_ratio <=0 or args.train_ratio >=1:
        raise ValueError("train_ratio必须在(0,1)之间")
    if args.prob_threshold <0 or args.prob_threshold>1:
        raise ValueError("prob_threshold必须在0‑1之间")
    if args.n_estimators <1 or args.max_depth <1:
        raise ValueError("n_estimators、max_depth必须大于等于1")

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

    logger.info(f"启动机器学习回测，股票:{stock_code}，时间区间 {args.start} ~ {args.end}")
    try:
        df_raw = fetch_data(token=args.token, code=stock_code, start=args.start, end=args.end)
        df_feat = build_features(df_raw)
        test_df, imp = ml_backtest(df_feat,
                                   train_ratio=args.train_ratio,
                                   prob_threshold=args.prob_threshold,
                                   n_estimators=args.n_estimators,
                                   max_depth=args.max_depth)
        ml_metric = calc_metrics(test_df["strategy_net"], RISK_FREE_RATE)
        bh_metric = calc_metrics(test_df["buyhold_net"], RISK_FREE_RATE)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = f"ml_report_{stock_code.replace('.','_')}_{timestamp_str}.csv"
        export_cols = ["trade_date","close","prob_up","signal","strategy_ret","strategy_net","buyhold_net"]
        test_df[export_cols].to_csv(out_csv, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅机器学习回测明细已导出：{out_csv}")

        report_txt = f"ml_summary_{stock_code.replace('.','_')}_{timestamp_str}.txt"
        with open(report_txt,"w",encoding="utf‑8") as f:
            f.write(f"股票代码:{stock_code}\n")
            f.write(f"时间区间:{args.start}‑{args.end}\n")
            f.write(f"训练集比例:{args.train_ratio},概率阈值:{args.prob_threshold}\n")
            f.write(f"n_estimators:{args.n_estimators},max_depth:{args.max_depth}\n")
            f.write(f"--no_plot: {args.no_plot}\n")
            f.write(f"完整命令行参数：{vars(args)}\n")
            f.write(f"总收益率 策略:{ml_metric['total_return']:.2%}  买入持有:{bh_metric['total_return']:.2%}\n")
            f.write(f"年化收益率 策略:{ml_metric['annual_return']:.2%}  买入持有:{bh_metric['annual_return']:.2%}\n")
            f.write(f"夏普比率 策略:{ml_metric['sharpe']:.3f}  买入持有:{bh_metric['sharpe']:.3f}\n")
            f.write(f"最大回撤 策略:{ml_metric['max_drawdown']:.2%}  买入持有:{bh_metric['max_drawdown']:.2%}\n")
        logger.info(f"✅文本回测报告已保存 {report_txt}")

        logger.info("\n" + "="*55)
        logger.info(f"{'指标':<16}{'机器学习策略':>13}{'买入持有':>13}")
        logger.info("-"*55)
        logger.info(f"总收益率    {ml_metric['total_return']:>12.2%}{bh_metric['total_return']:>12.2%}")
        logger.info(f"年化收益率   {ml_metric['annual_return']:>12.2%}{bh_metric['annual_return']:>12.2%}")
        logger.info(f"夏普比率    {ml_metric['sharpe']:>12.3f}{bh_metric['sharpe']:>12.3f}")
        logger.info(f"最大回撤    {ml_metric['max_drawdown']:>12.2%}{bh_metric['max_drawdown']:>12.2%}")
        logger.info("="*55)
        plot_result(test_df, imp, stock_code, no_plot=args.no_plot)
        logger.info("机器学习回测全部执行完毕")
    except Exception as e:
        logger.error(f"回测程序异常终止：{str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
