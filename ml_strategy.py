"""
机器学习股价预测策略 v1.0.5【完善最终版】
利用历史K线、均线、MACD、RSI、成交量等技术特征
使用随机森林分类器预测股票次日涨跌，生成机器学习交易策略
与买入持有策略做回测对比
⚠️重要说明：
本向量化回测为快速粗筛工具，**未模拟A股100股最小交易单位**；严谨回测评估请使用event_backtest.py事件驱动模块。
标签使用shift(-1)获取未来价格，属于离线回测未来函数；实盘需要滚动窗口重新训练，不可直接拿未来行情。
更新v1.0.5：
1.增加ret_1d空值防护，防止净值全部NaN
2.注释标明向量化版本不模拟A股1手规则
3.全局统一随机种子，和项目其他脚本对齐
4.全部输出定向output文件夹
5.加固边界与异常捕获
⚠️风险免责：本代码仅用于学习演示，不构成任何投资建议。历史回测结果不能代表未来实际收益。
"""
import os
import glob
import argparse
import time
import shlex
import random
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv
import tushare as ts
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

__version__ = "1.0.5"
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def safe_shell_join(args_list):
    """
    跨平台安全拼接shell命令参数
    :param args_list: list[str] 命令参数列表
    :return: str 拼接后的命令字符串
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
import matplotlib.dates as mdates


load_dotenv()
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

RISK_FREE_RATE = 0.02
COST_RATE = 0.0003
SLIPPAGE = 0.0005
PROB_THRESHOLD = 0.50
TRAIN_RATIO = 0.8
MAX_RETRY = 3
ADJ_MODE = 'hfq'
API_MIN_INTERVAL = 0.4

FEATURE_COLS = [
    'close_ma5_ratio', 'close_ma20_ratio',
    'rsi14', 'macd_dif', 'macd_dea', 'macd_hist',
    'vol_change', 'vol_ratio',
    'ret_1d', 'ret_5d', 'ret_10d', 'volatility'
]


def clear_all_cache():
    """清除本地所有cache_*.csv行情缓存文件"""
    cache_files = glob.glob("cache_*.csv")
    for f in cache_files:
        os.remove(f)
    logging.info(f"✅已清理行情缓存，共删除 {len(cache_files)} 个")


def clean_all_output():
    """清理向量化回测输出产物，保留行情cache缓存"""
    patterns = [
        os.path.join(OUTPUT_DIR, "ml_*.png"),
        os.path.join(OUTPUT_DIR, "ml_*.csv"),
        os.path.join(OUTPUT_DIR, "ml_*.txt"),
        os.path.join(OUTPUT_DIR, "*.log")
    ]
    cnt = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            cnt += 1
    print(f"🧹已清理输出报告/图片/日志，共删除 {cnt} 个；行情cache缓存已保留")


def setup_logger():
    """
    初始化日志对象，文件滚动保存，同时输出控制台
    :return: logging.Logger 日志实例
    """
    log_path = os.path.join(OUTPUT_DIR, "ml_backtest.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    file_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


def ts_pro_bar_with_retry(code, adj, start_date, end_date, max_retry=MAX_RETRY):
    """
    tushare pro_bar接口带重试请求，处理限流、积分异常
    :param code: str ts_code股票代码
    :param adj: str 复权模式 hfq/qfq
    :param start_date: str YYYYMMDD
    :param end_date: str YYYYMMDD
    :param max_retry: int 最大重试次数
    :return: pd.DataFrame | None 行情DataFrame，空则返回None
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
                raise RuntimeError("❌Tushare积分不足，请提升账号积分")
            if ("访问频率" in err_msg.lower()) or ("adj_factor" in err_msg):
                if attempt + 1 < max_retry:
                    sleep_sec = 60 + random.uniform(2, 8)
                    logger.warning(f"Tushare限流，随机休眠 {sleep_sec:.1f}s 重试 {attempt + 1}/{max_retry}")
                    time.sleep(sleep_sec)
            else:
                logger.warning(f"Tushare请求失败，重试 {attempt + 1}/{max_retry}, err:{err_msg}")
    raise RuntimeError("Tushare多次请求失败")


def fetch_data(token: str, code: str, start: str, end: str, refresh_cache: bool = False) -> pd.DataFrame:
    """
    获取单只股票行情，优先读取本地cache，无缓存则调用tushare接口
    :param token: tushare token
    :param code: 股票ts_code
    :param start: YYYYMMDD 起始时间
    :param end: YYYYMMDD 结束时间
    :param refresh_cache: bool True强制重新拉取不读缓存
    :return: pd.DataFrame 清洗完成的日线行情
    """
    if not token:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            token = input('请粘贴你的tushare token：').strip()
            if not token:
                raise RuntimeError("token不能为空，请配置.env或者输入有效token")
    ts.set_token(token)
    cache_file = f"cache_{code.replace('.','_')}_{ADJ_MODE}_{start}_{end}.csv"
    if (not refresh_cache) and os.path.exists(cache_file):
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
        logger.info("本地缓存不存在/强制刷新，调用Tushare接口拉取后复权数据...")
        df_raw = ts_pro_bar_with_retry(code, adj=ADJ_MODE, start_date=start, end_date=end)
        if df_raw is None or df_raw.empty:
            logger.warning(f"股票 {code} 获取数据集为空")
            raise RuntimeError("数据集为空，请检查股票代码、时间区间、Tushare权限")
        required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        miss_cols = [c for c in required_cols if c not in df_raw.columns]
        if miss_cols:
            raise RuntimeError(f"Tushare返回数据缺少必要字段：{','.join(miss_cols)}，接口返回异常")
        valid_ratio = (~df_raw[["open", "high", "low", "close", "vol"]].isna()).all(axis=1).mean()
        if valid_ratio < 0.3:
            raise RuntimeError(f"行情数据大量缺失，有效行占比仅{valid_ratio:.1%}，无法继续回测")
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
        df_raw_save = df_raw.copy()
        df_raw_save["trade_date"] = df_raw_save["trade_date"].dt.strftime("%Y%m%d")
        df_raw_save.to_csv(cache_file, index=False, encoding="utf-8-sig")
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 80:
        raise RuntimeError(f"有效日线数据行数不足，当前仅 {len(df)} 条，至少需要80条")
    logger.info(f"成功加载 {code} 日线数据，有效行数 {len(df)}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据原始日线计算全套技术因子与训练标签
    注意：shift(-1)未来函数，仅离线回测，实盘不可直接使用
    :param df: 原始日线DataFrame
    :return: 增加因子、label后的DataFrame
    """
    df = df.copy()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['close_ma5_ratio'] = df['close'] / df['ma5']
    df['close_ma20_ratio'] = df['close'] / df['ma20']
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    df['rsi14'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_dif'] - df['macd_dea']
    df['vol_change'] = df['vol'].pct_change()
    df['vol_ma5'] = df['vol'].rolling(5).mean()
    df['vol_ratio'] = df['vol'] / df['vol_ma5']
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['volatility'] = df['ret_1d'].rolling(14).std()
    df['label'] = (df['close'].shift(-1) > df['close']).astype(np.int64)
    up_rate = df['label'].mean()
    logger.info(f"样本标签分布：上涨 {up_rate:.2%}，下跌 {1-up_rate:.2%}")
    return df


def calc_metrics(net_value: pd.Series, rf: float):
    """
    回测绩效指标计算：总收益、年化收益、夏普比率、最大回撤
    :param net_value: pd.Series 归一化净值序列
    :param rf: float 无风险年利率
    :return: dict 指标字典 total_return / annual_return / sharpe / max_drawdown
    """
    net_value = net_value.dropna()
    if len(net_value) <= 1:
        logger.warning("⚠️净值序列有效数据过少，指标全部置空")
        return {"total_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    ret = net_value.pct_change().dropna()
    total_return = net_value.iloc[-1] / net_value.iloc[0] - 1
    years = len(ret) / 252
    if years <= 0:
        logger.warning("⚠️交易年数计算为0，指标全部置空")
        return {"total_return": np.nan, "annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
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


def ml_backtest(df: pd.DataFrame, train_ratio=0.8, prob_threshold=0.5, n_estimators=200, max_depth=5):
    """
    随机森林向量化回测主逻辑，时序划分+TimeSeriesSplit时序交叉验证
    :param df: 特征+标签完整数据集
    :param train_ratio: float 训练集划分比例
    :param prob_threshold: float 持仓概率阈值
    :param n_estimators: int 树数量
    :param max_depth: int 树最大深度
    :return: test_df, importance, cv_mean, cv_std, up_rate, train_acc, test_acc
    """
    feature_cols = FEATURE_COLS
    split_idx = int(len(df) * train_ratio)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()
    logger.info(f"训练集时间：{train_raw['trade_date'].min().date()} ~ {train_raw['trade_date'].max().date()}")
    logger.info(f"测试集时间：{test_raw['trade_date'].min().date()} ~ {test_raw['trade_date'].max().date()}")
    train = train_raw.dropna(subset=feature_cols + ["label"])
    test = test_raw.dropna(subset=feature_cols + ["label"])
    if len(train) < 20 or len(test) < 10:
        raise RuntimeError(f"样本不足！dropna后训练集{len(train)}，测试集{len(test)}，训练≥20、测试≥10")
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]
    max_cv_split = 5
    n_splits = min(max_cv_split, max(2, len(X_train) // 20))
    logger.info(f"动态设置时序交叉验证折数 n_splits = {n_splits}")
    logger.info("===== 时间序列交叉验证 =====")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    fold_model = None
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
        fold_model = None
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X_train.iloc[tr_idx])
        X_va = scaler_fold.transform(X_train.iloc[va_idx])
        fold_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED)
        fold_model.fit(X_tr, y_train.iloc[tr_idx])
        score = fold_model.score(X_train.iloc[va_idx], y_train.iloc[va_idx])
        cv_scores.append(score)
        logger.info(f"第{fold}折 验证集准确率: {score:.4f}")
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"交叉验证平均准确率: {cv_mean:.4f} ± {cv_std:.4f}")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED)
    model.fit(X_train_s, y_train)
    train_acc = model.score(X_train_s, y_train)
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n训练集准确率: {train_acc:.4f}")
    logger.info(f"测试集准确率: {test_acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    overfit_gap = train_acc - cv_mean
    if overfit_gap > 0.12:
        logger.warning(f"⚠️过拟合风险提示：训练集准确率‑CV平均准确率 = {overfit_gap:.4f}，差距偏大，注意调参")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\n特征重要性Top5:\n{importance.head()}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["下跌", "上涨"])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm)
    cm_filename = os.path.join(OUTPUT_DIR, f"ml_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        plt.savefig(cm_filename, dpi=120)
        logger.info(f"✅混淆矩阵图片已保存 {cm_filename}")
    except Exception as e:
        logger.warning(f"⚠️混淆矩阵图片保存失败，跳过，err:{str(e)}")
    plt.close(fig_cm)
    test = test.copy()
    test.loc[:, "prob_up"] = y_pred_proba
    test.loc[:, "signal"] = y_pred
    test.loc[:, "trade"] = test["signal"].diff().abs()
    # 空值防护，ret_1d为NaN填充0，防止cumprod全部变成NaN
    test["ret_1d"] = test["ret_1d"].fillna(0.0)
    cost_per_switch = 2 * COST_RATE + SLIPPAGE
    test.loc[:, "strategy_ret"] = test["signal"].shift(1) * test["ret_1d"] - test["trade"].shift(1) * cost_per_switch
    test.loc[test.index[0], "strategy_ret"] = 0.0
    test.loc[:, "strategy_net"] = (1 + test["strategy_ret"]).cumprod()
    test.loc[:, "buyhold_net"] = (1 + test["ret_1d"].fillna(0.0)).cumprod()
    trade_cnt = test["trade"].sum()
    logger.info(f"样本外区间总交易次数：{int(trade_cnt)}")
    up_rate = df["label"].mean()
    return test, importance, cv_mean, cv_std, up_rate, train_acc, test_acc


def plot_result(test_df: pd.DataFrame, importance: pd.Series, stock_code: str, no_plot=False):
    """
    绘制向量化回测结果：净值对比图、买卖信号标记、特征重要性柱状图
    :param test_df: 测试集结果DataFrame
    :param importance: pd.Series 特征重要性
    :param stock_code: str 股票代码
    :param no_plot: bool 是否关闭绘图
    """
    if no_plot:
        logger.info("📌 --no_plot已开启，跳过绘图")
        return
    test_df = test_df.copy()
    date_list = test_df["trade_date"].dt.to_pydatetime()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(date_list, test_df["strategy_net"], label="机器学习策略", linewidth=1.5)
    axes[0].plot(date_list, test_df["buyhold_net"], label="买入持有", alpha=0.7)
    axes[0].set_title(f"{stock_code}:机器学习策略 VS 买入持有（样本外测试集）")
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
    filename = os.path.join(OUTPUT_DIR, f"ml_result_{stock_code.replace('.','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        plt.savefig(filename, dpi=150)
        logger.info(f"回测净值图表保存至：{filename}")
    except Exception as e:
        logger.warning(f"⚠️净值图表保存失败，跳过绘图，err:{str(e)}")
    plt.close(fig)


def main():
    """程序入口，解析命令行参数，执行随机森林向量化回测"""
    parser = argparse.ArgumentParser(
        description="随机森林股票涨跌预测回测程序，本代码仅学习演示，不构成任何投资建议"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--code", type=str, default=None, help="股票ts代码，不传进入交互输入")
    parser.add_argument("--start", type=str, default="20200101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--token", type=str, default="", help="Tushare token")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="训练集占比 (0,1)")
    parser.add_argument("--prob_threshold", type=float, default=PROB_THRESHOLD, help="持仓概率阈值0‑1")
    parser.add_argument("--n_estimators", type=int, default=200, help="随机森林树数量")
    parser.add_argument("--max_depth", type=int, default=5, help="树最大深度")
    parser.add_argument("--refresh_cache", action="store_true", help="强制刷新行情缓存，重新拉取数据")
    parser.add_argument("--clear_cache", action="store_true", help="清除cache_*.csv行情缓存")
    parser.add_argument("--clean_output", action="store_true", help="清理输出报告图片日志，保留缓存")
    parser.add_argument("--no_plot", action="store_true", help="关闭绘图，适合服务器环境")
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
    if not (0 < args.train_ratio < 1):
        raise RuntimeError("train_ratio 必须介于0‑1之间")
    if not (0 <= args.prob_threshold <= 1):
        raise RuntimeError("prob_threshold 必须介于0‑1之间")
    if args.n_estimators < 1 or args.max_depth < 1:
        raise RuntimeError("n_estimators、max_depth必须≥1")
    if args.n_estimators > 1000:
        logger.warning("⚠️n_estimators设置过大，建议不要超过1000，容易内存溢出")
    if args.max_depth > 15:
        logger.warning("⚠️max_depth设置过大，模型容易过拟合")
    if abs(args.prob_threshold - 1.0) < 1e-6:
        logger.warning("⚠️prob_threshold=1.0，阈值设置为1，几乎不会产生任何持仓信号")
    if abs(args.prob_threshold) < 1e-6:
        logger.warning("⚠️prob_threshold=0.0，阈值设置为0，几乎全程保持持仓")
    if args.prob_threshold < 0.4:
        logger.warning("⚠️概率阈值设置过低，容易产生大量频繁交易")
    if args.prob_threshold > 0.6:
        logger.warning("⚠️概率阈值设置过高，可能几乎没有持仓信号")

    logger.info("========== 程序启动参数 ==========")
    logger.info(f"脚本版本: {__version__}")
    for k, v in vars(args).items():
        logger.info(f"{k:20s}: {v}")
    logger.info("==================================")

    shell_cmd_parts = ["python", "ml_strategy.py"]
    if args.code:
        shell_cmd_parts.extend(["--code", args.code])
    shell_cmd_parts.extend(["--start", args.start, "--end", args.end])
    if args.token:
        shell_cmd_parts.extend(["--token", args.token])
    shell_cmd_parts.extend(["--train_ratio", str(args.train_ratio)])
    shell_cmd_parts.extend(["--prob_threshold", str(args.prob_threshold)])
    shell_cmd_parts.extend(["--n_estimators", str(args.n_estimators)])
    shell_cmd_parts.extend(["--max_depth", str(args.max_depth)])
    if args.refresh_cache:
        shell_cmd_parts.append("--refresh_cache")
    if args.no_plot:
        shell_cmd_parts.append("--no_plot")
    run_shell_cmd = safe_shell_join(shell_cmd_parts)

    default_code = "600519.SH"
    if args.code is not None:
        stock_code = args.code.upper()
        logger.info(f"💻命令行指定股票：{stock_code}")
    else:
        print("-" * 60)
        user_input = input(f"请输入股票代码(回车使用默认 {default_code})：").strip()
        stock_code = user_input.upper() if user_input else default_code

    logger.info(f"回测标的 {stock_code}，时间区间 {args.start} ~ {args.end}")
    try:
        df_raw = fetch_data(token=args.token, code=stock_code, start=args.start, end=args.end, refresh_cache=args.refresh_cache)
        df_feat = build_features(df_raw)
        test_df, imp, cv_mean, cv_std, up_rate, train_acc, test_acc = ml_backtest(
            df_feat,
            train_ratio=args.train_ratio,
            prob_threshold=args.prob_threshold,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
        ml_metric = calc_metrics(test_df["strategy_net"], RISK_FREE_RATE)
        bh_metric = calc_metrics(test_df["buyhold_net"], RISK_FREE_RATE)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_name = os.path.join(OUTPUT_DIR, f"ml_report_{stock_code.replace('.','_')}_{timestamp_str}.csv")
        export_cols = ["trade_date", "close", "prob_up", "signal", "strategy_ret", "strategy_net", "buyhold_net"]
        test_df[export_cols].to_csv(csv_name, index=False, encoding="utf‑8‑sig")
        logger.info(f"✅回测明细CSV导出 {csv_name}")
        txt_name = os.path.join(OUTPUT_DIR, f"ml_summary_{stock_code.replace('.','_')}_{timestamp_str}.txt")
        with open(txt_name, "w", encoding="utf‑8") as f:
            f.write("⚠️本结果仅用于学习演示，不构成任何投资建议。历史回测不能代表未来收益。\n")
            f.write("⚠️注意：本向量化回测未模拟A股100股最小交易单位；严谨评估请使用event_backtest.py事件驱动模块\n")
            f.write(f"脚本名称: ml_strategy.py【随机森林简化收益率回测】\n")
            f.write(f"脚本版本: {__version__}\n")
            f.write(f"本地运行时间: {local_run_time}\n")
            f.write(f"股票代码:{stock_code}\n")
            f.write(f"时间区间:{args.start}‑{args.end}\n")
            f.write(f"复权模式:{ADJ_MODE}\n")
            f.write(f"训练集比例:{args.train_ratio},概率阈值:{args.prob_threshold}\n")
            f.write(f"n_estimators:{args.n_estimators},max_depth:{args.max_depth}\n")
            f.write(f"--refresh_cache: {args.refresh_cache}\n")
            f.write(f"--no_plot: {args.no_plot}\n")
            f.write(f"完整启动参数: {vars(args)}\n")
            f.write(f"可复现运行命令:\n{run_shell_cmd}\n")
            f.write(f"标签上涨占比:{up_rate:.2%}\n")
            f.write(f"训练集准确率:{train_acc:.4f}\n")
            f.write(f"时序交叉验证平均准确率:{cv_mean:.4f} ± {cv_std:.4f}\n")
            f.write(f"测试集准确率:{test_acc:.4f}\n")
            f.write(f"【机器学习策略】总收益率:{ml_metric['total_return']:.2%} 年化:{ml_metric['annual_return']:.2%} 夏普:{ml_metric['sharpe']:.3f} 最大回撤:{ml_metric['max_drawdown']:.2%}\n")
            f.write(f"【买入持有】总收益率:{bh_metric['total_return']:.2%} 年化:{bh_metric['annual_return']:.2%} 夏普:{bh_metric['sharpe']:.3f} 最大回撤:{bh_metric['max_drawdown']:.2%}\n")
            f.write("# End of report\n")
        logger.info(f"✅汇总报告txt导出 {txt_name}")
        logger.info("\n" + "=" * 60)
        logger.info(f"{'指标':<16}{'机器学习策略':>14}{'买入持有':>14}")
        logger.info("-" * 60)
        logger.info(f"总收益率    {ml_metric['total_return']:>13.2%}{bh_metric['total_return']:>13.2%}")
        logger.info(f"年化收益率   {ml_metric['annual_return']:>13.2%}{bh_metric['annual_return']:>13.2%}")
        logger.info(f"夏普比率    {ml_metric['sharpe']:>13.3f}{bh_metric['sharpe']:>13.3f}")
        logger.info(f"最大回撤    {ml_metric['max_drawdown']:>13.2%}{bh_metric['max_drawdown']:>13.2%}")
        logger.info("=" * 60)
        plot_result(test_df, imp, stock_code, no_plot=args.no_plot)
        logger.info("✅全部回测流程执行完成")
    except KeyboardInterrupt:
        logger.info("\n👋 用户手动 Ctrl+C 终止程序，正常退出")
    except MemoryError:
        logger.error("❌内存不足，请缩小回测时间区间，减少行情数据量")
    except Exception as e:
        logger.error(f"程序异常终止：{str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 用户手动 Ctrl+C 终止程序，正常退出")
    except MemoryError:
        logger.error("❌内存不足，请缩小回测区间")
    except Exception as e:
        logger.error(f"顶层程序异常终止：{str(e)}", exc_info=True)
