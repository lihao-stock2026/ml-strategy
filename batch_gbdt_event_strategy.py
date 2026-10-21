"""
batch_gbdt_event_strategy.py v1.5.5【MySQL全链路持久化｜双数据源tushare/akshare】
批量多股票 GBDT(LightGBM/XGBoost可选) + 事件驱动回测
更新：
1. 订单生成失败打印资金不足警告（提示可用现金、1手预估成本）
2. 回测结束如果0笔交易输出告警，提示排查资金/阈值，跳过绘图避免报错
3. 绘图使用merge按日期对齐，解决数组长度不一致绘图错位直线问题
4. 修复calc_metrics返回net_series但未写入equity_df，消除KeyError
5.train_predict增加模型fit异常捕获，打印调试上下文
6.打印每只股票特征dropna前后样本数量，便于定位样本丢失
7.汇总csv新增fail_reason失败原因字段
8.GbdtWrapStrategy增加reset()，每只股票回测重置持仓，杜绝多股票状态污染
9.XGBoost自动根据训练集正负样本比例计算scale_pos_weight，对齐LGB样本不均衡
10.训练输出precision/recall/f1‑score，不再仅依赖accuracy
11.全部输出统一output文件夹，自动创建
12.【MySQL优化】单只股票只打开一次数据库连接，全部读写完成finally统一关闭
13.feat:新增AKShare免费数据源，命令行切换tushare/akshare；akshare延迟导入
14.fix:聚类预处理跟随--data‑source参数，双数据源都支持聚类实验
15.feat:训练完成自动输出LGB/XGB特征重要性图，按股票代码命名保存至images目录；--no‑plot统一关闭全部绘图输出
⚠️风险免责：本代码仅用于学习演示，不构成任何投资建议。历史回测结果不能代表未来实际收益。
【运行示例】
python batch_gbdt_event_strategy.py --codes 600519.SH --start 20200101 --end 20241231 --model lgb
python batch_gbdt_event_strategy.py --codes 600519.SH --start 20200101 --end 20241231 --model xgb --prob-threshold 0.30 --sell-threshold 0.40
# 使用免费akshare
python batch_gbdt_event_strategy.py --codes 600519.SH --start 20200101 --end 20241231 --model lgb --data-source akshare
# akshare + 聚类
python batch_gbdt_event_strategy.py --codes 600519.SH --start 20200101 --end 20241231 --model lgb --data-source akshare --enable_cluster --n_cluster 2
"""
import os
import glob
import argparse
import time
import shlex
import random
import logging
from datetime import datetime
from dotenv import load_dotenv
import tushare as ts
import pandas as pd
from pandas import errors
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from event_backtest import (
    Portfolio, ExecutionHandler,
    Event, MarketEvent, SignalEvent, OrderEvent, FillEvent,
    calc_metrics, DataHandler
)
from db_utils import StockDB

__version__ = "1.5.5"
MAX_BATCH_STOCKS = 20
API_MIN_INTERVAL = 0.4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局随机种子，保证实验可复现
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

load_dotenv()

# 多操作系统兼容中文字体：Windows / MacOS / Linux
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def setup_logger():
    """
    初始化日志对象，同时输出到控制台与本地日志文件batch_gbdt.log
    :return: logging.Logger 配置完成的logger实例
    """
    log_path = os.path.join(OUTPUT_DIR, "batch_gbdt.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding="utf‑8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logger()

if importlib is not None:
    try:
        tushare_ver = importlib.metadata.version("tushare")
        ver_major, ver_minor, ver_patch = map(int, tushare_ver.split(".")[:3])
        if (ver_major, ver_minor, ver_patch) < (1, 2, 89):
            logger.warning(f"⚠️警告：检测到tushare版本 {tushare_ver} 过低，建议执行：pip install tushare --upgrade")
    except Exception:
        logger.warning("⚠️警告：无法获取tushare版本，请确认tushare>=1.2.89已正确安装")
else:
    logger.warning("⚠️当前Python版本较低，跳过tushare版本检测，请手动确认tushare>=1.2.89")

# --------------------------全局业务常量，全部收拢在此处--------------------------
RISK_FREE_RATE = 0.02
COST_RATE = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.0005
MAX_RETRY = 3
ADJ_MODE = "hfq"
TRAIN_RATIO = 0.8
# 默认阈值，可通过命令行参数覆盖
DEFAULT_PROB_THRESHOLD = 0.35
DEFAULT_SELL_THRESHOLD = 0.45
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": SEED,
    "is_unbalance": True
}
FEATURE_COLS = [
    'close_ma5_ratio', 'close_ma20_ratio',
    'rsi14', 'macd_dif', 'macd_dea', 'macd_hist',
    'vol_change', 'vol_ratio',
    'ret_1d', 'ret_5d', 'ret_10d', 'volatility'
]
# --------------------------------------------------------------------------------


class GbdtWrapStrategy:
    """
    GBDT机器学习信号封装策略类
    根据模型输出上涨/下跌概率生成交易SignalEvent
    :param signal_df: DataFrame，包含trade_date, prob_up, prob_down
    :param prob_threshold: float，买入上涨概率阈值
    :param sell_threshold: float，卖出下跌概率阈值
    """
    def __init__(self, signal_df, prob_threshold, sell_threshold):
        self.df = signal_df.set_index("trade_date")
        self.in_position = False
        self.prob_threshold = prob_threshold
        self.sell_threshold = sell_threshold

    def reset(self):
        """每只股票回测前调用，重置持仓状态，避免多股票状态互相污染"""
        self.in_position = False

    def calculate_signal(self, bar: MarketEvent):
        """
        根据单根K线MarketEvent计算交易信号
        :param bar: MarketEvent 对象，当日行情bar
        :return: SignalEvent | None，有信号返回对象，无信号返回None
        """
        if bar.date not in self.df.index:
            return None
        p_up = self.df.loc[bar.date, "prob_up"]
        p_down = self.df.loc[bar.date, "prob_down"]
        if pd.isna(p_up) or pd.isna(p_down):
            return None
        p_up = float(p_up)
        p_down = float(p_down)
        if not self.in_position:
            if p_up >= self.prob_threshold:
                self.in_position = True
                return SignalEvent(bar.date, 1)
        else:
            if p_down >= self.sell_threshold:
                self.in_position = False
                return SignalEvent(bar.date, -1)
        return None


def clear_all_cache():
    """清除本地所有cache_*.csv行情缓存文件"""
    pattern = os.path.join(BASE_DIR, "cache_*.csv")
    cache_files = glob.glob(pattern)
    for f in cache_files:
        os.remove(f)
    logging.info(f"✅已清理行情缓存，共删除 {len(cache_files)} 个")


def clean_all_output():
    """清理批量回测输出产物：图片、csv、txt、日志，保留行情cache缓存"""
    patterns = [
        os.path.join(OUTPUT_DIR, "batch_*.png"),
        os.path.join(OUTPUT_DIR, "batch_*.csv"),
        os.path.join(OUTPUT_DIR, "batch_*.txt"),
        os.path.join(OUTPUT_DIR, "batch_batch_summary*.csv"),
        os.path.join(OUTPUT_DIR, "batch_gbdt.log"),
        os.path.join(BASE_DIR, "images", "*feature_importance_*.png")
    ]
    cnt = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            cnt += 1
    logger.info(f"🧹清理批量输出产物 {cnt} 个；行情cache缓存保留")


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
                logger.warning(f"Tushare返回空数据 code={code}")
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
            token = input("粘贴你的tushare token：").strip()
            if not token:
                raise RuntimeError("token不能为空，请配置.env或者输入有效token")
    ts.set_token(token)
    cache_name = f"cache_{code.replace('.', '_')}_{ADJ_MODE}_{start}_{end}.csv"
    cache_file = os.path.join(BASE_DIR, cache_name)
    if (not refresh_cache) and os.path.exists(cache_file):
        logger.info(f"读取缓存 {cache_file}")
        try:
            df_raw = pd.read_csv(cache_file, encoding="utf‑8‑sig").reset_index(drop=True)
        except errors.EmptyDataError:
            logger.warning(f"⚠️缓存文件损坏 {cache_file}，删除损坏缓存，重新拉取")
            os.remove(cache_file)
            df_raw = None
        if df_raw is not None:
            df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
    else:
        df_raw = None
    if df_raw is None:
        logger.info(f"拉取 {code} 行情数据")
        df_raw = ts_pro_bar_with_retry(code, adj=ADJ_MODE, start_date=start, end_date=end)
        if df_raw is None or df_raw.empty:
            raise RuntimeError(f"{code} 获取数据为空")
        required_cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
        miss_cols = [c for c in required_cols if c not in df_raw.columns]
        if miss_cols:
            raise RuntimeError(f"返回数据缺失字段 {','.join(miss_cols)}")
        valid_ratio = (~df_raw[["open", "high", "low", "close", "vol"]].isna()).all(axis=1).mean()
        if valid_ratio < 0.3:
            raise RuntimeError(f"行情大量缺失，有效占比{valid_ratio:.1%}")
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
        df_raw_save = df_raw.copy()
        df_raw_save["trade_date"] = df_raw_save["trade_date"].dt.strftime("%Y%m%d")
        df_raw_save.to_csv(cache_file, index=False, encoding="utf‑8‑sig")
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 80:
        raise RuntimeError(f"{code}有效K线{len(df)}条，至少需要80条")
    return df


def fetch_data_akshare(code: str, start: str, end: str, refresh_cache: bool = False) -> pd.DataFrame:
    """
    使用AKShare获取A股日线，免费无积分，输出格式对齐Tushare pro_bar
    :param code: ts_code格式 "600519.SH"
    :param start: YYYYMMDD
    :param end: YYYYMMDD
    :param refresh_cache: 是否强制刷新缓存
    :return: 和tushare输出结构完全一致的DataFrame
    """
    import akshare as ak
    cache_name = f"cache_ak_{code.replace('.', '_')}_{start}_{end}.csv"
    cache_file = os.path.join(BASE_DIR, cache_name)
    if not refresh_cache and os.path.exists(cache_file):
        logger.info(f"读取AKShare本地缓存 {cache_file}")
        df_raw = pd.read_csv(cache_file, encoding="utf‑8‑sig")
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"].astype(str), format="%Y%m%d")
    else:
        sym, exchange = code.split(".")
        ak_code = exchange.lower() + sym
        logger.info(f"AKShare拉取行情 {ak_code}")
        df_raw = ak.stock_zh_a_daily(symbol=ak_code, adjust="hfq")
        df_raw = df_raw.rename(columns={
            "date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "vol"
        })
        df_raw["ts_code"] = code
        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"])
        df_raw = df_raw.sort_values("trade_date").reset_index(drop=True)
        df_raw_save = df_raw.copy()
        df_raw_save["trade_date"] = df_raw_save["trade_date"].dt.strftime("%Y%m%d")
        df_raw_save.to_csv(cache_file, index=False, encoding="utf‑8‑sig")
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt = pd.to_datetime(end, format="%Y%m%d")
    df = df_raw[(df_raw["trade_date"] >= start_dt) & (df_raw["trade_date"] <= end_dt)].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    df = df.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    df = df.sort_values("trade_date").reset_index(drop=True)
    if len(df) < 80:
        raise RuntimeError(f"{code}有效K线{len(df)}条，至少需要80条")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据原始日线计算全套技术因子与训练标签
    注意：使用未来窗口价格构造标签，仅离线回测使用，实盘禁止使用该标签
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

    # ----------------------
    # 【当前启用：原始标签】未来第5日收盘价大于当日，标记上涨样本
    # 样本分布相对均衡，适合网格搜索观察过拟合；仅离线回测，存在未来函数风险
    # ----------------------
    df['label'] = (df['close'].shift(-5) > df['close']).astype(np.int64)

    # ----------------------
    # 【备选实验标签，已注释】未来5日区间最大涨幅大于3%标记上涨样本
    # 该标签会造成正样本数量极少，模型输出prob_up普遍偏低，极易出现零交易；仅用于对比实验
    # ----------------------
    # def calc_future_max_high(series, window=5):
    #     out = np.full(len(series), np.nan)
    #     for i in range(len(series) - window):
    #         out[i] = series.iloc[i:i + window].max()
    #     return out
    # df['future_high_5'] = calc_future_max_high(df['high'], window=5)
    # df['future_return_5'] = df['future_high_5'] / df['close'] - 1
    # df['label'] = (df['future_return_5'] > 0.03).astype(np.int64)
    # df.drop(columns=["future_high_5", "future_return_5"], inplace=True)

    return df


def train_predict(df: pd.DataFrame, model_type: str, params: dict, stock_code: str, no_plot: bool):
    """
    时序划分数据集，TimeSeriesSplit时序交叉验证，训练lgb/xgb并输出涨跌概率
    :param df: 带特征与label的数据集
    :param model_type: str "lgb" / "xgb"
    :param params: dict 模型超参
    :param stock_code: str 当前回测股票代码，用于特征重要性图片文件名
    :param no_plot: bool 是否关闭全部绘图输出
    :return: df_out, cv_mean, cv_std, test_acc
        df_out:原始df新增prob_up/prob_down
        cv_mean:交叉验证平均acc
        cv_std:交叉验证标准差
        test_acc:测试集准确率
    """
    feature_cols = FEATURE_COLS
    split_idx = int(len(df) * TRAIN_RATIO)
    train_raw = df.iloc[:split_idx].copy()
    test_raw = df.iloc[split_idx:].copy()
    train_before_dropna = len(train_raw)
    test_before_dropna = len(test_raw)
    train = train_raw.dropna(subset=feature_cols + ["label"])
    test = test_raw.dropna(subset=feature_cols + ["label"])
    logger.info(f"样本过滤：训练集dropna前:{train_before_dropna},后:{len(train)}；测试集dropna前:{test_before_dropna},后:{len(test)}")
    if len(train) < 20 or len(test) < 10:
        raise RuntimeError(f"样本不足，训练:{len(train)} 测试:{len(test)}")
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]
    pos_cnt = int(np.sum(y_train == 1))
    neg_cnt = int(np.sum(y_train == 0))
    logger.info(f"训练集标签分布：正样本{pos_cnt},负样本{neg_cnt}")
    scale_pos_weight_val = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    fold_model = None
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
        fold_model = None
        if model_type == "lgb":
            fold_model = lgb.LGBMClassifier(**params)
        elif model_type == "xgb":
            fold_params = {k: v for k, v in params.items() if k != "is_unbalance"}
            fold_params["scale_pos_weight"] = scale_pos_weight_val
            fold_model = XGBClassifier(**fold_params)
        try:
            fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        except Exception as e:
            logger.error(f"fold{fold}训练异常：{str(e)}", exc_info=True)
            raise RuntimeError(f"fold{fold}模型训练失败:{str(e)}")
        score = fold_model.score(X_train.iloc[va_idx], y_train.iloc[va_idx])
        cv_scores.append(score)
        logger.info(f"fold{fold} cv acc:{score:.4f}")
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"cv平均acc {cv_mean:.4f} ±{cv_std:.4f}")
    try:
        if model_type == "lgb":
            model = lgb.LGBMClassifier(**params)
        elif model_type == "xgb":
            final_params = {k: v for k, v in params.items() if k != "is_unbalance"}
            final_params["scale_pos_weight"] = scale_pos_weight_val
            model = XGBClassifier(**final_params)
        model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"主模型训练失败:{str(e)}", exc_info=True)
        raise RuntimeError(f"主模型训练异常:{str(e)}")

    # ========== LGB / XGB 特征重要性绘图 ==========
    if not no_plot:
        img_save_dir = os.path.join(BASE_DIR, "images")
        os.makedirs(img_save_dir, exist_ok=True)
        feature_names = feature_cols
        try:
            importance = model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance
            }).sort_values("importance", ascending=True)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(imp_df["feature"], imp_df["importance"])
            if model_type == "lgb":
                ax.set_title("LightGBM 特征重要性")
            else:
                ax.set_title("XGBoost 特征重要性")
            ax.set_xlabel("重要性得分")
            plt.tight_layout()
            imp_img_path = os.path.join(img_save_dir, f"{model_type}_feature_importance_{stock_code.replace('.','_')}.png")
            plt.savefig(imp_img_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"✅特征重要性图已保存 {imp_img_path}")
        except Exception as imp_err:
            logger.warning(f"⚠️特征重要性绘图跳过，err:{str(imp_err)}")
    # =================================================

    y_pred_proba_full = model.predict_proba(X_test)
    y_pred_proba_up = y_pred_proba_full[:, 1]
    y_pred_proba_down = y_pred_proba_full[:, 0]
    y_pred_cls = (y_pred_proba_up >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_cls)
    logger.info(f"测试集acc:{acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred_cls, zero_division=0))

    test = test.copy()
    test["prob_up"] = y_pred_proba_up
    test["prob_down"] = y_pred_proba_down
    df_out = df.copy()
    df_out = df_out.merge(test[["trade_date", "prob_up", "prob_down"]], how="left", on="trade_date")
    return df_out, cv_mean, cv_std, acc


def run_single_stock(token, stock_code, start, end, initial_capital, no_plot,
                     model_type: str, model_params: dict,
                     prob_threshold: float, sell_threshold: float,
                     cluster_id: int, data_source: str, refresh_cache: bool, df_feat_arg=None):
    """
    单只股票完整回测流水线：拉取数据→因子计算→入库→模型训练→事件驱动回测→回测结果入库
    :param token: tushare token
    :param stock_code: 股票代码
    :param start: YYYYMMDD
    :param end: YYYYMMDD
    :param initial_capital: 初始回测资金
    :param no_plot: bool 是否关闭绘图
    :param model_type: lgb/xgb
    :param model_params: 模型参数字典
    :param prob_threshold:买入概率阈值
    :param sell_threshold:卖出概率阈值
    :param cluster_id:聚类编号，无聚类为-1
    :param data_source:数据源 tushare / akshare
    :param refresh_cache: bool 是否强制刷新行情缓存
    :param df_feat_arg: 传入预处理好的特征df，为批量聚类复用缓存；None则现场拉取计算
    :return: dict 单只股票回测结果汇总字典
    """
    stock_code = stock_code.upper()
    if df_feat_arg is None:
        if data_source == "tushare":
            df_raw = fetch_data(token, stock_code, start, end, refresh_cache=refresh_cache)
        elif data_source == "akshare":
            df_raw = fetch_data_akshare(stock_code, start, end, refresh_cache=refresh_cache)
        df_feat = build_features(df_raw)
    else:
        df_raw = df_feat_arg.copy()
        df_feat = df_feat_arg

    # =========第一处：写入日线、因子，操作完立刻关闭连接=========
    db1 = StockDB()
    try:
        if db1.connect():
            db1.insert_stock_daily(df_raw)
            logger.info(f"✅{stock_code} stock_daily 写入完成")
            db1.insert_stock_factor(df_feat)
            logger.info(f"✅{stock_code} stock_factor 写入完成")
    except Exception as e:
        import traceback
        logger.warning(f"⚠️ {stock_code} 日线/因子入库失败，继续回测，err:{str(e)}")
        logger.warning(traceback.format_exc())
    finally:
        db1.close()

    df_signal, cv_mean, cv_std, test_acc = train_predict(df_feat, model_type, model_params, stock_code, no_plot)
    df_backtest = df_signal.copy()
    strategy = GbdtWrapStrategy(df_backtest[["trade_date", "prob_up", "prob_down"]],
                                prob_threshold=prob_threshold,
                                sell_threshold=sell_threshold)
    strategy.reset()
    data_handler = DataHandler(df_backtest)
    portfolio = Portfolio(initial_capital=initial_capital)
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
            if order is None:
                cash_now = portfolio.cash
                min_lot_cost = bar.open_ * 100 * (1 + COST_RATE + SLIPPAGE)
                logger.warning(
                    f"⚠️【订单被拒绝】日期:{bar.date}, 开盘价:{bar.open_:.2f} "
                    f"可用现金:{cash_now:.2f}, 买入1手最少需要:{min_lot_cost:.2f} "
                    f"初始资金设置过小或者触发信号时股价太高，无法开仓！"
                )
            else:
                pending_order = order
    equity_df = pd.DataFrame(portfolio.equity_curve)
    trade_records = portfolio.trade_records
    if len(trade_records) == 0:
        logger.warning("############################")
        logger.warning("⚠️ 警告：本次回测【0笔成交记录】！")
        logger.warning(
            "可能原因：①初始资金不足，买不起100股；②prob_threshold阈值太高，几乎没有买入信号；③预测集内没有触发信号")
        logger.warning("############################")
    total_ret, annual_ret, sharpe, max_dd, net_series = calc_metrics(equity_df, RISK_FREE_RATE)
    equity_df["net_series"] = net_series
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(OUTPUT_DIR, f"batch_report_{stock_code.replace('.', '_')}_{timestamp}.csv")
    df_signal.to_csv(csv_name, index=False, encoding="utf‑8‑sig")
    txt_name = os.path.join(OUTPUT_DIR, f"batch_summary_{stock_code.replace('.', '_')}_{timestamp}.txt")
    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("⚠️本结果仅用于学习演示，不构成任何投资建议。历史回测不能代表未来收益。\n")
        f.write(f"脚本名称:batch_gbdt_event_strategy.py【批量GBDT事件驱动回测】\n")
        f.write(f"脚本版本:{__version__}\n")
        f.write(f"本地运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"股票:{stock_code}\n")
        f.write(f"模型:{model_type}\n")
        f.write(f"prob_threshold:{prob_threshold}, sell_threshold:{sell_threshold}\n")
        f.write(f"cluster_id:{cluster_id}\n")
        f.write(f"时间区间:{start}-{end}\n")
        f.write(f"复权模式:{ADJ_MODE}\n")
        f.write(f"cv_mean_acc:{cv_mean:.4f} cv_std:{cv_std:.4f}\n")
        f.write(f"测试集acc:{test_acc:.4f}\n")
        f.write(f"总收益:{total_ret:.2%} 年化:{annual_ret:.2%} 夏普:{sharpe:.3f} 最大回撤:{max_dd:.2%}\n")
        f.write("# End of report\n")
    if not no_plot:
        if len(trade_records) == 0:
            logger.warning(f"{stock_code} 无任何成交记录，跳过绘图")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            bh_df = df_backtest[["trade_date", "close"]].copy()
            bh_df = bh_df.dropna(subset=["close"]).reset_index(drop=True)
            base_price = bh_df["close"].iloc[0]
            bh_df["bh_net"] = bh_df["close"] / base_price
            equity_df["trade_date"] = pd.to_datetime(equity_df["date"])
            plot_df = pd.merge(
                equity_df[["trade_date", "net_series"]],
                bh_df[["trade_date", "bh_net"]],
                on="trade_date",
                how="inner"
            )
            logger.info(f"合并后绘图数据集行数 {len(plot_df)}, base_price={base_price:.2f}")
            ax.plot(plot_df["trade_date"], plot_df["net_series"], label=f"{model_type}策略", lw=1.4)
            ax.plot(plot_df["trade_date"], plot_df["bh_net"], label="买入持有", alpha=0.7)
            ax.set_title(f"{stock_code} {model_type}-事件驱动回测净值 | cluster_id={cluster_id}")
            ax.legend()
            ax.grid(alpha=0.3)
            png_name = os.path.join(OUTPUT_DIR, f"batch_result_{stock_code.replace('.', '_')}_{timestamp}.png")
            try:
                plt.savefig(png_name, dpi=120)
                logger.info(f"图片保存 {png_name}")
            except Exception as e:
                logger.warning(f"⚠️绘图保存失败，跳过图片输出 err:{str(e)}")
            plt.close(fig)
    out_dict = {
        "stock_code": stock_code,
        "model_type": model_type,
        "cluster_id": cluster_id,
        "prob_threshold": prob_threshold,
        "sell_threshold": sell_threshold,
        "cv_mean_acc": cv_mean,
        "cv_std_acc": cv_std,
        "test_acc": test_acc,
        "total_return": total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trade_cnt": -1,
        "note": "ok",
        "fail_reason": ""
    }
    out_dict["trade_cnt"] = len(trade_records)
    # =========第二处：回测全部跑完，新建实例重新连接写回测结果=========
    db2 = StockDB()
    try:
        strategy_name = f"gbdt_{model_type}"
        if db2.connect():
            summary_dict = {
                "strategy_name": strategy_name,
                "start_date": start,
                "end_date": end,
                "initial_capital": initial_capital,
                "final_capital": float(equity_df["equity"].iloc[-1]),
                "total_return": float(total_ret if not np.isnan(total_ret) else 0.0),
                "sharpe_ratio": float(sharpe if not np.isnan(sharpe) else 0.0),
                "max_drawdown": float(max_dd if not np.isnan(max_dd) else 0.0),
                "win_rate": 0.0
            }
            db2.insert_backtest_summary(summary_dict)
            if len(trade_records) > 0:
                trade_df = pd.DataFrame(trade_records)
                trade_df["strategy_name"] = strategy_name
                trade_df["ts_code"] = stock_code
                trade_df = trade_df.rename(columns={
                    "quantity": "volume",
                    "stamp_tax": "tax"
                })
                trade_save_df = trade_df[
                    ["strategy_name", "ts_code", "trade_date", "direction", "price", "volume", "commission", "tax"]
                ].copy()
                db2.insert_backtest_trade(trade_save_df)
            logger.info(f"✅{stock_code} [{strategy_name}] 回测结果成功写入MySQL")
    except Exception as db_err:
        import traceback
        logger.warning(f"⚠️ {stock_code}写入回测结果失败，不影响回测主流程 err:{str(db_err)}")
        logger.warning(traceback.format_exc())
    finally:
        db2.close()
    out_dict["trade_cnt"] = len(trade_records)
    logger.info(f"✅{stock_code} 完成回测 {out_dict}")
    return out_dict


def main():
    """程序入口，解析命令行参数，执行批量股票回测逻辑"""
    parser = argparse.ArgumentParser(
        description="批量GBDT事件驱动回测脚本，支持lgb/xgb切换；可选KMeans股票聚类实验，本代码仅学习演示，不构成任何投资建议"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--codes", type=str, required=True, help="股票代码列表，逗号分割，例：600519.SH,000858.SZ")
    parser.add_argument("--start", type=str, default="20200101", help="起始YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束YYYYMMDD")
    parser.add_argument("--token", type=str, default="", help="tushare token")
    parser.add_argument("--capital", type=float, default=None, help="每只股票回测初始资金；不填则手动输入，回车默认300000")
    parser.add_argument("--model", type=str, choices=["lgb", "xgb"], default="lgb", help="选择模型 lgb / xgb")
    parser.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD, help="买入上涨概率阈值")
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD, help="卖出下跌概率阈值")
    parser.add_argument("--enable_cluster", action="store_true", help="开启KMeans股票聚类实验(仅训练集特征，防泄露)")
    parser.add_argument("--n_cluster", type=int, default=3, help="聚类簇数量，仅--enable_cluster生效")
    parser.add_argument("--refresh_cache", action="store_true", help="强制刷新行情缓存，重新拉取数据")
    parser.add_argument("--clear_cache", action="store_true", help="清空全部cache行情缓存")
    parser.add_argument("--clean_output", action="store_true", help="清理批量输出报告图片")
    parser.add_argument("--no_plot", action="store_true", help="关闭全部绘图输出")
    parser.add_argument("--data-source", type=str, choices=["tushare", "akshare"], default="tushare",
                        help="行情数据源：tushare(需要积分，默认)；akshare免费无积分")
    args = parser.parse_args()
    if args.capital is None:
        input_str = input("请输入回测初始资金，直接回车使用默认值300000：").strip()
        if input_str == "":
            initial_capital = 300000.0
        else:
            try:
                initial_capital = float(input_str)
            except ValueError:
                logger.warning("输入格式错误，将使用默认300000")
                initial_capital = 300000.0
    else:
        initial_capital = args.capital
    if args.clean_output:
        clean_all_output()
    if args.clear_cache:
        clear_all_cache()
    start_dt = pd.to_datetime(args.start, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end, format="%Y%m%d")
    delta_years = (end_dt - start_dt).days / 365.0
    if delta_years > 8:
        logger.warning(f"⚠️时间跨度 {delta_years:.1f}年，容易限流和内存溢出")
    for name, val in [("start", args.start), ("end", args.end)]:
        if not (val.isdigit() and len(val) == 8):
            raise RuntimeError(f"--{name}={val}格式错误，必须YYYYMMDD")
    if not (0.0 <= args.prob_threshold <= 1.0):
        raise RuntimeError("--prob-threshold必须0~1之间")
    if not (0.0 <= args.sell_threshold <= 1.0):
        raise RuntimeError("--sell-threshold必须0~1之间")
    shell_parts = ["python", "batch_gbdt_event_strategy.py"]
    shell_parts.extend(["--codes", args.codes])
    shell_parts.extend(["--start", args.start, "--end", args.end])
    if args.token:
        shell_parts.extend(["--token", args.token])
    shell_parts.extend(["--capital", str(initial_capital)])
    shell_parts.extend(["--model", args.model])
    shell_parts.extend(["--prob-threshold", str(args.prob_threshold)])
    shell_parts.extend(["--sell-threshold", str(args.sell_threshold)])
    if args.enable_cluster:
        shell_parts.append("--enable_cluster")
        shell_parts.extend(["--n_cluster", str(args.n_cluster)])
    if args.refresh_cache:
        shell_parts.append("--refresh_cache")
    if args.no_plot:
        shell_parts.append("--no_plot")
    run_cmd = safe_shell_join(shell_parts)
    logger.info("========批量GBDT回测启动【v1.5.5 MySQL全链路持久化｜双数据源】========")
    logger.info(f"脚本版本:{__version__}")
    logger.info(f"运行命令:{run_cmd}")
    logger.info(f"股票列表:{args.codes}")
    logger.info(f"选择模型:{args.model}")
    logger.info(f"prob_threshold={args.prob_threshold}, sell_threshold={args.sell_threshold}")
    logger.info(f"enable_cluster:{args.enable_cluster}  n_cluster:{args.n_cluster}")
    stock_list = [s.strip().upper() for s in args.codes.split(",")]
    stock_cnt = len(stock_list)
    if stock_cnt > MAX_BATCH_STOCKS:
        raise RuntimeError(f"批量股票最多{MAX_BATCH_STOCKS}只，减少股票数量规避限流")
    mem_df_cache = dict()
    stock_cluster_map = dict()
    use_cluster = False
    if args.enable_cluster:
        if stock_cnt <= args.n_cluster:
            logger.warning(f"⚠️股票数量({stock_cnt}) <= n_cluster({args.n_cluster})，不满足聚类条件，自动关闭聚类功能")
        else:
            logger.info("🔍开始K‑Means聚类预处理，仅使用【训练集】特征，防止数据泄露")
            stock_train_feat_list = []
            valid_stock_for_cluster = []
            for s_code in stock_list:
                try:
                    if args.data_source == "tushare":
                        df_raw = fetch_data(args.token, s_code, args.start, args.end, refresh_cache=args.refresh_cache)
                    elif args.data_source == "akshare":
                        df_raw = fetch_data_akshare(s_code, args.start, args.end, refresh_cache=args.refresh_cache)
                    df_feat = build_features(df_raw)
                    mem_df_cache[s_code] = df_feat
                    split_idx = int(len(df_feat) * TRAIN_RATIO)
                    train_df = df_feat.iloc[:split_idx].dropna(subset=FEATURE_COLS)
                    if len(train_df) >= 20:
                        feat_mean = train_df[FEATURE_COLS].mean()
                        stock_train_feat_list.append(feat_mean.values)
                        valid_stock_for_cluster.append(s_code)
                    else:
                        logger.warning(f"⚠️{s_code}训练集有效样本不足，不参与聚类")
                except Exception as e:
                    logger.warning(f"聚类预处理跳过{s_code}:{str(e)}")
            if len(valid_stock_for_cluster) >= args.n_cluster:
                X = np.array(stock_train_feat_list)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                km = KMeans(n_clusters=args.n_cluster, random_state=SEED, n_init="auto")
                labels = km.fit_predict(X_scaled)
                for s, l in zip(valid_stock_for_cluster, labels):
                    stock_cluster_map[s] = int(l)
                for s in stock_list:
                    if s not in stock_cluster_map:
                        stock_cluster_map[s] = -1
                use_cluster = True
                logger.info(f"✅聚类完成，股票‑簇映射:{stock_cluster_map}")
            else:
                logger.warning(f"⚠️参与聚类有效股票数量{len(valid_stock_for_cluster)} < {args.n_cluster}，关闭聚类")
    if not use_cluster:
        for s in stock_list:
            stock_cluster_map[s] = -1
    batch_result = []
    for s_code in stock_list:
        cid = stock_cluster_map[s_code]
        logger.info(f"\n====开始处理 {s_code} | cluster_id={cid} ====")
        cur_params = DEFAULT_PARAMS.copy()
        df_feat_arg = mem_df_cache.get(s_code, None)
        try:
            res = run_single_stock(
                token=args.token,
                stock_code=s_code,
                start=args.start,
                end=args.end,
                initial_capital=initial_capital,
                no_plot=args.no_plot,
                model_type=args.model,
                model_params=cur_params,
                prob_threshold=args.prob_threshold,
                sell_threshold=args.sell_threshold,
                cluster_id=cid,
                data_source=args.data_source,
                refresh_cache=args.refresh_cache,
                df_feat_arg=df_feat_arg,
            )
            batch_result.append(res)
        except KeyboardInterrupt:
            logger.info("👋用户Ctrl‑C终止批量任务")
            raise
        except Exception as e:
            logger.error(f"❌股票 {s_code} 回测失败 {str(e)}", exc_info=True)
            note_msg = "unknown_error"
            fail_reason = str(e)
            if "fetch" in str(e).lower():
                note_msg = "fetch_fail"
            elif "样本不足" in str(e):
                note_msg = "train_nan"
            batch_result.append({
                "stock_code": s_code,
                "model_type": args.model,
                "cluster_id": cid,
                "prob_threshold": args.prob_threshold,
                "sell_threshold": args.sell_threshold,
                "cv_mean_acc": np.nan,
                "cv_std_acc": np.nan,
                "test_acc": np.nan,
                "total_return": np.nan,
                "annual_return": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "trade_cnt": -1,
                "note": note_msg,
                "fail_reason": fail_reason
            })
    summary_df = pd.DataFrame(batch_result)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sum_csv = os.path.join(OUTPUT_DIR, f"batch_batch_summary_{ts_str}.csv")
    summary_df.to_csv(sum_csv, index=False, encoding="utf‑8‑sig")
    logger.info(f"\n✅全部批量任务结束，汇总报告输出 {sum_csv}")
    logger.info("\n" + summary_df.to_string())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 用户手动Ctrl+C终止程序，正常退出")
    except MemoryError:
        logger.error("❌内存不足，请缩小回测区间或者减少股票数量")
    except Exception as e:
        logger.error(f"顶层程序异常终止 {str(e)}", exc_info=True)
