# A股量化回测项目（均线策略+机器学习拓展）
> ⚠️免责声明：本项目仅用于编程与量化学习演示，**不构成任何投资建议**。历史回测结果不能代表未来实际收益。

本项目实现一套完整A股机器学习量化回测框架，包含两套回测实现：
1. **事件驱动回测**：严格模拟实盘订单生命周期，支持手续费、印花税、滑点、A股100股最小交易单位，用于严谨策略评估；
2. **向量化回测**：运算速度快，适合快速调参粗筛，不模拟真实撮合细节。

模型支持 LightGBM / XGBoost / 随机森林，基于量价技术指标做涨跌分类预测；使用时序交叉验证避免时间泄露。

## 项目文件结构
```
├── event_backtest.py           # 事件驱动底层回测内核，双均线策略示例
├── batch_gbdt_event_strategy.py# 批量多股票 GBDT (LGB/XGB) + 事件驱动回测
├── ml_strategy.py              # 随机森林向量化快速回测（粗筛工具）
├── db_utils.py                 # MySQL 数据库读写工具类
├── load_stock_basic.py         # 股票基础信息初始化脚本
├── init_db.sql                 # MySQL 建表 SQL 脚本
├── requirements.txt            # Python 依赖清单
├── .gitignore                  # Git 忽略配置
├── .env.example                # 环境配置模板（提交至 Git）
└── .env                        # 本地私有配置，复制.env.example 生成，不上传 Git
```

### 文件说明
- `event_backtest.py`：事件驱动回测内核，模拟真实下单、成交、持仓逻辑；支持单只标的回测。
- `batch_gbdt_event_strategy.py`：批量回测脚本，调用事件驱动内核，支持LGB、XGBoost模型；支持K‑Means股票聚类实验。
- `ml_strategy.py`：向量化快速回测，**不模拟A股100手最小交易单位**，仅用于快速实验；严谨评估请使用事件驱动版本。
- `db_utils.py`：MySQL数据库工具类，封装所有表的插入、查询操作；解决`np.nan`入库报错，处理事务提交、回滚、连接安全关闭，规避MySQL空闲断开问题。
- `load_stock_basic.py`：手动写入股票基础信息，绕过Tushare stock_basic接口限流。
- `init_db.sql`：MySQL数据库建表语句，包含行情、因子、回测结果全套表结构。

## 环境准备
### 1. Python版本
推荐 Python `3.9 ~ 3.11`

### 2. 安装依赖
```bash
pip install -r requirements.txt

