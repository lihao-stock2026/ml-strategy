-- 创建数据库（不存在才创建）
CREATE DATABASE IF NOT EXISTS stock_quant DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE stock_quant;

-- 1.股票基础信息表 stock_basic
DROP TABLE IF EXISTS stock_basic;
CREATE TABLE stock_basic (
    ts_code VARCHAR(20) NOT NULL COMMENT '股票ts代码 600519.SH',
    symbol VARCHAR(10) NOT NULL COMMENT '股票代码',
    name VARCHAR(50) NOT NULL COMMENT '股票名称',
    industry VARCHAR(50) COMMENT '所属行业',
    list_date VARCHAR(20) NOT NULL COMMENT '上市日期 YYYY‑MM‑DD',
    PRIMARY KEY (ts_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='股票基础信息';

-- 2.日线行情表 stock_daily 【移除amount，适配tushare pro_bar复权接口】
DROP TABLE IF EXISTS stock_daily;
CREATE TABLE stock_daily (
    ts_code VARCHAR(20) NOT NULL COMMENT '股票ts代码',
    trade_date VARCHAR(20) NOT NULL COMMENT '交易日 YYYY‑MM‑DD',
    open DECIMAL(12,2) NOT NULL COMMENT '开盘价',
    high DECIMAL(12,2) NOT NULL COMMENT '最高价',
    low DECIMAL(12,2) NOT NULL COMMENT '最低价',
    close DECIMAL(12,2) NOT NULL COMMENT '收盘价',
    vol DECIMAL(18,2) NOT NULL COMMENT '成交量 手',
    PRIMARY KEY (ts_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='股票日线行情';

-- 3.因子数据表 stock_factor【对齐build_features全部因子字段】
DROP TABLE IF EXISTS stock_factor;
CREATE TABLE stock_factor (
    ts_code VARCHAR(20) NOT NULL COMMENT '股票ts代码',
    trade_date VARCHAR(20) NOT NULL COMMENT '交易日 YYYY‑MM‑DD',
    ma5 DECIMAL(12,4) NULL,
    ma20 DECIMAL(12,4) NULL,
    close_ma5_ratio DECIMAL(10,6) NULL,
    close_ma20_ratio DECIMAL(10,6) NULL,
    rsi14 DECIMAL(10,4) NULL,
    macd_dif DECIMAL(12,4) NULL,
    macd_dea DECIMAL(12,4) NULL,
    macd_hist DECIMAL(12,4) NULL,
    vol_change DECIMAL(12,6) NULL,
    vol_ratio DECIMAL(12,4) NULL,
    ret_1d DECIMAL(12,6) NULL,
    ret_5d DECIMAL(12,6) NULL,
    ret_10d DECIMAL(12,6) NULL,
    volatility DECIMAL(12,6) NULL,
    label TINYINT NULL COMMENT '涨跌标签0/1',
    PRIMARY KEY (ts_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='技术因子数据表';

-- 4.回测每笔交易记录表 backtest_trade
DROP TABLE IF EXISTS backtest_trade;
CREATE TABLE backtest_trade (
    id BIGINT AUTO_INCREMENT COMMENT '自增主键',
    strategy_name VARCHAR(100) NOT NULL COMMENT '策略名称，例如gbdt_lgb',
    ts_code VARCHAR(20) NOT NULL COMMENT '股票ts代码',
    trade_date VARCHAR(20) NOT NULL COMMENT '成交日期 YYYY‑MM‑DD',
    direction VARCHAR(10) NOT NULL COMMENT 'BUY / SELL',
    price DECIMAL(12,2) NOT NULL COMMENT '成交价格',
    volume BIGINT NOT NULL COMMENT '成交股数',
    commission DECIMAL(14,2) NOT NULL COMMENT '手续费',
    tax DECIMAL(14,2) NOT NULL COMMENT '印花税',
    PRIMARY KEY(id),
    UNIQUE KEY uk_strategy_stock_date (strategy_name, ts_code, trade_date, direction)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='回测每笔成交记录';

-- 5.回测汇总指标表 backtest_summary
DROP TABLE IF EXISTS backtest_summary;
CREATE TABLE backtest_summary (
    id BIGINT AUTO_INCREMENT COMMENT '自增主键',
    strategy_name VARCHAR(100) NOT NULL COMMENT '策略名称',
    start_date VARCHAR(20) NOT NULL COMMENT '回测开始日期',
    end_date VARCHAR(20) NOT NULL COMMENT '回测结束日期',
    initial_capital DECIMAL(18,2) NOT NULL COMMENT '初始资金',
    final_capital DECIMAL(18,2) NOT NULL COMMENT '期末资金',
    total_return DECIMAL(10,4) NOT NULL COMMENT '总收益率，小数形式 0.2代表20%',
    sharpe_ratio DECIMAL(10,4) COMMENT '夏普比率',
    max_drawdown DECIMAL(10,4) NOT NULL COMMENT '最大回撤，负数',
    win_rate DECIMAL(6,4) COMMENT '胜率',
    PRIMARY KEY(id),
    UNIQUE KEY uk_strategy_timerange (strategy_name, start_date, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='回测汇总指标';
