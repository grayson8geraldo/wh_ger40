#!/usr/bin/env python3
"""
Backtest for WH GER40 Bounce Indicator.
Replicates the Pine Script logic on hourly CSV data.
"""

import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG (mirrors Pine Script defaults)
# ─────────────────────────────────────────────
RIBBON_TYPE = "EMA"
RIBBON_LEN = 200
SIG_LOOKBACK = 50
SIG_METHOD = "StdDev"  # or "ATR"
T1_MULT = 1.5
T2_MULT = 2.5
T3_MULT = 3.5
RR_RATIO = 1.0
LIQ_LOOKBACK = 5
LIQ_DEPTH = 5

# Higher TF for liquidity: simulate Daily and 4H from hourly data
LIQ_TF1_HOURS = 24  # Daily
LIQ_TF2_HOURS = 4   # 4H


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_all_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "DEU.IDX-EUR_Hour_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["UTC"], dayfirst=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("UTC", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────
def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calc_sma(series, window):
    return series.rolling(window, min_periods=window).mean()


def calc_hma(series, period):
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = series.rolling(half, min_periods=half).mean()
    wma_full = series.rolling(period, min_periods=period).mean()
    diff = 2 * wma_half - wma_full
    return diff.rolling(sqrt_p, min_periods=sqrt_p).mean()


def calc_dema(series, span):
    ema1 = series.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    return 2 * ema1 - ema2


def calc_ma(series, length, ma_type):
    if ma_type == "EMA":
        return calc_ema(series, length)
    elif ma_type == "SMA":
        return calc_sma(series, length)
    elif ma_type == "HMA":
        return calc_hma(series, length)
    elif ma_type == "DEMA":
        return calc_dema(series, length)
    else:
        return calc_ema(series, length)


def calc_deviation(df, lookback, method):
    if method == "StdDev":
        return df["Close"].rolling(lookback, min_periods=lookback).std()
    else:  # ATR
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(lookback, min_periods=lookback).mean()


def resample_ohlc(df, hours):
    """Resample hourly data to higher TF for liquidity pivot detection."""
    df_r = df.set_index("UTC").resample(f"{hours}h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    return df_r


def find_pivots(highs, lows, lb):
    """Find pivot highs and lows with lookback lb on each side."""
    pivot_highs = pd.Series(np.nan, index=highs.index)
    pivot_lows = pd.Series(np.nan, index=lows.index)
    for i in range(lb, len(highs) - lb):
        h = highs.iloc[i]
        if all(h >= highs.iloc[i - lb:i]) and all(h >= highs.iloc[i + 1:i + lb + 1]):
            pivot_highs.iloc[i] = h
        lo = lows.iloc[i]
        if all(lo <= lows.iloc[i - lb:i]) and all(lo <= lows.iloc[i + 1:i + lb + 1]):
            pivot_lows.iloc[i] = lo
    return pivot_highs, pivot_lows


def map_htf_pivots_to_hourly(df_hourly, df_htf, pivot_highs, pivot_lows):
    """Map higher TF pivots back to hourly index, forward-fill."""
    ph = pivot_highs.dropna()
    pl = pivot_lows.dropna()
    # Create series aligned to hourly
    ph_hourly = pd.Series(np.nan, index=df_hourly.index)
    pl_hourly = pd.Series(np.nan, index=df_hourly.index)
    for ts, val in ph.items():
        idx = df_hourly["UTC"].searchsorted(ts)
        if idx < len(df_hourly):
            ph_hourly.iloc[idx] = val
    for ts, val in pl.items():
        idx = df_hourly["UTC"].searchsorted(ts)
        if idx < len(df_hourly):
            pl_hourly.iloc[idx] = val
    return ph_hourly, pl_hourly


# ─────────────────────────────────────────────
# LIQUIDITY LEVEL TRACKER
# ─────────────────────────────────────────────
class LiquidityTracker:
    def __init__(self, max_depth):
        self.highs = []
        self.lows = []
        self.max_depth = max_depth

    def add_high(self, val):
        if np.isnan(val):
            return
        for v in self.highs:
            if abs(v - val) < 0.5:
                return
        self.highs.append(val)
        if len(self.highs) > self.max_depth:
            self.highs.pop(0)

    def add_low(self, val):
        if np.isnan(val):
            return
        for v in self.lows:
            if abs(v - val) < 0.5:
                return
        self.lows.append(val)
        if len(self.lows) > self.max_depth:
            self.lows.pop(0)

    def sweep(self, close):
        self.highs = [h for h in self.highs if close <= h]
        self.lows = [l for l in self.lows if close >= l]

    def nearest_above(self, close):
        above = [h for h in self.highs if h > close]
        return min(above) if above else None

    def nearest_below(self, close):
        below = [l for l in self.lows if l < close]
        return max(below) if below else None


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(df):
    n = len(df)

    # Trend Ribbon
    df["ribbon"] = calc_ma(df["Close"], RIBBON_LEN, RIBBON_TYPE)
    df["is_bullish"] = df["Close"] > df["ribbon"]
    df["is_bearish"] = df["Close"] < df["ribbon"]

    # Mean & Deviation
    df["mean"] = calc_sma(df["Close"], SIG_LOOKBACK)
    df["dev"] = calc_deviation(df, SIG_LOOKBACK, SIG_METHOD)

    # Bands
    df["upper_t1"] = df["mean"] + df["dev"] * T1_MULT
    df["lower_t1"] = df["mean"] - df["dev"] * T1_MULT
    df["upper_t2"] = df["mean"] + df["dev"] * T2_MULT
    df["lower_t2"] = df["mean"] - df["dev"] * T2_MULT
    df["upper_t3"] = df["mean"] + df["dev"] * T3_MULT
    df["lower_t3"] = df["mean"] - df["dev"] * T3_MULT

    # Crossover/Crossunder detection
    def crossunder(src, level):
        return (src.shift(1) >= level.shift(1)) & (src < level)

    def crossover(src, level):
        return (src.shift(1) <= level.shift(1)) & (src > level)

    # Signals
    df["t3_buy"] = crossunder(df["Close"], df["lower_t3"]) & df["is_bullish"]
    df["t3_sell"] = crossover(df["Close"], df["upper_t3"]) & df["is_bearish"]
    df["t2_buy"] = crossunder(df["Close"], df["lower_t2"]) & df["is_bullish"] & ~df["t3_buy"]
    df["t2_sell"] = crossover(df["Close"], df["upper_t2"]) & df["is_bearish"] & ~df["t3_sell"]
    df["t1_buy"] = crossunder(df["Close"], df["lower_t1"]) & df["is_bullish"] & ~df["t2_buy"] & ~df["t3_buy"]
    df["t1_sell"] = crossover(df["Close"], df["upper_t1"]) & df["is_bearish"] & ~df["t2_sell"] & ~df["t3_sell"]

    df["any_buy"] = df["t1_buy"] | df["t2_buy"] | df["t3_buy"]
    df["any_sell"] = df["t1_sell"] | df["t2_sell"] | df["t3_sell"]

    # Compute liquidity pivots from higher TFs
    df_daily = resample_ohlc(df, LIQ_TF1_HOURS)
    df_4h = resample_ohlc(df, LIQ_TF2_HOURS)

    ph_d, pl_d = find_pivots(df_daily["High"], df_daily["Low"], LIQ_LOOKBACK)
    ph_4h, pl_4h = find_pivots(df_4h["High"], df_4h["Low"], LIQ_LOOKBACK)

    ph_d_hourly, pl_d_hourly = map_htf_pivots_to_hourly(df, df_daily, ph_d, pl_d)
    ph_4h_hourly, pl_4h_hourly = map_htf_pivots_to_hourly(df, df_4h, ph_4h, pl_4h)

    # ── Simulate trades ──
    trades = []
    in_trade = False
    entry_price = 0
    tp_price = 0
    sl_price = 0
    trade_dir = None  # "long" or "short"
    entry_bar = 0
    entry_tier = ""

    liq_tf1 = LiquidityTracker(LIQ_DEPTH)
    liq_tf2 = LiquidityTracker(LIQ_DEPTH)

    # Skip initial warmup
    start_idx = max(RIBBON_LEN, SIG_LOOKBACK) + 10

    for i in range(start_idx, n):
        close = df["Close"].iloc[i]
        high_i = df["High"].iloc[i]
        low_i = df["Low"].iloc[i]

        # Update liquidity levels
        if not np.isnan(ph_d_hourly.iloc[i]):
            liq_tf1.add_high(ph_d_hourly.iloc[i])
        if not np.isnan(pl_d_hourly.iloc[i]):
            liq_tf1.add_low(pl_d_hourly.iloc[i])
        if not np.isnan(ph_4h_hourly.iloc[i]):
            liq_tf2.add_high(ph_4h_hourly.iloc[i])
        if not np.isnan(pl_4h_hourly.iloc[i]):
            liq_tf2.add_low(pl_4h_hourly.iloc[i])

        # Sweep levels
        liq_tf1.sweep(close)
        liq_tf2.sweep(close)

        # Check trade exit
        if in_trade:
            if trade_dir == "long":
                if high_i >= tp_price:
                    trades.append({
                        "entry_bar": entry_bar, "exit_bar": i,
                        "entry_time": df["UTC"].iloc[entry_bar],
                        "exit_time": df["UTC"].iloc[i],
                        "dir": "LONG", "tier": entry_tier,
                        "entry": entry_price, "tp": tp_price, "sl": sl_price,
                        "exit_price": tp_price, "result": "TP",
                        "pnl": tp_price - entry_price,
                        "pnl_pct": (tp_price - entry_price) / entry_price * 100
                    })
                    in_trade = False
                elif low_i <= sl_price:
                    trades.append({
                        "entry_bar": entry_bar, "exit_bar": i,
                        "entry_time": df["UTC"].iloc[entry_bar],
                        "exit_time": df["UTC"].iloc[i],
                        "dir": "LONG", "tier": entry_tier,
                        "entry": entry_price, "tp": tp_price, "sl": sl_price,
                        "exit_price": sl_price, "result": "SL",
                        "pnl": sl_price - entry_price,
                        "pnl_pct": (sl_price - entry_price) / entry_price * 100
                    })
                    in_trade = False
            elif trade_dir == "short":
                if low_i <= tp_price:
                    trades.append({
                        "entry_bar": entry_bar, "exit_bar": i,
                        "entry_time": df["UTC"].iloc[entry_bar],
                        "exit_time": df["UTC"].iloc[i],
                        "dir": "SHORT", "tier": entry_tier,
                        "entry": entry_price, "tp": tp_price, "sl": sl_price,
                        "exit_price": tp_price, "result": "TP",
                        "pnl": entry_price - tp_price,
                        "pnl_pct": (entry_price - tp_price) / entry_price * 100
                    })
                    in_trade = False
                elif high_i >= sl_price:
                    trades.append({
                        "entry_bar": entry_bar, "exit_bar": i,
                        "entry_time": df["UTC"].iloc[entry_bar],
                        "exit_time": df["UTC"].iloc[i],
                        "dir": "SHORT", "tier": entry_tier,
                        "entry": entry_price, "tp": tp_price, "sl": sl_price,
                        "exit_price": sl_price, "result": "SL",
                        "pnl": entry_price - sl_price,
                        "pnl_pct": (entry_price - sl_price) / entry_price * 100
                    })
                    in_trade = False

        # New signal (only if not in trade)
        if not in_trade:
            signal_buy = df["any_buy"].iloc[i]
            signal_sell = df["any_sell"].iloc[i]

            tier = ""
            if df["t3_buy"].iloc[i] or df["t3_sell"].iloc[i]:
                tier = "T3"
            elif df["t2_buy"].iloc[i] or df["t2_sell"].iloc[i]:
                tier = "T2"
            elif df["t1_buy"].iloc[i] or df["t1_sell"].iloc[i]:
                tier = "T1"

            if signal_buy:
                # Find nearest liquidity above
                target_tf1 = liq_tf1.nearest_above(close)
                target_tf2 = liq_tf2.nearest_above(close)
                if target_tf1 is not None and target_tf2 is not None:
                    target = min(target_tf1, target_tf2)
                elif target_tf1 is not None:
                    target = target_tf1
                elif target_tf2 is not None:
                    target = target_tf2
                else:
                    target = None

                if target is not None and target > close:
                    dist = target - close
                    entry_price = close
                    tp_price = target
                    sl_price = close - dist * RR_RATIO
                    trade_dir = "long"
                    entry_bar = i
                    entry_tier = tier
                    in_trade = True

            elif signal_sell:
                target_tf1 = liq_tf1.nearest_below(close)
                target_tf2 = liq_tf2.nearest_below(close)
                if target_tf1 is not None and target_tf2 is not None:
                    target = max(target_tf1, target_tf2)
                elif target_tf1 is not None:
                    target = target_tf1
                elif target_tf2 is not None:
                    target = target_tf2
                else:
                    target = None

                if target is not None and target < close:
                    dist = close - target
                    entry_price = close
                    tp_price = target
                    sl_price = close + dist * RR_RATIO
                    trade_dir = "short"
                    entry_bar = i
                    entry_tier = tier
                    in_trade = True

    return trades, df


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
def print_report(trades, df):
    if not trades:
        print("No trades generated.")
        return

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = tdf[tdf["result"] == "TP"]
    losses = tdf[tdf["result"] == "SL"]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100

    total_pnl = tdf["pnl"].sum()
    avg_pnl = tdf["pnl"].mean()
    avg_win = wins["pnl"].mean() if win_count > 0 else 0
    avg_loss = losses["pnl"].mean() if loss_count > 0 else 0

    # Holding time
    tdf["hold_bars"] = tdf["exit_bar"] - tdf["entry_bar"]
    avg_hold = tdf["hold_bars"].mean()

    # Max drawdown (cumulative PnL)
    tdf["cum_pnl"] = tdf["pnl"].cumsum()
    tdf["cum_max"] = tdf["cum_pnl"].cummax()
    tdf["drawdown"] = tdf["cum_pnl"] - tdf["cum_max"]
    max_dd = tdf["drawdown"].min()

    # Profit factor
    gross_profit = wins["pnl"].sum() if win_count > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if loss_count > 0 else 0.01
    profit_factor = gross_profit / gross_loss

    # By tier
    tier_stats = {}
    for tier in ["T1", "T2", "T3"]:
        t = tdf[tdf["tier"] == tier]
        if len(t) > 0:
            tw = t[t["result"] == "TP"]
            tier_stats[tier] = {
                "trades": len(t),
                "wins": len(tw),
                "wr": len(tw) / len(t) * 100,
                "pnl": t["pnl"].sum(),
                "avg_pnl": t["pnl"].mean()
            }

    # By direction
    longs = tdf[tdf["dir"] == "LONG"]
    shorts = tdf[tdf["dir"] == "SHORT"]

    # ── Print ──
    print("=" * 65)
    print("  WH GER40 BOUNCE INDICATOR — BACKTEST REPORT")
    print("=" * 65)
    print(f"  Period:        {df['UTC'].iloc[0].strftime('%Y-%m-%d')} → {df['UTC'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Timeframe:     1H (hourly)")
    print(f"  Total bars:    {len(df):,}")
    print(f"  Config:        Ribbon={RIBBON_TYPE}({RIBBON_LEN}), "
          f"Dev={SIG_METHOD}({SIG_LOOKBACK}), "
          f"Tiers={T1_MULT}/{T2_MULT}/{T3_MULT}, RR=1:{RR_RATIO}")
    print("-" * 65)

    print(f"\n  OVERALL RESULTS")
    print(f"  {'Total trades:':<25} {total}")
    print(f"  {'Wins (TP):':<25} {win_count}  ({win_rate:.1f}%)")
    print(f"  {'Losses (SL):':<25} {loss_count}  ({100 - win_rate:.1f}%)")
    print(f"  {'Total PnL (points):':<25} {total_pnl:+,.1f}")
    print(f"  {'Avg PnL per trade:':<25} {avg_pnl:+,.1f}")
    print(f"  {'Avg Win:':<25} {avg_win:+,.1f}")
    print(f"  {'Avg Loss:':<25} {avg_loss:+,.1f}")
    print(f"  {'Profit Factor:':<25} {profit_factor:.2f}")
    print(f"  {'Max Drawdown (pts):':<25} {max_dd:+,.1f}")
    print(f"  {'Avg Hold (bars/hours):':<25} {avg_hold:.1f}")

    print(f"\n  BY TIER")
    print(f"  {'Tier':<6} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>10} {'Avg PnL':>10}")
    for tier in ["T1", "T2", "T3"]:
        if tier in tier_stats:
            s = tier_stats[tier]
            print(f"  {tier:<6} {s['trades']:>7} {s['wins']:>6} {s['wr']:>6.1f}% {s['pnl']:>+10.1f} {s['avg_pnl']:>+10.1f}")
        else:
            print(f"  {tier:<6} {'—':>7}")

    print(f"\n  BY DIRECTION")
    for label, subset in [("LONG", longs), ("SHORT", shorts)]:
        if len(subset) > 0:
            sw = subset[subset["result"] == "TP"]
            print(f"  {label:<7} Trades={len(subset):>4}  "
                  f"WR={len(sw)/len(subset)*100:>5.1f}%  "
                  f"PnL={subset['pnl'].sum():>+10.1f}  "
                  f"Avg={subset['pnl'].mean():>+8.1f}")

    # Monthly breakdown
    tdf["month"] = pd.to_datetime(tdf["entry_time"]).dt.to_period("M")
    monthly = tdf.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("result", lambda x: (x == "TP").sum())
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'Month':<10} {'Trades':>7} {'WR%':>7} {'PnL':>12}")
    for idx, row in monthly.iterrows():
        print(f"  {str(idx):<10} {int(row['trades']):>7} {row['wr']:>6.1f}% {row['pnl']:>+12.1f}")

    # Streak analysis
    results = tdf["result"].tolist()
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for r in results:
        if r == "TP":
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)

    print(f"\n  STREAKS")
    print(f"  {'Max win streak:':<25} {max_win_streak}")
    print(f"  {'Max loss streak:':<25} {max_loss_streak}")

    print("\n" + "=" * 65)

    # Print last 10 trades
    print(f"\n  LAST 10 TRADES")
    print(f"  {'Date':<18} {'Dir':<6} {'Tier':<5} {'Entry':>10} {'TP':>10} {'SL':>10} {'Result':<4} {'PnL':>10}")
    for _, t in tdf.tail(10).iterrows():
        dt = pd.to_datetime(t["entry_time"]).strftime("%Y-%m-%d %H:%M")
        print(f"  {dt:<18} {t['dir']:<6} {t['tier']:<5} {t['entry']:>10.1f} {t['tp']:>10.1f} {t['sl']:>10.1f} {t['result']:<4} {t['pnl']:>+10.1f}")

    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    print("Loading GER40 hourly data...")
    df = load_all_data(data_dir)
    print(f"Loaded {len(df):,} bars from {df['UTC'].iloc[0]} to {df['UTC'].iloc[-1]}")
    print("Running backtest...\n")
    trades, df = run_backtest(df)
    print_report(trades, df)
