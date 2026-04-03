#!/usr/bin/env python3
"""Compare tier configurations for WH GER40 Bounce Indicator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import load_all_data, run_backtest, calc_ma, calc_sma, calc_deviation
import backtest as bt
import pandas as pd
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))
df_base = load_all_data(data_dir)

configs = [
    {"name": "T1 only",              "t1": 1.5, "t2": None, "t3": None},
    {"name": "T1+T2 (default)",      "t1": 1.5, "t2": 2.5,  "t3": 3.5},
    {"name": "T1+T2 (T2=2.0)",       "t1": 1.5, "t2": 2.0,  "t3": None},
    {"name": "T1+T2 (T2=3.0)",       "t1": 1.5, "t2": 3.0,  "t3": None},
    {"name": "T1 (mult=1.2)",        "t1": 1.2, "t2": None, "t3": None},
    {"name": "T1 (mult=1.8)",        "t1": 1.8, "t2": None, "t3": None},
    {"name": "T1+T2+T3 tight 1.0/1.8/2.5", "t1": 1.0, "t2": 1.8, "t3": 2.5},
]

print(f"{'Config':<35} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'AvgPnL':>9} {'PF':>6} {'MaxDD':>10} {'AvgHold':>8}")
print("-" * 100)

for cfg in configs:
    df = df_base.copy()

    bt.T1_MULT = cfg["t1"]
    # Disable T2/T3 by setting very high multiplier if None
    bt.T2_MULT = cfg["t2"] if cfg["t2"] is not None else 999.0
    bt.T3_MULT = cfg["t3"] if cfg["t3"] is not None else 999.0

    trades, _ = run_backtest(df)

    if not trades:
        print(f"  {cfg['name']:<33} {'No trades':>7}")
        continue

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = len(tdf[tdf["result"] == "TP"])
    wr = wins / total * 100
    pnl = tdf["pnl"].sum()
    avg_pnl = tdf["pnl"].mean()
    avg_hold = (tdf["exit_bar"] - tdf["entry_bar"]).mean()

    gross_profit = tdf[tdf["result"] == "TP"]["pnl"].sum()
    gross_loss = abs(tdf[tdf["result"] == "SL"]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    cum = tdf["pnl"].cumsum()
    dd = (cum - cum.cummax()).min()

    print(f"  {cfg['name']:<33} {total:>7} {wr:>6.1f}% {pnl:>+10.1f} {avg_pnl:>+9.1f} {pf:>6.2f} {dd:>+10.1f} {avg_hold:>7.1f}h")

# Reset defaults
bt.T1_MULT = 1.5
bt.T2_MULT = 2.5
bt.T3_MULT = 3.5
