"""
DASHBOARD SERVER — Normandy Trading House
==========================================
Serves the live monitoring dashboard at http://localhost:5050

Reads directly from the engine output files:
  - output/execution_log.csv     → trades, positions, P&L
  - output/dry_run_balance.json  → virtual balance
  - output/quant_{asset}.csv     → trend mode per asset (if present)

Usage:
  python dashboard_server.py
  Then open: http://localhost:5050

Dependencies:
  pip install flask pandas
"""

import os
import time
import json
import math
import logging
from datetime import datetime, date
from typing import Optional

import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_from_directory
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False

# ─── config ───────────────────────────────────────────────────────────────────
PORT        = 5050

# ── Path config ───────────────────────────────────────────────────────────────
# STATIC_DIR: where index.html, crest.png, status.json live (your GitHub Pages repo)
# OUTPUT_DIR: where the engine writes execution_log.csv and dry_run_balance.json

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # this script's folder (WEBSITES)
STATIC_DIR  = BASE_DIR                                     # GitHub Pages repo root

# Engine output folder — adjust this path if needed
OUTPUT_DIR  = os.path.join(
    os.path.expanduser("~"),
    "Documents", "COMPANIES", "Normandy Trading House", "QUANT ENGINE", "output"
)

ASSET_DISPLAY = {
    "xau":     ("XAU / USD",  "Gold · Spot"),
    "sp500":   ("S&P 500",    "US Index · CFD"),
    "ftse100": ("FTSE 100",   "UK Index · CFD"),
    "gbpchf":  ("GBP / CHF",  "FX · Spot"),
    "xrp":     ("XRP / USD",  "Crypto · Spot"),
    "ukoil":   ("UK Oil",     "Brent · CFD"),
}

log = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

# ─── app ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)
if _CORS_AVAILABLE:
    CORS(app)


def _sanitise(obj):
    """Recursively convert numpy/pandas types to plain Python for JSON."""
    if isinstance(obj, dict):    return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):    return [_sanitise(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None   # NaN / Inf → null
    return obj   # flask-cors optional; won't matter if serving HTML from same origin


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/crest.png")
def crest():
    return send_from_directory(STATIC_DIR, "crest.png")


@app.route("/api/status")
def status():
    try:
        data = _sanitise(_build_status())
        _write_status_json(data)
        return jsonify(data)
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        log.error(f"Status build error: {e}")
        log.error(msg)
        return jsonify({"error": str(e), "traceback": msg,
                        "timestamp": datetime.now().isoformat()}), 500


# ─── data builders ────────────────────────────────────────────────────────────

def _write_status_json(data: dict):
    """Write status.json into the static dir (repo root) for GitHub Pages."""
    try:
        path = os.path.join(STATIC_DIR, "status.json")
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except Exception as e:
        log.debug(f"status.json write failed: {e}")


def _build_status() -> dict:
    balance   = _read_balance()
    exec_df   = _read_execution_log()
    modes     = _read_asset_modes()

    analytics   = _compute_analytics(exec_df)
    per_asset   = _compute_per_asset(exec_df, modes)
    recent      = _compute_recent_trades(exec_df)
    open_pos    = _compute_open_positions(exec_df)
    health      = _compute_health(exec_df, analytics)

    return {
        "timestamp":      datetime.now().isoformat(),
        "balance":        balance,
        "analytics":      analytics,
        "per_asset":      per_asset,
        "recent_trades":  recent,
        "open_positions": open_pos,
        "health":         health,
    }


def _read_balance() -> dict:
    path = os.path.join(OUTPUT_DIR, "dry_run_balance.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            bal   = data.get("balance", 10000.0)
            start = data.get("daily_start", bal)
            daily_pnl     = bal - start
            daily_pnl_pct = (daily_pnl / start * 100) if start else 0
            return {
                "balance":        round(bal, 2),
                "daily_start":    round(start, 2),
                "daily_pnl":      round(daily_pnl, 2),
                "daily_pnl_pct":  round(daily_pnl_pct, 3),
                "mode":           "dry_run",
            }
        except Exception:
            pass
    return {"balance": 10000.0, "daily_start": 10000.0, "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0, "mode": "unknown"}


def _read_execution_log() -> Optional[pd.DataFrame]:
    path = os.path.join(OUTPUT_DIR, "execution_log.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["pnl"]       = pd.to_numeric(df["pnl"],   errors="coerce").fillna(0)
        df["price"]     = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["lots"]      = pd.to_numeric(df["lots"],  errors="coerce").fillna(0)
        return df
    except Exception as e:
        log.warning(f"Could not read execution log: {e}")
        return None


def _read_asset_modes() -> dict:
    """Read the last trend_mode from each quant_{asset}.csv."""
    modes = {}
    for key in ASSET_DISPLAY:
        path = os.path.join(OUTPUT_DIR, f"quant_{key}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=["trend_mode"])
                if not df.empty:
                    modes[key] = df["trend_mode"].iloc[-1]
            except Exception:
                pass
    return modes


def _compute_analytics(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "max_dd_pct": 0, "avg_pnl_pts": 0}

    closed = df[df["action"].str.startswith("CLOSE", na=False) & (df["pnl"] != 0)]
    if closed.empty:
        return {"total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "total_pnl": 0, "max_dd_pct": 0, "avg_pnl_pts": 0}

    wins   = closed[closed["pnl"] > 0]["pnl"]
    losses = closed[closed["pnl"] < 0]["pnl"]
    total  = len(closed)
    win_rate = len(wins) / total * 100 if total else 0
    gross_win  = wins.sum()  if len(wins)   else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0
    pf         = gross_win / gross_loss if gross_loss else (gross_win if gross_win else 0)

    # Running balance for drawdown
    balance = _read_balance()
    start   = balance.get("daily_start", 10000.0)
    cumulative = closed["pnl"].cumsum()
    running_bal = start + cumulative
    peak        = running_bal.cummax()
    drawdown    = ((running_bal - peak) / peak * 100).min()

    total_pnl   = closed["pnl"].sum()
    total_pnl_pct = total_pnl / start * 100 if start else 0

    return {
        "total_trades":   int(total),
        "win_rate":       round(win_rate, 2),
        "profit_factor":  round(pf, 2),
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl_pct, 3),
        "max_dd_pct":     round(drawdown, 3),
        "avg_pnl_pts":    round(closed["pnl"].mean(), 4),
        "long_trades":    int(len(closed[closed["direction"] == "LONG"])),
        "short_trades":   int(len(closed[closed["direction"] == "SHORT"])),
        "long_wins":      int(len(closed[(closed["direction"]=="LONG")  & (closed["pnl"]>0)])),
        "short_wins":     int(len(closed[(closed["direction"]=="SHORT") & (closed["pnl"]>0)])),
    }


def _compute_per_asset(df: Optional[pd.DataFrame], modes: dict) -> list:
    assets = []
    for key, (display_name, market) in ASSET_DISPLAY.items():
        if df is not None and not df.empty:
            adf     = df[df["asset"] == key]
            closed  = adf[adf["action"].str.startswith("CLOSE", na=False) & (adf["pnl"] != 0)]
            total   = len(closed)
            wins    = closed[closed["pnl"] > 0]
            losses  = closed[closed["pnl"] < 0]
            win_pct = round(len(wins) / total * 100, 1) if total else 0
            gw      = wins["pnl"].sum()   if len(wins)   else 0
            gl      = abs(losses["pnl"].sum()) if len(losses) else 0
            pf      = round(gw / gl, 2) if gl else (round(gw, 2) if gw else 0)
            total_pnl = closed["pnl"].sum()
            balance_start = _read_balance().get("daily_start", 10000.0)
            perf_pct = round(total_pnl / balance_start * 100, 3) if balance_start else 0
        else:
            total = 0; win_pct = 0; pf = 0; perf_pct = 0

        # Open position for this asset
        pos_state = "FLAT"
        if df is not None and not df.empty:
            adf_open   = df[df["asset"] == key]
            opens      = adf_open[adf_open["action"].str.match(r"OPEN_", na=False)]
            closes     = adf_open[adf_open["action"].str.match(r"CLOSE_", na=False)]
            if len(opens) > len(closes):
                last_open = opens.iloc[-1]
                pos_state = last_open["direction"]   # "LONG" or "SHORT"

        mode = modes.get(key, "")

        assets.append({
            "key":         key,
            "name":        display_name,
            "market":      market,
            "pos":         pos_state,
            "perf_pct":    perf_pct,
            "trades":      total,
            "win_pct":     win_pct,
            "pf":          pf,
            "mode":        mode,
        })
    return assets


def _compute_open_positions(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {}
    result = {}
    for key in ASSET_DISPLAY:
        adf    = df[df["asset"] == key]
        opens  = adf[adf["action"].str.match(r"OPEN_", na=False)]
        closes = adf[adf["action"].str.match(r"CLOSE_", na=False)]
        if len(opens) > len(closes):
            last = opens.iloc[-1]
            result[key] = {
                "direction":   last["direction"],
                "entry_price": last["price"],
                "lots":        last["lots"],
                "opened_at":   str(last["timestamp"])[:19],
            }
    return result


def _compute_recent_trades(df: Optional[pd.DataFrame], n: int = 10) -> list:
    if df is None or df.empty:
        return []
    closed = df[df["action"].str.startswith("CLOSE", na=False) & (df["pnl"] != 0)].tail(n)
    trades = []
    cumulative = 0.0
    balance_start = _read_balance().get("daily_start", 10000.0)

    for i, (_, row) in enumerate(closed.iterrows(), 1):
        pnl = row["pnl"]
        cumulative += pnl
        cum_pct = cumulative / balance_start * 100 if balance_start else 0
        pnl_pct = pnl / balance_start * 100 if balance_start else 0
        trades.append({
            "n":          i,
            "symbol":     row["asset"].upper() + "/USD" if row["asset"] in ("xau","xrp","ukoil") else row["asset"].upper(),
            "side":       row["direction"],
            "exit_time":  str(row["timestamp"])[:16],
            "exit_price": round(float(row["price"]), 5),
            "pnl":        round(pnl, 2),
            "pnl_pct":    round(pnl_pct, 4),
            "cum_pct":    round(cum_pct, 4),
            "reason":     row.get("reason", ""),
        })
    return trades


def _compute_health(df: Optional[pd.DataFrame], analytics: dict) -> dict:
    if df is None or df.empty or not analytics.get("total_trades"):
        return {}

    closed = df[df["action"].str.startswith("CLOSE", na=False) & (df["pnl"] != 0)]
    if closed.empty:
        return {}

    # Rolling 20-trade expectancy
    tail20    = closed.tail(20)["pnl"]
    exp_20    = round(tail20.mean(), 4) if len(tail20) else 0
    exp_20_pct = round(exp_20 / _read_balance().get("daily_start", 10000.0) * 100, 4)

    # Duration from timestamps (if open/close rows can be paired)
    longs   = analytics.get("long_trades", 0)
    shorts  = analytics.get("short_trades", 0)
    lw      = analytics.get("long_wins", 0)
    sw      = analytics.get("short_wins", 0)
    long_wr  = round(lw / longs  * 100, 1) if longs  else 0
    short_wr = round(sw / shorts * 100, 1) if shorts else 0

    pnl_series = closed["pnl"]
    wins        = pnl_series[pnl_series > 0]
    losses      = pnl_series[pnl_series < 0]
    pf          = analytics.get("profit_factor", 0)
    max_dd      = analytics.get("max_dd_pct", 0)
    total_return = analytics.get("total_pnl_pct", 0)
    recovery    = round(abs(total_return / max_dd), 2) if max_dd and total_return else 0
    volatility  = round(pnl_series.std() / _read_balance().get("daily_start", 10000.0) * 100, 4) if len(pnl_series) > 1 else 0

    return {
        "rolling_expectancy_20":  exp_20_pct,
        "volatility_pct":         volatility,
        "recovery_factor":        recovery,
        "long_wr":                long_wr,
        "short_wr":               short_wr,
        "total_long":             longs,
        "total_short":            shorts,
        "total_trades":           analytics.get("total_trades", 0),
    }


# ─── run ──────────────────────────────────────────────────────────────────────
def _git_push_status():
    """
    Runs in background. Every 30 seconds:
      - writes status.json
      - git add / commit / push  (if repo is detected)
    Requires git to be installed and the repo to be authenticated
    (e.g. GitHub Desktop handles this automatically).
    """
    import threading, subprocess

    def _push():
        try:
            repo = STATIC_DIR
            # Only push if this folder is actually a git repo
            check = subprocess.run(
                ["git", "-C", repo, "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True
            )
            if check.returncode != 0:
                log.warning("[GIT] STATIC_DIR is not a git repo — GitHub Pages push disabled")
                return False
            return True
        except FileNotFoundError:
            log.warning("[GIT] git not found — GitHub Pages push disabled")
            return False

    def _loop():
        is_repo = _push()
        push_cycle = 0
        while True:
            try:
                data = _sanitise(_build_status())
                _write_status_json(data)

                # Push every 6 cycles = every 30 seconds
                if is_repo and push_cycle % 6 == 0:
                    repo = STATIC_DIR
                    subprocess.run(["git", "-C", repo, "add", "status.json"],
                                   capture_output=True)
                    result = subprocess.run(
                        ["git", "-C", repo, "commit", "-m",
                         f"chore: status update {datetime.now().strftime('%H:%M:%S')} [skip ci]"],
                        capture_output=True, text=True
                    )
                    if "nothing to commit" not in result.stdout:
                        push = subprocess.run(
                            ["git", "-C", repo, "push"],
                            capture_output=True, text=True, timeout=15
                        )
                        if push.returncode == 0:
                            log.info("[GIT] status.json pushed to GitHub Pages")
                        else:
                            log.warning(f"[GIT] Push failed: {push.stderr.strip()}")
            except Exception as e:
                log.debug(f"[GIT] background error: {e}")
            push_cycle += 1
            time.sleep(5)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


def _background_status_writer():
    """Alias — calls _git_push_status which handles both writing and pushing."""
    _git_push_status()


if __name__ == "__main__":
    if _CORS_AVAILABLE:
        CORS(app)

    _background_status_writer()
    log.info(f"Dashboard server starting on http://localhost:{PORT}")
    log.info(f"Static dir (GitHub Pages repo): {os.path.abspath(STATIC_DIR)}")
    log.info(f"Engine output dir:              {os.path.abspath(OUTPUT_DIR)}")
    exists = os.path.exists(OUTPUT_DIR)
    log.info(f"Engine output dir exists:       {exists}")
    if not exists:
        log.warning(f"  ⚠  Output dir not found — check OUTPUT_DIR path in config")
    log.info("Press Ctrl+C to stop")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
