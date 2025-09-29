# -*- coding: utf-8 -*-
"""
Bybit V5 — 4H Swing + 15m Bollinger Hybrid (Patched)
- 메인/서브 모두 DRY_RUN & VIRTUAL_PAPER 일관 처리
- 주문 응답 검증 / 실패시 안전 중단
- TP1→BE 전환: 포지션 수량 감소(실제 체결) 감지 기반
- 동일봉 쿨다운: 심볼 단일키 (롱/숏 교차 진입 차단)
- TP/SL 틱 라운딩 일관화
- 30m 레인지 SL 완화(0.15%)
"""

import os, csv, math, time, zipfile, uuid, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ===========================
# ========= CONFIG ==========
# ===========================
API_KEY    = "6D7H5kEMR5uGOlzlzA"    # <-- 본인 키
API_SECRET = "lT6XEk2Mn1xj2vcZIMS5LGxYlviQgbiuXY7p"    # <-- 본인 시크릿
TESTNET    = True
BASE_URL   = "https://api.bybit.com"

LEADER   = "BTCUSDT"
SYMBOLS  = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]

MAX_MAIN_POS = 1
MAX_SUB_POS  = 1

VIRTUAL_PAPER = False
DRY_RUN       = False
POLL_SEC      = 5
HEARTBEAT_EVERY = 6

STARTUP_RECONCILE = False
RUNTIME_REVERSE   = True

# (옵션) 명목 예산 캡
USE_NOTIONAL_BUDGET = True
PER_SYMBOL_BUDGET_USDT = {"ETHUSDT": 20000.0, "SOLUSDT": 10000.0, "BNBUSDT": 20000.0, "DEFAULT": 15000.0}

# 수량 프리셋
MAIN_FIXED_QTY = {"ETHUSDT": 0.7, "SOLUSDT": 2.0, "BNBUSDT": 0.8, "DEFAULT": 0.5}
SUB_FIXED_QTY  = {"ETHUSDT": 0.4, "SOLUSDT": 1.2, "BNBUSDT": 0.6, "DEFAULT": 0.4}

# 컨플루언스 가중
CONF_WEIGHT_STEP_MAIN = 0.5
CONF_MAX_WEIGHT_MAIN  = 2.0
CONF_WEIGHT_STEP_SUB  = 0.5
CONF_MAX_WEIGHT_SUB   = 2.0

# ===== 메인(4H) =====
MAIN_EMA_LEN         = 200
MAIN_RR_GATE         = 1.8
MAIN_ATR_LEN_4H      = 14
MAIN_SL_ATR_K        = 0.5
MAIN_TP1_RATIO       = 0.4

# EMA200 가드
EMA_NEUTRAL_PCT        = 0.002
EMA_NEUTRAL_ATR_K      = 0.3
SWITCH_UP_PCT          = 0.0015
SWITCH_DOWN_PCT        = 0.0025
EMA_SLOPE_MIN          = 0.0

# 4H Early-Expansion
MAIN_EE_ENABLE          = True
MAIN_EE_BBW_LOOKBACK    = 40
MAIN_EE_SQUEEZE_K       = 0.7
MAIN_EE_EXPAND_K        = 0.9
MAIN_EE_MIN_RR          = 2.0
MAIN_EE_ATR_LEN         = 14
MAIN_EE_SL_ATR_K        = 0.6
MAIN_EE_USE_NEAR_SL     = True
MAIN_EE_ALIGN_1H_GUARD  = True
MAIN_EE_ALLOW_LOW_CONF  = True
MAIN_EE_SIZE_FACTOR     = 0.6

# ===== 서브(15m 볼밴) =====
BB_PERIOD        = 20
BB_STD           = 2.0
ATR_LEN_1H       = 14
ATR_LEN_15M      = 14
RR_GATE_SUB      = 1.6

# 30m 확장-횡보 Range
RANGE_EXPANDED_BBW_MIN   = 0.020
SIDEWAYS_SLOPE_ABS_MAX   = 0.0004
SIDEWAYS_LOOKBACK_EMA    = 6
SIDEWAYS_PIVOT_LOOK      = 24
RANGE_TOUCH_EPS          = 0.0005
RANGE_SL_PAD_PCT         = 0.0015   # ★완화(0.15%)
MIN_RR_RANGE30           = 1.20

VOL_MULT_SUB_MIN = 1.05
VOL_MULT_SUB_MAX = 1.20
SL_MIN_PCT       = 0.0025
TP1_RATIO_SUB    = 0.40

SQUEEZE_LOOKBACK = 40
SQUEEZE_K        = 0.7
EXPAND_MULT      = 1.25
LTF_BAND_VETO_ATR_K = 0.20

SPLIT_ENTRY_ENABLE    = True
SPLIT_ENTRY_FIRST_PCT = 0.5
RETEST_ENABLE         = True
RETEST_TIMEOUT_BARS   = 8
RETEST_MAX_DRIFT_PCT  = 0.004
RETEST_TOUCH_MODE     = "band_or_mid"
RETEST_TOUCH_EPS      = 0.0020

SUB_STOP_MODE     = "BAND_RECOVERY"   # "ATR"|"MIDLINE"|"BAND_RECOVERY"
STOP_ATR_K        = 0.5
MID_BUF_PCT       = 0.0015
BAND_RECOVERY_EPS_PCT = 0.0008
SUB_MIN_STOP_DIST_PCT = 0.0012
SUB_SL_CHOICE     = "NEAR"

RANGE_BBWIDTH_THR = 0.007
RANGE_TGT_RATIO   = 2.0/3.0
FORBID_MID_RANGE  = True

ALLOW_LONG_AT_UPPER_BAND = True
LBU_REQUIRE_TREND_30M_UP = True
LBU_VOL_MULT              = 1.25
LBU_MAX_PULLIN_PCT        = 0.0020

USE_15M_EE_TRIGGER        = True
EE15M_BBW_EXPAND_K        = 0.9
EE15M_MIN_RR              = 2.0

ALLOW_SHORT_AT_LOWER_BAND   = True
SLB_REQUIRE_TREND_30M_DOWN  = True
SLB_VOL_MULT                = 1.25
SLB_MAX_BOUNCE_PCT          = 0.0015

ATR_15M_MIN  = 0.0
BBW_15M_MIN  = 0.0

SUB_ENTRY_CONFIRM_MODE   = "FIRST_OR_2"   # "FIRST_ONLY"|"TWO_ONLY"|"FIRST_OR_2"
STRONG_BAR_BODY_PCT      = 0.60
STRONG_BAR_VOL_MULT      = 1.50

SUB_TP2_TRAIL_MODE       = "ATR"          # "ATR"|"SWING"
SUB_TP2_TRAIL_ATR_K      = 0.8
SUB_TP2_TRAIL_LOOKBACK   = 5

TILT_MAX_LOSSES_DIR = 2
TILT_DEAD_BARS      = 6

# BTC 15m 동조 필수
BTC_SYNC_REQUIRED  = True
BTC_SYNC_EPS_PCT   = 0.0008

# ===== 로깅 =====
LOG_TO_CSV       = True
LOG_DIR          = "logs"
LOG_FILE         = f"{LOG_DIR}/trade_log.csv"
LOG_SKIP_REASONS = True
SKIP_LOG_FILE    = f"{LOG_DIR}/skip_log.csv"
LOG_RETENTION_DAYS = 7
LOG_MAX_BYTES      = 5_000_000

LOOPS_PER_15M = max(1, int((15*60)//POLL_SEC))

# 동일봉 쿨다운(엔진×심볼) — ★방향 제거
LAST_EXEC_BAR: Dict[Tuple[str,str], int] = {}
LAST_LOSS_DIR: Dict[Tuple[str,str], Dict[str,int]] = {}  # (engine,symbol)->{"dir": "long"/"short","losses":n,"dead":bars}

# ===========================
# ===== 유틸/로깅 ===========
# ===========================
def ensure_log_dir():
    Path(LOG_DIR).mkdir(exist_ok=True)

def _csv_write(file_path: str, header: list, row: dict):
    try:
        p = Path(file_path); exists = p.exists()
        with p.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists: w.writeheader()
            for k in header: row.setdefault(k, "")
            w.writerow(row)
    except PermissionError:
        print(f"[WARN] CSV busy: {file_path} (skipped one row)")

def log_trade_event(event: str, **kw):
    if not LOG_TO_CSV: return
    ensure_log_dir()
    header = ["ts_utc","event","symbol","engine","side","qty","entry","sl","tp1","tp2","rr",
              "mode","extra","exit_price","bbw","came_from_squeeze","range_mode",
              "rebreak_ok","ee_flag"]
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event, "symbol": kw.get("symbol"), "engine": kw.get("engine"),
        "side": kw.get("side"), "qty": kw.get("qty"),
        "entry": kw.get("entry"), "sl": kw.get("sl"), "tp1": kw.get("tp1"), "tp2": kw.get("tp2"),
        "rr": kw.get("rr"), "mode": kw.get("mode"), "extra": kw.get("extra"),
        "exit_price": kw.get("exit_price"), "bbw": kw.get("bbw"),
        "came_from_squeeze": kw.get("came_from_squeeze"),
        "range_mode": kw.get("range_mode"), "rebreak_ok": kw.get("rebreak_ok"),
        "ee_flag": kw.get("ee_flag"),
    }
    _csv_write(LOG_FILE, header, row)

def log_skip_reason(symbol: str, reason: str, extra: str=""):
    if not LOG_SKIP_REASONS: return
    ensure_log_dir()
    header = ["ts_utc","symbol","reason","extra"]
    row = {"ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "symbol": symbol, "reason": reason, "extra": extra}
    _csv_write(SKIP_LOG_FILE, header, row)

def _zip_file(src_path: Path, zip_name: str):
    if not src_path.exists(): return
    with zipfile.ZipFile(zip_name, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src_path, arcname=src_path.name)
    try: src_path.unlink()
    except: pass

def rotate_logs():
    ensure_log_dir()
    now = datetime.now(timezone.utc)
    for fp in [Path(LOG_FILE), Path(SKIP_LOG_FILE)]:
        if not fp.exists(): continue
        try:
            if fp.stat().st_size >= LOG_MAX_BYTES:
                zip_name = Path(f"{fp.with_suffix('')}_{now.strftime('%Y%m%d_%H%M%S')}.zip")
                _zip_file(fp, str(zip_name)); continue
        except: pass
        try:
            mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
            if (now - mtime) > timedelta(days=LOG_RETENTION_DAYS):
                zip_name = Path(f"{fp.with_suffix('')}_{mtime.strftime('%Y%m%d')}.zip")
                _zip_file(fp, str(zip_name))
        except: pass

# ===========================
# ===== Bybit Wrapper =======
# ===========================
def _retry(func, max_try=3, pause=0.3):
    for i in range(max_try):
        try: return func()
        except Exception as e:
            print("[WARN retry]", i+1, "err=", e)
            time.sleep(pause)
    return {"status":"error","error":"max_retry"}

class Bybit:
    def __init__(self):
        self.sess = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=TESTNET)

    @staticmethod
    def _tf_map(tf: str) -> str:
        return {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","4h":"240","1d":"D"}[tf]

    @staticmethod
    def ok(resp: dict) -> bool:
        if not isinstance(resp, dict): return False
        if resp.get("status") == "dryrun": return True
        return resp.get("retCode", 1) == 0

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int=600, retries: int=3, pause: float=0.6) -> pd.DataFrame:
        for attempt in range(1, retries+1):
            try:
                res = self.sess.get_kline(category="linear", symbol=symbol, interval=self._tf_map(timeframe), limit=limit)
                rows = list(reversed(res["result"]["list"]))
                df = pd.DataFrame(rows)
                if df.shape[1] < 6: raise ValueError("kline columns <6")
                std_cols = ["start","open","high","low","close","volume","turnover"]
                use_n = min(df.shape[1], len(std_cols))
                df = df.iloc[:, :use_n].copy(); df.columns = std_cols[:use_n]
                for c in ["open","high","low","close","volume"]:
                    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
                start = pd.to_numeric(df["start"], errors="coerce")
                unit = "ms" if start.iloc[-1] > 1e12 else "s"
                df["timestamp"] = pd.to_datetime(start, unit=unit, utc=True)
                if "high" not in df.columns: df["high"] = df["close"]
                if "low"  not in df.columns: df["low"]  = df["close"]
                use_cols = [c for c in ["timestamp","open","high","low","close","volume"] if c in df.columns]
                return df[use_cols]
            except Exception as e:
                print(f"[WARN fetch_ohlcv] {symbol} {timeframe} attempt={attempt}/{retries} err={e}")
                if attempt == retries: return pd.DataFrame()
                time.sleep(pause)

    def get_filters(self, symbol: str) -> Dict:
        try:
            info = self.sess.get_instruments_info(category="linear", symbol=symbol)
            d = info["result"]["list"][0]
            lot = d["lotSizeFilter"]; prc = d["priceFilter"]
            return {"qtyStep": float(lot["qtyStep"]), "minQty": float(lot["minOrderQty"]), "tickSize": float(prc["tickSize"])}
        except Exception as e:
            print("[WARN get_filters]", symbol, e)
            return {"qtyStep": 0.001, "minQty": 0.001, "tickSize": 0.01}

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            res = self.sess.get_positions(category="linear", symbol=symbol)
            lst = res["result"]["list"]
            if not lst: return None
            best = None
            for p in lst:
                if float(p.get("size", 0) or 0) != 0:
                    if best is None or abs(float(p["size"])) > abs(float(best["size"])): best = p
            return best
        except Exception as e:
            print("[ERROR get_position]", symbol, e); return None

    def close_position(self, symbol: str):
        pos = self.get_position(symbol)
        if not pos: return {"status":"no_pos"}
        side = "Sell" if float(pos["size"]) > 0 else "Buy"
        qty  = abs(float(pos["size"]))
        return self.place_market(symbol, side, qty, reduce_only=True)

    def place_market(self, symbol: str, side: str, qty: float, reduce_only: bool=False):
        if DRY_RUN:
            print(f"[DRYRUN] MARKET {symbol} {side} qty={qty} RO={reduce_only}")
            return {"status":"dryrun"}
        oid = str(uuid.uuid4())[:20]
        return _retry(lambda: self.sess.place_order(
            category="linear", symbol=symbol, side=side, orderType="Market",
            qty=str(qty), reduceOnly=reduce_only, orderLinkId=oid
        ))

    def place_limit(self, symbol: str, side: str, qty: float, price: float, reduce_only: bool=False):
        if DRY_RUN:
            print(f"[DRYRUN] LIMIT {symbol} {side} qty={qty} px={price} RO={reduce_only}")
            return {"status":"dryrun"}
        oid = str(uuid.uuid4())[:20]
        return _retry(lambda: self.sess.place_order(
            category="linear", symbol=symbol, side=side, orderType="Limit",
            qty=str(qty), price=str(price), reduceOnly=reduce_only, timeInForce="GTC",
            orderLinkId=oid
        ))

    def place_reduce_limit(self, symbol: str, side: str, qty: float, price: float):
        return self.place_limit(symbol, side, qty, price, reduce_only=True)

    def set_stop_loss_mark(self, symbol: str, stop_price: float):
        if DRY_RUN:
            print(f"[DRYRUN] SL SET {symbol} stop={stop_price}")
            return {"status":"dryrun"}
        try:
            tick = self.get_filters(symbol)["tickSize"]
            stop_price = round_price(stop_price, tick)
            return self.sess.set_trading_stop(category="linear", symbol=symbol, stopLoss=str(stop_price))
        except Exception as e:
            print("[ERROR set_trading_stop]", symbol, stop_price, e)
            return {"status":"error","error":str(e)}

# ===========================
# ===== Indicators/Helpers ==
# ===========================
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-prev_close).abs(),
        (df["low"]-prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def bollinger(df: pd.DataFrame, n: int=20, k: float=2.0) -> pd.DataFrame:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=0)
    df["bb_ma"] = ma
    df["bb_up"] = ma + k*sd
    df["bb_lo"] = ma - k*sd
    base = ma.replace(0, np.nan)
    df["bb_w"]  = (df["bb_up"] - df["bb_lo"]) / base
    return df

def round_price(px: float, tick: float) -> float:
    if tick <= 0: return px
    n = round(px / tick)
    return round(n * tick, 8)

def quantize_qty(raw: float, step: float, min_qty: float) -> float:
    if raw < min_qty: return 0.0
    if step <= 0: return raw
    q = math.floor(raw / step) * step
    return q if q >= min_qty else 0.0

def rr(entry: float, sl: float, tp: float, side: str) -> float:
    if side == "long":
        risk = max(entry - sl, 1e-9); reward = max(tp - entry, 1e-9)
    else:
        risk = max(sl - entry, 1e-9); reward = max(entry - tp, 1e-9)
    return reward / risk

def simple_vpvr_poc(df: pd.DataFrame, bins: int=50) -> Optional[float]:
    px = ((df["high"] + df["low"]) / 2.0).values
    vol = df["volume"].values
    if len(px) < 10: return None
    hist, edges = np.histogram(px, bins=bins, weights=vol)
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx+1]) / 2.0)

def detect_ob_levels(df: pd.DataFrame, look: int=240) -> List[float]:
    body = (df["close"] - df["open"]).abs()
    th = body.tail(look).mean() * 1.2
    idx = body[body > th].index
    levels: List[float] = []
    for i in idx:
        mid = (df.loc[i, "open"] + df.loc[i, "close"]) / 2.0
        levels.append(round(float(mid), 2))
    return sorted(set(levels))

def detect_fvg_levels(df: pd.DataFrame, look: int=240) -> List[Tuple[float,float]]:
    levels=[]
    for i in range(2, len(df)):
        lo2, hi2 = df.iloc[i-2]["low"], df.iloc[i-2]["high"]
        lo0, hi0 = df.iloc[i]["low"], df.iloc[i]["high"]
        if lo0 > hi2: levels.append((hi2, lo0))   # 상승 FVG
        if hi0 < lo2: levels.append((hi0, lo2))   # 하락 FVG
    return levels

def pivot_trend(df: pd.DataFrame, look:int=24) -> int:
    if len(df) < max(look, 25): return 0
    seg = df.tail(look).copy()
    ema20 = ema(seg["close"], 20)
    slope = (ema20.iloc[-1] - ema20.iloc[-5]) / max(abs(ema20.iloc[-5]), 1e-9)
    highs = seg["high"].values; lows  = seg["low"].values
    HH = highs[-1] >= (highs[:-1].max() * 0.999)
    HL = lows[-1]  >= (lows[:-1].min()  * 1.001)
    LL = lows[-1]  <= (lows[:-1].min()  * 1.001)
    LH = highs[-1] <= (highs[:-1].max() * 0.999)
    up = (HH and HL); dn = (LL and LH)
    if up and not dn: return +1
    if dn and not up: return -1
    if up and dn:
        return +1 if slope > 0 else (-1 if slope < 0 else 0)
    return 0

def bb_phase(df: pd.DataFrame, n:int=20, k:float=2.0) -> Tuple[float, float]:
    df = df.copy()
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=0)
    up, lo = ma + k*sd, ma - k*sd
    pctB = (df["close"].iloc[-1] - lo.iloc[-1]) / max(up.iloc[-1] - lo.iloc[-1], 1e-9)
    mid_slope = (ma.iloc[-1] - ma.iloc[-5]) / max(abs(ma.iloc[-5]), 1e-9)
    return pctB, mid_slope

def _apply_conf_weight(base_qty: float, conf_count: int, step: float, cap: float) -> float:
    weight = min(1.0 + step * max(0, int(conf_count)), cap)
    return base_qty * weight

def _near_confluence_4h(price: float, d4: pd.DataFrame, tol_k: float=0.6) -> Tuple[int, List[str]]:
    notes = []; add = 0
    look = 240
    ob  = detect_ob_levels(d4.tail(look))
    fvg = detect_fvg_levels(d4.tail(look))
    vpoc = simple_vpvr_poc(d4.tail(look), bins=40)
    atr4 = float(atr(d4, 14).iloc[-1] or 0)
    tol  = max(atr4 * tol_k, price * 0.001)
    for lv in ob:
        if abs(price - lv) <= tol: add += 1; notes.append(f"OB≈{lv:.0f}"); break
    for lo, hi in fvg:
        if (lo - tol) <= price <= (hi + tol): add += 1; notes.append("FVG zone"); break
    if vpoc and abs(price - vpoc) <= tol: add += 1; notes.append("VPOC zone")
    return add, notes

# ===========================
# ===== BTC동조 + 서브엔진 ==
# ===========================
class BBEngine:
    def __init__(self, api: Bybit):
        self.api = api
        self.state: Dict[str, Dict] = {}
        self.rebreak: Dict[str, Dict] = {}

    def _frames(self, sym: str):
        d15 = self.api.fetch_ohlcv(sym,"15m",300)
        d30 = self.api.fetch_ohlcv(sym,"30m",300)
        d1h = self.api.fetch_ohlcv(sym,"1h",300)
        d4  = self.api.fetch_ohlcv(sym,"4h",240)
        if d15.empty or d30.empty or d1h.empty or d4.empty: return None
        d15 = bollinger(d15.copy(), BB_PERIOD, BB_STD)
        d30 = bollinger(d30.copy(), BB_PERIOD, BB_STD); d30["ema20"] = ema(d30["close"], 20)
        d1h["atr"] = atr(d1h.copy(), ATR_LEN_1H)
        return (d15, d30, d1h, d4)

    def _btc_frames_15m(self) -> Optional[pd.DataFrame]:
        try:
            d15_btc = self.api.fetch_ohlcv(LEADER, "15m", 200)
            if d15_btc.empty or len(d15_btc) < 30: return None
            return bollinger(d15_btc.copy(), BB_PERIOD, BB_STD)
        except Exception:
            return None

    def _btc_in_sync(self, side: str, d15_btc: Optional[pd.DataFrame]) -> bool:
        if d15_btc is None or len(d15_btc) < 2: return False
        b  = d15_btc.iloc[-2]
        c  = float(b["close"])
        up = float(b["bb_up"]); lo = float(b["bb_lo"])
        eps = BTC_SYNC_EPS_PCT
        if side == "long"  and (c >= up * (1.0 - eps)): return True
        if side == "short" and (c <= lo * (1.0 + eps)): return True
        return False

    def mode_range(self, d30: pd.DataFrame) -> bool:
        bbw = float(d30["bb_w"].iloc[-1] or 0)
        return bbw <= RANGE_BBWIDTH_THR

    def _came_from_squeeze(self, d15: pd.DataFrame, j:int) -> bool:
        win = d15["bb_w"].iloc[max(0, j-SQUEEZE_LOOKBACK):j]
        if len(win) < 5: return False
        avg = float(win.mean() or 0)
        prev = float(d15["bb_w"].iloc[j-1] or 0)
        now  = float(d15["bb_w"].iloc[j]   or 0)
        return (prev <= avg*SQUEEZE_K) and (now >= avg*SQUEEZE_K*EXPAND_MULT)

    def _early_expansion_15m(self, d15: pd.DataFrame, d30: pd.DataFrame, d1h: pd.DataFrame) -> Optional[Dict]:
        if not USE_15M_EE_TRIGGER or len(d15)<40 or len(d30)<25: return None
        j = -2
        bbw = d15["bb_w"]; avg = float(bbw.iloc[-20:].mean() or 0)
        bbw_prev = float(bbw.iloc[-3] or 0); bbw_now  = float(bbw.iloc[-2] or 0)
        if not (bbw_prev <= avg*SQUEEZE_K and bbw_now >= avg*EE15M_BBW_EXPAND_K): return None
        prev = d15.iloc[j]; price = float(prev["close"]); up=float(prev["bb_up"]); lo=float(prev["bb_lo"])
        ema20_30 = d30["ema20"]
        slope30 = (ema20_30.iloc[-1] - ema20_30.iloc[-5]) / max(abs(ema20_30.iloc[-5]),1e-9)
        side=None
        if price>up and slope30>0: side="long"
        elif price<lo and slope30<0: side="short"
        else: return None
        vol_avg5 = d15["volume"].iloc[-6:-1].mean()
        if float(prev["volume"]) < max(1.05*vol_avg5, 1.0): return None
        atr15 = float(atr(d15, ATR_LEN_15M).iloc[-2] or 0)
        if BBW_15M_MIN>0 and float(d15["bb_w"].iloc[-2])<BBW_15M_MIN: return None
        if ATR_15M_MIN>0 and atr15<ATR_15M_MIN: return None
        if side=="long":
            sl = min(float(prev["low"]), price - 0.5*atr15)
        else:
            sl = max(float(prev["high"]), price + 0.5*atr15)
        risk = abs(price-sl); tp1 = price + 2.0*risk if side=="long" else price - 2.0*risk
        R = rr(price, sl, tp1, side)
        if R < EE15M_MIN_RR: return None
        tp2 = price + 3.0*risk if side=="long" else price - 3.0*risk
        return {"side":side,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":R,"ee_flag":True,
                "came_from_squeeze":True,"range_mode":self.mode_range(d30),
                "bbw":float(d15["bb_w"].iloc[-2])}

    def _entry_confirmation_ok(self, d15: pd.DataFrame, side: str) -> bool:
        if len(d15) < 3: return False
        b1 = d15.iloc[-3]; b2 = d15.iloc[-2]
        if side=="long":
            c1 = float(b1["close"]) > float(b1["bb_up"])
            c2 = float(b2["close"]) > float(b2["bb_up"])
        else:
            c1 = float(b1["close"]) < float(b1["bb_lo"])
            c2 = float(b2["close"]) < float(b2["bb_lo"])
        if SUB_ENTRY_CONFIRM_MODE == "FIRST_ONLY": return c2
        if SUB_ENTRY_CONFIRM_MODE == "TWO_ONLY":   return (c1 and c2)
        if c1 and c2: return True
        rng = max(float(b2["high"]) - float(b2["low"]), 1e-9)
        body = abs(float(b2["close"]) - float(b2["open"])) / rng
        vol_avg5 = d15["volume"].iloc[-7:-2].mean()
        strong = (body >= STRONG_BAR_BODY_PCT) and (float(b2["volume"]) >= STRONG_BAR_VOL_MULT * max(vol_avg5,1e-9))
        return c2 and strong

    def build_signal(self, sym: str, leader_px: float, main_active: bool) -> Optional[Dict]:
        pack = self._frames(sym)
        if pack is None:
            log_skip_reason(sym, "frames_empty"); return None
        d15, d30, d1h, d4 = pack
        if len(d15) < 50:
            log_skip_reason(sym, "insufficient_bars"); return None

        d15_btc = self._btc_frames_15m()

        # 1) 15m Early Expansion
        ee = self._early_expansion_15m(d15, d30, d1h)
        if ee:
            add_conf, _ = _near_confluence_4h(ee["entry"], d4, tol_k=0.6)
            ee["conf_count"] = add_conf + 1
            ee["symbol"]=sym
            if BTC_SYNC_REQUIRED and not self._btc_in_sync(ee["side"], d15_btc):
                log_skip_reason(sym, "btc_sync_fail_ee", f"side={ee['side']}")
            else:
                return ee

        # 2) 일반 돌파/이탈
        j = -2
        prev = d15.iloc[j]
        price = float(prev["close"])
        up, lo = float(prev["bb_up"]), float(prev["bb_lo"])
        mid = float(prev["bb_ma"])
        bbw_now = float(d15["bb_w"].iloc[j] or 0)

        # 거래량 체크(수축구간 완화/확장구간 강화)
        mult = VOL_MULT_SUB_MIN if bbw_now < 0.01 else (1.10 if bbw_now < 0.02 else VOL_MULT_SUB_MAX)
        vol_window = d15["volume"].iloc[j-4:j+1]
        vol_avg5 = float(vol_window.mean()) if len(vol_window)==5 else float(d15["volume"].iloc[:j+1].tail(5).mean())
        vol_ok = float(prev["volume"]) >= mult * (vol_avg5 or 1e-9)
        if not vol_ok:
            log_skip_reason(sym, "vol_fail", f"bbw={bbw_now:.4f}, mult={mult:.2f}"); return None

        came_from_squeeze = self._came_from_squeeze(d15, len(d15)+j)

        # 후보: 상단 돌파 롱 / 하단 돌파 숏
        cand=[]
        atr1h = float(d1h["atr"].iloc[-1] or 0)
        if price>up: cand.append(("long", price))
        if price<lo: cand.append(("short", price))
        if not cand:
            log_skip_reason(sym, "no_candidate"); return None

        # 진입 확인
        side_try = "long" if price>up else "short"
        if not self._entry_confirmation_ok(d15, side_try):
            log_skip_reason(sym, "entry_confirm_fail", SUB_ENTRY_CONFIRM_MODE); return None

        # 손절/TP 산출
        best=None; best_rr=-1
        for side, e in cand:
            if SUB_STOP_MODE.upper()=="BAND_RECOVERY":
                if side=="long":
                    band_sl = up * (1.0 - BAND_RECOVERY_EPS_PCT)
                    atr_sl  = e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)
                    sl = min(band_sl, atr_sl, e*(1.0 - SUB_MIN_STOP_DIST_PCT))
                else:
                    band_sl = lo * (1.0 + BAND_RECOVERY_EPS_PCT)
                    atr_sl  = e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)
                    sl = max(band_sl, atr_sl, e*(1.0 + SUB_MIN_STOP_DIST_PCT))
            elif SUB_STOP_MODE.upper()=="MIDLINE":
                if side=="long": sl = max(e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid*(1.0 - MID_BUF_PCT))
                else:            sl = min(e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid*(1.0 + MID_BUF_PCT))
            else:  # "ATR"
                sl = (e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)) if side=="long" else (e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K))
            risk = abs(e - sl)
            tp1 = (e + 2*risk) if side=="long" else (e - 2*risk)
            tp2 = (e + 3*risk) if side=="long" else (e - 3*risk)
            R = rr(e, sl, tp1, side)
            if R > best_rr:
                best_rr = R; best = (side, e, sl, tp1, tp2)

        if (best is None) or (best_rr < RR_GATE_SUB):
            log_skip_reason(sym, "rr_gate_fail", f"rr={best_rr:.2f}"); return None

        side, entry, sl, tp1, tp2 = best

        # BTC 동조 필수
        # build_signal(...) 안
        if d15_btc is None or (isinstance(d15_btc, pd.DataFrame) and d15_btc.empty):
            d15_btc = self._btc_frames_15m()


        if BTC_SYNC_REQUIRED and not self._btc_in_sync(side, d15_btc):
            log_skip_reason(sym, "btc_sync_fail", f"side={side}"); return None

        # 컨플루언스 카운트
        conf_count = 0
        if came_from_squeeze: conf_count += 1
        if best_rr >= (RR_GATE_SUB + 0.3): conf_count += 1
        add4h, _ = _near_confluence_4h(entry, d4, tol_k=0.6); conf_count += add4h

        # 리테스트 가격
        retest_px = None
        if RETEST_ENABLE:
            if RETEST_TOUCH_MODE=="band": retest_px = up if side=="long" else lo
            elif RETEST_TOUCH_MODE=="mid": retest_px = mid
            else: retest_px = max(up, mid) if side=="long" else min(lo, mid)

        return {"symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":best_rr,
                "size_factor":1.0,"range_mode": self.mode_range(d30),
                "mid": mid, "up":up, "lo":lo, "came_from_squeeze":came_from_squeeze,
                "retest_px": retest_px, "conf_count": conf_count,
                "bbw": bbw_now, "rebreak_ok": False, "ee_flag": False}

    # --- 사이징(서브) ---
    def sizing(self, sym: str, size_factor: float, conf_count:int) -> Tuple[float, Dict]:
        f = self.api.get_filters(sym)
        base = SUB_FIXED_QTY.get(sym, SUB_FIXED_QTY.get("DEFAULT", 0.4))
        raw_weighted  = _apply_conf_weight(base, conf_count, CONF_WEIGHT_STEP_SUB, CONF_MAX_WEIGHT_SUB)
        raw = max(base*0.25, raw_weighted) * size_factor
        if USE_NOTIONAL_BUDGET:
            px_ref_df = self.api.fetch_ohlcv(sym, "15m", 2)
            if not px_ref_df.empty:
                px_ref = float(px_ref_df["close"].iloc[-1])
                budget = PER_SYMBOL_BUDGET_USDT.get(sym, PER_SYMBOL_BUDGET_USDT.get("DEFAULT", 15000.0))
                max_qty = budget / max(px_ref, 1e-9)
                raw = min(raw, max_qty)
        qty  = quantize_qty(raw, f["qtyStep"], f["minQty"])
        return qty, f

    def after_entry_virtual(self, sym, side, entry, tp1, tp2, sl, qty, sub_mode, retest_px, meta):
        self.state[sym] = {"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"tp1_done":False,
                           "init_size":qty,"breakeven_set":False,"mode":sub_mode,
                           "split_left": (SPLIT_ENTRY_FIRST_PCT < 1.0), "retest_px":retest_px,
                           "retest_deadline": RETEST_TIMEOUT_BARS, "add_done": False,
                           "ee_flag": meta.get("ee_flag", False), "trail_on": False}

    def after_entry(self, sym, side, entry, tp1, tp2, sl, qty, filters, sub_mode, retest_px, meta):
        # 체결 직후 포지션 조회는 약간 지연될 수 있어 재시도
        pos = None
        for _ in range(3):
            pos = self.api.get_position(sym)
            if pos: break
            time.sleep(0.25)
        if not pos: return
        sl = round_price(sl, self.api.get_filters(sym)["tickSize"])
        self.state[sym] = {"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"tp1_done":False,
                           "init_size":abs(float(pos["size"])),"breakeven_set":False,"mode":sub_mode,
                           "split_left": (SPLIT_ENTRY_FIRST_PCT < 1.0), "retest_px":retest_px,
                           "retest_deadline": RETEST_TIMEOUT_BARS, "add_done": False,
                           "ee_flag": meta.get("ee_flag", False), "trail_on": False}
        self.api.set_stop_loss_mark(sym, sl)
        opp = "Sell" if side=="long" else "Buy"
        px1 = round_price(tp1, self.api.get_filters(sym)["tickSize"])
        tp_qty = quantize_qty(qty*TP1_RATIO_SUB, filters["qtyStep"], filters["minQty"])
        if tp_qty > 0:
            self.api.place_reduce_limit(sym, opp, tp_qty, px1)

    def _maybe_trail_after_tp1(self, sym: str, api: Bybit):
        st = self.state[sym]
        if not st.get("trail_on", False): return
        df = api.fetch_ohlcv(sym, "15m", max(20, SUB_TP2_TRAIL_LOOKBACK+5))
        if df.empty: return
        atr15 = float(atr(df, ATR_LEN_15M).iloc[-1] or 0)
        price = float(df["close"].iloc[-1])
        cur_sl = st["sl"]
        if st["side"]=="long":
            new_sl = max(cur_sl, price - SUB_TP2_TRAIL_ATR_K*atr15) if SUB_TP2_TRAIL_MODE.upper()=="ATR" \
                     else max(cur_sl, df["low"].iloc[-SUB_TP2_TRAIL_LOOKBACK:].min())
        else:
            new_sl = min(cur_sl, price + SUB_TP2_TRAIL_ATR_K*atr15) if SUB_TP2_TRAIL_MODE.upper()=="ATR" \
                     else min(cur_sl, df["high"].iloc[-SUB_TP2_TRAIL_LOOKBACK:].max())
        if (st["side"]=="long" and new_sl>cur_sl) or (st["side"]=="short" and new_sl<cur_sl):
            st["sl"] = round_price(new_sl, api.get_filters(sym)["tickSize"])
            api.set_stop_loss_mark(sym, st["sl"])

    def _maybe_add_on_retest(self, sym: str, api: Bybit):
        st = self.state[sym]
        if (not SPLIT_ENTRY_ENABLE) or st["add_done"] or (not st["split_left"]) or (not RETEST_ENABLE): return
        if st["retest_px"] is None:
            st["add_done"]=True; return
        df = api.fetch_ohlcv(sym,"15m",2)
        if df.empty: return
        px = float(df["close"].iloc[-1])
        f = api.get_filters(sym)

        def _commit_add(side, add_qty):
            api.place_market(sym, "Buy" if side=="long" else "Sell", add_qty, reduce_only=False)
            new_entry = (st["entry"]*st["init_size"] + px*add_qty) / (st["init_size"]+add_qty)
            st["entry"] = new_entry; st["init_size"] += add_qty
            if st.get("breakeven_set", False):
                be = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                api.set_stop_loss_mark(sym, be); st["sl"] = be

        if st["side"]=="long":
            if px > st["entry"]*(1.0 + RETEST_MAX_DRIFT_PCT): st["retest_deadline"] -= 1
            if px <= st["retest_px"]*(1.0 + RETEST_TOUCH_EPS):
                add_qty = quantize_qty(st["init_size"]*(1.0 - SPLIT_ENTRY_FIRST_PCT), f["qtyStep"], f["minQty"])
                if add_qty > 0: _commit_add("long", add_qty)
                st["add_done"]=True
        else:
            if px < st["entry"]*(1.0 - RETEST_MAX_DRIFT_PCT): st["retest_deadline"] -= 1
            if px >= st["retest_px"]*(1.0 - RETEST_TOUCH_EPS):
                add_qty = quantize_qty(st["init_size"]*(1.0 - SPLIT_ENTRY_FIRST_PCT), f["qtyStep"], f["minQty"])
                if add_qty > 0: _commit_add("short", add_qty)
                st["add_done"]=True

        # (옵션) 리테스트 실패 시 추격
        if st["retest_deadline"] <= 0 and (not st["add_done"]):
            pass  # 필요시 추격진입 로직 유지/생략

        if st["retest_deadline"] <= 0: st["add_done"]=True

    def poll_manage_virtual(self, sym: str, api: Bybit):
        if sym not in self.state: return
        st = self.state[sym]
        df = api.fetch_ohlcv(sym,"15m",2)
        if df.empty: return
        price = float(df["close"].iloc[-1])

        self._maybe_add_on_retest(sym, api)

        hit_tp1 = price>=st["tp1"] if st["side"]=="long" else price<=st["tp1"]
        hit_sl  = price<=st["sl"]  if st["side"]=="long" else price>=st["sl"]

        if (not st["tp1_done"]) and hit_tp1:
            st["tp1_done"]=True; st["trail_on"]=True
            if not st["breakeven_set"]:
                st["sl"] = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                st["breakeven_set"]=True
            log_trade_event("TP1", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"]*TP1_RATIO_SUB,
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], mode=st["mode"],
                            extra="virtual SL→BE", ee_flag=st.get("ee_flag",False))

        if st.get("trail_on", False):
            self._maybe_trail_after_tp1(sym, api)

        if hit_sl or ((st["tp1_done"]) and (price<=st["sl"] if st["side"]=="long" else price>=st["sl"])):
            log_trade_event("EXIT", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], mode=st["mode"],
                            exit_price=price, extra="virtual_exit", ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None)

    def poll_manage(self, sym: str, api: Bybit):
        if sym not in self.state: return
        st = self.state[sym]
        pos = api.get_position(sym)
        if not pos or abs(float(pos.get("size",0) or 0))==0:
            df_last = api.fetch_ohlcv(sym,"15m",2)
            exit_px = float(df_last["close"].iloc[-1]) if not df_last.empty else None
            log_trade_event("EXIT", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], mode=st["mode"],
                            exit_price=exit_px, extra="flat", ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None); return

        self._maybe_add_on_retest(sym, api)

        # ★실계좌: 포지션 수량 감소 감지 → TP1 체결로 간주 → BE 전환
        cur_sz = abs(float(pos["size"]))
        if (not st["tp1_done"]) and (cur_sz < st["init_size"]):
            st["tp1_done"]=True; st["trail_on"]=True
            if not st["breakeven_set"]:
                be = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                api.set_stop_loss_mark(sym, be)
                st["sl"]=be; st["breakeven_set"]=True
                realized = st["init_size"] - cur_sz
                log_trade_event("TP1", symbol=sym, engine="sub", side=st["side"], qty=realized,
                                entry=st["entry"], sl=be, tp1=st["tp1"], tp2=st["tp2"], mode=st["mode"],
                                extra="by_fill", ee_flag=st.get("ee_flag",False))

        if st.get("trail_on", False):
            self._maybe_trail_after_tp1(sym, api)

# ===========================
# ====== Main (Swing) =======
# ===========================
class SwingEngine:
    def __init__(self, api: Bybit):
        self.api = api
        self.state: Dict[str, Dict] = {}

    def _frames(self, sym: str):
        d4 = self.api.fetch_ohlcv(sym,"4h",400)
        d1 = self.api.fetch_ohlcv(sym,"1d",400)
        d1h= self.api.fetch_ohlcv(sym,"1h",400)
        if d4.empty or d1.empty or d1h.empty: return None
        d4["ema200"]= ema(d4["close"], MAIN_EMA_LEN)
        d4["atr"]   = atr(d4.copy(), MAIN_ATR_LEN_4H)
        return d4, d1, d1h

    def confluence_score(self, d4: pd.DataFrame, d1: pd.DataFrame) -> int:
        score = 0
        score += 1
        if detect_ob_levels(d4.tail(240)): score += 1
        if detect_fvg_levels(d4.tail(240)): score += 1
        rng_now = abs(d1["close"].iloc[-1]-d1["open"].iloc[-1])
        rng_avg = abs(d1["close"]-d1["open"]).tail(20).mean()
        if rng_now >= 0.6*(rng_avg or rng_now): score += 1
        hh = d4["high"].iloc[-20:].max(); ll = d4["low"].iloc[-20:].min(); c  = d4["close"].iloc[-1]
        if (c>=hh) or (c<=ll): score += 1
        vpoc = simple_vpvr_poc(d4.tail(240), bins=40)
        atr4 = float(d4["atr"].iloc[-1] or 0)
        if vpoc and abs(float(c)-vpoc) <= max(atr4*0.5, float(c)*0.001): score += 1
        return score

    def _early_expansion_4h(self, d4: pd.DataFrame, d1h: pd.DataFrame) -> Optional[Dict]:
        if not MAIN_EE_ENABLE or len(d4) < max(MAIN_EE_BBW_LOOKBACK+5, 30): return None
        d = d4.copy()
        ma = d["close"].rolling(20).mean()
        sd = d["close"].rolling(20).std(ddof=0)
        up = ma + 2.0*sd; lo = ma - 2.0*sd
        bbw = (up - lo) / ma.replace(0, np.nan)
        j = -2
        price = float(d["close"].iloc[j]); upj=float(up.iloc[j]); loj=float(lo.iloc[j])
        win = bbw.iloc[(-MAIN_EE_BBW_LOOKBACK-2):j]
        if len(win) < 10: return None
        avg_bbw = float(win.mean() or 0)
        bbw_prev = float(bbw.iloc[j-1] or 0); bbw_now  = float(bbw.iloc[j]   or 0)
        if not (bbw_prev <= avg_bbw * MAIN_EE_SQUEEZE_K and bbw_now >= avg_bbw * MAIN_EE_EXPAND_K): return None
        side=None
        if price > upj: side="long"
        elif price < loj: side="short"
        else: return None
        if MAIN_EE_ALIGN_1H_GUARD and (d1h is not None and not d1h.empty and len(d1h) >= 210):
            d1h_local = d1h.copy(); d1h_local["ema200"] = ema(d1h_local["close"], 200)
            p1h = float(d1h_local["close"].iloc[-1]); e200 = float(d1h_local["ema200"].iloc[-1])
            if side=="long" and not (p1h >= e200): return None
            if side=="short" and not (p1h <= e200): return None
        atr4 = float(atr(d, MAIN_EE_ATR_LEN).iloc[j] or 0)
        hi   = float(d["high"].iloc[j]); lo_  = float(d["low"].iloc[j])
        if MAIN_EE_USE_NEAR_SL:
            if side=="long":
                sl = min(lo_, price - MAIN_EE_SL_ATR_K*atr4)
            else:
                sl = max(hi, price + MAIN_EE_SL_ATR_K*atr4)
        else:
            sl = price - MAIN_SL_ATR_K*atr4 if side=="long" else price + MAIN_SL_ATR_K*atr4
        risk = abs(price - sl); tp1 = price + 2.0*risk if side=="long" else price - 2.0*risk
        R = rr(price, sl, tp1, side)
        if R < MAIN_EE_MIN_RR: return None
        tp2 = price + 3*risk if side=="long" else price - 3*risk
        return {"symbol":None,"side":side,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":R,
                "ee_flag":True, "size_factor": (MAIN_EE_SIZE_FACTOR if MAIN_EE_ALLOW_LOW_CONF else 1.0)}

    def build_signal(self, sym: str) -> Optional[Dict]:
        pack = self._frames(sym)
        if pack is None: return None
        d4, d1, d1h = pack
        price = float(d4["close"].iloc[-1]); ema200 = float(d4["ema200"].iloc[-1]); atr4 = float(d4["atr"].iloc[-1] or 0)

        # EE 우선
        ee_sig = self._early_expansion_4h(d4, d1h)
        if ee_sig is not None:
            ee_sig["symbol"] = sym
            return ee_sig

        trend = pivot_trend(d4, look=24)
        side = "long" if price>ema200 else "short"

        # EMA200 가드
        d1h_local = d1h.copy(); d1h_local["ema200"] = ema(d1h_local["close"], 200)
        p1h = float(d1h_local["close"].iloc[-1]); e200 = float(d1h_local["ema200"].iloc[-1])
        dist_pct = (p1h - e200)/max(e200,1e-9)
        slope = (d1h_local["ema200"].iloc[-1] - d1h_local["ema200"].iloc[-5]) / max(abs(d1h_local["ema200"].iloc[-5]),1e-9)
        neutral_band = max(EMA_NEUTRAL_PCT, (atr4/price if price else 0)*EMA_NEUTRAL_ATR_K)
        ok_guard = (abs(dist_pct) <= neutral_band and slope >= EMA_SLOPE_MIN) or \
                   (side=="long" and dist_pct >= SWITCH_UP_PCT and slope>=EMA_SLOPE_MIN) or \
                   (side=="short" and dist_pct <= -SWITCH_DOWN_PCT and slope>=EMA_SLOPE_MIN)
        if not ok_guard:
            log_skip_reason(sym,"main_ema_guard", f"dist={dist_pct:.4f}, slope={slope:.4f}, nb={neutral_band:.4f}")
            return None

        pctB, mid_slope = bb_phase(d4)
        if (side == "long" and (mid_slope < 0 and pctB < 0.5)) or (side == "short" and (mid_slope > 0 and pctB > 0.5)):
            log_skip_reason(sym, "main_bb_phase_guard"); return None

        score = self.confluence_score(d4,d1)
        size_factor = 1.0 if score>=2 else (MAIN_EE_SIZE_FACTOR if MAIN_EE_ALLOW_LOW_CONF else None)
        if size_factor is None:
            log_skip_reason(sym, "main_score_low", f"score={score}"); return None

        sl = price - MAIN_SL_ATR_K*atr4 if side=="long" else price + MAIN_SL_ATR_K*atr4
        risk = abs(price - sl)
        tp1 = price + 2*atr4 if side=="long" else price - 2*atr4
        tp2 = price + 3*atr4 if side=="long" else price - 3*atr4
        R  = rr(price, sl, tp1, side)
        if R < MAIN_RR_GATE:
            log_skip_reason(sym, "main_rr_fail", f"rr={R:.2f}")
            return None

        conf_count = 0
        if (side=="long" and trend>0) or (side=="short" and trend<0): conf_count += 1
        if R >= (MAIN_RR_GATE + 0.3): conf_count += 1
        add, _ = _near_confluence_4h(price, d4, tol_k=0.6); conf_count += add

        return {"symbol":sym,"side":side,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":R,
                "conf_count": conf_count, "size_factor": size_factor, "ee_flag": False}

    # --- 사이징(메인) ---
    def sizing(self, sym: str, conf_count:int, size_factor: float=1.0) -> Tuple[float, Dict]:
        f = self.api.get_filters(sym)
        base = MAIN_FIXED_QTY.get(sym, MAIN_FIXED_QTY.get("DEFAULT", 0.5))
        raw_weighted  = _apply_conf_weight(base, conf_count, CONF_WEIGHT_STEP_MAIN, CONF_MAX_WEIGHT_MAIN)
        raw = max(base*0.25, raw_weighted) * size_factor
        if USE_NOTIONAL_BUDGET:
            px_ref_df = self.api.fetch_ohlcv(sym, "15m", 2)
            if not px_ref_df.empty:
                px_ref = float(px_ref_df["close"].iloc[-1])
                budget = PER_SYMBOL_BUDGET_USDT.get(sym, PER_SYMBOL_BUDGET_USDT.get("DEFAULT", 15000.0))
                max_qty = budget / max(px_ref, 1e-9)
                raw = min(raw, max_qty)
        qty  = quantize_qty(raw, f["qtyStep"], f["minQty"])
        return qty, f

    def after_entry_virtual(self, sym, side, entry, tp1, tp2, sl, qty, ee_flag=False):
        self.state[sym] = {"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"tp1_done":False,
                           "init_size":qty,"breakeven_set":False,"mode":"main","ee_flag":ee_flag}
        log_trade_event("ENTRY", symbol=sym, engine="main", side=side, qty=qty,
                        entry=entry, sl=sl, tp1=tp1, tp2=tp2, mode="main", extra="virtual_entry")

    def after_entry(self, sym, side, entry, tp1, tp2, sl, qty, filters, ee_flag=False):
        # 체결 직후 포지션 조회 재시도
        pos = None
        for _ in range(3):
            pos = self.api.get_position(sym)
            if pos: break
            time.sleep(0.25)
        if not pos: return
        sl = round_price(sl, self.api.get_filters(sym)["tickSize"])
        tp_qty = quantize_qty(qty * MAIN_TP1_RATIO, filters["qtyStep"], filters["minQty"])
        self.state[sym] = {"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"tp1_done":False,
                           "init_size":abs(float(pos["size"])),"breakeven_set":False,"mode":"main",
                           "tp1_qty": tp_qty, "ee_flag": ee_flag}
        self.api.set_stop_loss_mark(sym, sl)
        opp = "Sell" if side=="long" else "Buy"
        px1 = round_price(tp1, self.api.get_filters(sym)["tickSize"])
        if tp_qty > 0:
            self.api.place_reduce_limit(sym, opp, tp_qty, px1)

    def _manage_common(self, sym: str, price: float, virtual: bool):
        st = self.state[sym]
        hit_tp1 = price>=st["tp1"] if st["side"]=="long" else price<=st["tp1"]
        hit_sl  = price<=st["sl"]  if st["side"]=="long" else price>=st["sl"]
        if (not st["tp1_done"]) and virtual and hit_tp1:
            st["tp1_done"]=True
            if not st["breakeven_set"]:
                st["sl"] = round_price(st["entry"], self.api.get_filters(sym)["tickSize"])
                st["breakeven_set"]=True
            log_trade_event("TP1", symbol=sym, engine="main", side=st["side"],
                            qty=st.get("tp1_qty", st["init_size"] * MAIN_TP1_RATIO),
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"),
                            mode="main", extra="virtual SL→BE", ee_flag=st.get("ee_flag",False))
        if hit_sl or ((st["tp1_done"]) and (price<=st["sl"] if st["side"]=="long" else price>=st["sl"])) :
            log_trade_event("EXIT", symbol=sym, engine="main", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"), mode="main",
                            exit_price=price, extra=("virtual_exit" if virtual else "flat"),
                            ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None)

    def poll_manage_virtual(self, sym: str, api: Bybit):
        if sym not in self.state: return
        df = api.fetch_ohlcv(sym,"15m",2)
        if df.empty: return
        price = float(df["close"].iloc[-1])
        self._manage_common(sym, price, virtual=True)

    def poll_manage(self, sym: str, api: Bybit):
        if sym not in self.state: return
        st = self.state[sym]
        pos = api.get_position(sym)
        if not pos or abs(float(pos.get("size",0) or 0))==0:
            df_last = api.fetch_ohlcv(sym,"15m",2)
            exit_px = float(df_last["close"].iloc[-1]) if not df_last.empty else None
            log_trade_event("EXIT", symbol=sym, engine="main", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"), mode="main",
                            exit_price=exit_px, extra="flat", ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None); return

        # ★실계좌: 수량 감소 감지 → TP1 체결로 간주 → BE 전환
        cur_sz = abs(float(pos["size"]))
        if (not st["tp1_done"]) and (cur_sz < st["init_size"]):
            st["tp1_done"] = True
            if not st["breakeven_set"]:
                be = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                api.set_stop_loss_mark(sym, be)
                st["sl"] = be
                st["breakeven_set"] = True
                realized = st["init_size"] - cur_sz
                log_trade_event("TP1", symbol=sym, engine="main", side=st["side"],
                                qty=realized, entry=st["entry"], sl=be, tp1=st["tp1"], tp2=st.get("tp2"),
                                mode="main", extra="by_fill", ee_flag=st.get("ee_flag",False))

        df = api.fetch_ohlcv(sym,"15m",2)
        if df.empty: return
        price = float(df["close"].iloc[-1])
        self._manage_common(sym, price, virtual=False)

# ===========================
# ========== MAIN ===========
# ===========================
def cooldown_ok(engine: str, sym: str, direction: str, bar_ts: int) -> bool:
    k = (engine, sym)   # ★방향 제거
    last = LAST_EXEC_BAR.get(k, None)
    if last is not None and last == bar_ts:
        log_skip_reason(sym, "cooldown_same_bar", engine)
        return False
    return True

def register_exec(engine: str, sym: str, direction: str, bar_ts: int):
    LAST_EXEC_BAR[(engine, sym)] = bar_ts

def record_loss(engine: str, sym: str, direction: str):
    key=(engine, sym)
    rec = LAST_LOSS_DIR.get(key, {"dir":direction,"losses":0,"dead":0})
    if rec["dir"] == direction:
        rec["losses"] += 1
    else:
        rec = {"dir":direction,"losses":1,"dead":0}
    if rec["losses"] >= TILT_MAX_LOSSES_DIR:
        rec["dead"] = TILT_DEAD_BARS
    LAST_LOSS_DIR[key]=rec

def tick_dead(engine: str, sym: str) -> bool:
    key=(engine, sym)
    rec = LAST_LOSS_DIR.get(key, None)
    if not rec: return False
    if rec["dead"] > 0:
        rec["dead"] -= 1
        return True
    return False

def main():
    ensure_log_dir(); rotate_logs()
    api = Bybit()
    sub_engine  = BBEngine(api)
    main_engine = SwingEngine(api)

    print(f"[{datetime.now(timezone.utc)}] START symbols={SYMBOLS}, VIRTUAL_PAPER={VIRTUAL_PAPER}, DRY_RUN={DRY_RUN}")

    loop_i = 0

    while True:
        try:
            d4btc = api.fetch_ohlcv(LEADER,"4h",200)
            leader_px = float(d4btc["close"].iloc[-1]) if not d4btc.empty else 0.0

            main_open_cnt = len(main_engine.state)
            sub_open_cnt  = len(sub_engine.state)

            if loop_i % HEARTBEAT_EVERY == 0:
                print(f"[HB] {datetime.now(timezone.utc).strftime('%H:%M:%S')} "
                      f"leader={LEADER} {leader_px:.1f}  main={main_open_cnt} sub={sub_open_cnt}")

            for sym in SYMBOLS:
                # === MAIN ===
                if main_open_cnt < MAX_MAIN_POS and not tick_dead("main", sym):
                    sigM = main_engine.build_signal(sym)
                    if sigM:
                        # 동일봉 쿨다운: 4H -2 확정봉
                        d4 = api.fetch_ohlcv(sym, "4h", 3)
                        if d4.empty or len(d4) < 2:
                            log_skip_reason(sym, "cooldown_no_bar", "main"); 
                        else:
                            bar_ts = int(d4["timestamp"].iloc[-2].timestamp())
                            if cooldown_ok("main", sym, sigM["side"], bar_ts):
                                # 라운딩 일관화
                                tick = api.get_filters(sym)["tickSize"]
                                sigM["sl"]  = round_price(sigM["sl"], tick)
                                sigM["tp1"] = round_price(sigM["tp1"], tick)
                                sigM["tp2"] = round_price(sigM["tp2"], tick)

                                qty, f = main_engine.sizing(sym, sigM["conf_count"], sigM["size_factor"])
                                if qty > 0:
                                    side_order = "Buy" if sigM["side"]=="long" else "Sell"
                                    resp = api.place_market(sym, side_order, qty, reduce_only=False)
                                    if not api.ok(resp):
                                        log_skip_reason(sym, "order_failed_main", str(resp))
                                    else:
                                        if VIRTUAL_PAPER or DRY_RUN:
                                            main_engine.after_entry_virtual(sym, sigM["side"], sigM["entry"], sigM["tp1"], sigM["tp2"], sigM["sl"], qty, sigM.get("ee_flag", False))
                                        else:
                                            main_engine.after_entry(sym, sigM["side"], sigM["entry"], sigM["tp1"], sigM["tp2"], sigM["sl"], qty, f, sigM.get("ee_flag", False))
                                        register_exec("main", sym, sigM["side"], bar_ts)
                                        log_trade_event("ENTRY", symbol=sym, engine="main", side=sigM["side"], qty=qty,
                                                        entry=sigM["entry"], sl=sigM["sl"], tp1=sigM["tp1"], tp2=sigM["tp2"],
                                                        rr=sigM["rr"], mode="main", extra="market")
                                        main_open_cnt = len(main_engine.state)

                # === SUB ===
                if sub_open_cnt < MAX_SUB_POS and not tick_dead("sub", sym):
                    sigS = sub_engine.build_signal(sym, leader_px, main_open_cnt>0)
                    if sigS:
                        d15 = api.fetch_ohlcv(sym, "15m", 3)
                        if d15.empty or len(d15) < 2:
                            log_skip_reason(sym, "cooldown_no_bar", "sub")
                        else:
                            bar_ts = int(d15["timestamp"].iloc[-2].timestamp())
                            if cooldown_ok("sub", sym, sigS["side"], bar_ts):
                                # 라운딩 일관화
                                tick = api.get_filters(sym)["tickSize"]
                                sigS["sl"]  = round_price(sigS["sl"], tick)
                                sigS["tp1"] = round_price(sigS["tp1"], tick)
                                sigS["tp2"] = round_price(sigS["tp2"], tick)

                                qty, f = sub_engine.sizing(sym, sigS["size_factor"], sigS["conf_count"])
                                if qty > 0:
                                    side_order = "Buy" if sigS["side"]=="long" else "Sell"
                                    resp = api.place_market(sym, side_order, qty, reduce_only=False)
                                    if not api.ok(resp):
                                        log_skip_reason(sym, "order_failed_sub", str(resp))
                                    else:
                                        if VIRTUAL_PAPER or DRY_RUN:
                                            sub_engine.after_entry_virtual(sym, sigS["side"], sigS["entry"], sigS["tp1"], sigS["tp2"], sigS["sl"],
                                                                           qty, "sub", sigS.get("retest_px"), sigS)
                                        else:
                                            sub_engine.after_entry(sym, sigS["side"], sigS["entry"], sigS["tp1"], sigS["tp2"], sigS["sl"],
                                                                   qty, f, "sub", sigS.get("retest_px"), sigS)
                                        register_exec("sub", sym, sigS["side"], bar_ts)
                                        log_trade_event("ENTRY", symbol=sym, engine="sub", side=sigS["side"], qty=qty,
                                                        entry=sigS["entry"], sl=sigS["sl"], tp1=sigS["tp1"], tp2=sigS["tp2"],
                                                        rr=sigS["rr"], mode="sub", extra="market", bbw=sigS["bbw"],
                                                        came_from_squeeze=sigS["came_from_squeeze"], range_mode=sigS["range_mode"],
                                                        rebreak_ok=sigS["rebreak_ok"], ee_flag=sigS["ee_flag"])
                                        sub_open_cnt = len(sub_engine.state)

                # 관리 루프
                for symM in list(main_engine.state.keys()):
                    if VIRTUAL_PAPER or DRY_RUN: main_engine.poll_manage_virtual(symM, api)
                    else:                        main_engine.poll_manage(symM, api)

                for symS in list(sub_engine.state.keys()):
                    if VIRTUAL_PAPER or DRY_RUN: sub_engine.poll_manage_virtual(symS, api)
                    else:                        sub_engine.poll_manage(symS, api)

            loop_i += 1
            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            print("Bye"); break
        except Exception as e:
            print("[LOOP ERROR]", e, traceback.format_exc())
            time.sleep(1.0)

if __name__ == "__main__":
    main()