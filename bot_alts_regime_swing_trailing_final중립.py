# bot_swing_bb_final_pro_slim_v3.py
# 최종본(중립형, 개선판)
# - 15m 하단 추격 숏 금지 (예외: 수축→아래방향 확장)
# - 엔트리 게이트(점수 문턱) 추가
# - ATR Pullback 정의 수정(진짜 되돌림만 가점)
# - Bybit stop triggerDirection 적용
# - 리스크 기반 사이징 옵션 유지

import os
import time
import json
import math
import random
import ccxt
import pandas as pd
from datetime import datetime, timezone

# ========= 실행/계정 =========
DRYRUN = 0
API_KEY = "eb2aY5kC775hqNDYO2"
API_SECRET = "lcBdmA34s6yrtB3N1GRpu8Uo8I6NYL6NnQLz"
if not API_KEY or not API_SECRET:
    raise RuntimeError("환경변수 BYBIT_KEY/BYBIT_SECRET 필요")

SYMBOLS = ["ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
BTC_SYMBOL = "BTC/USDT:USDT"

MARGIN_MODE = "cross"
DEFAULT_LEV = 10

TAG = "SWG"
LOG_PATH = "./trade_log_final_slim_v3.jsonl"

# ========= 예산/비중(달러 고정) & 리스크 사이징 옵션 =========
PER_SYMBOL_BUDGET_USDT = 20000.0
BASE_PORTION = 0.20
MAX_PORTION = 1.00

RISK_SIZING_ENABLED = True
RISK_PCT = 0.006
RISK_MIN_USDT = 10.0

# 컨플루언스 가중(핵심/보조) + 캡 매핑
CORE_W = 2   # FIB, CHAN, VPVR
AUX_W = 1    # OB, FVG, RSI
SIGMOID_A, SIGMOID_T = 0.8, 3.0  # 점수→비중 시그모이드

# ========= 제한/가드 =========
MAX_ACTIVE_POS = 2
LOOP_SLEEP_SEC = 15
MAX_SPREAD_PCT = 0.0015  # 0.15%

# ========= 레짐/볼밴 =========
REGIME_LOOKBACK = 200
SQUEEZE_Q, EXPAND_Q = 0.25, 0.75

# EMA 필터(소프트 페널티)
USE_EMA_FILTER = True
EMA_SLOPE_N = 10
EMA_PENALTY = 0.80  # -20%

# 엔트리 TF
ENTRY_TFS = ["15m", "30m", "1h"]

# 수축→확장 판정
LTF_SQUEEZE_TH = 0.35
LTF_EXPAND_MIN = 0.48

# 마이크로 트리거(가점)
MICRO_TFS = ["15m", "30m"]
MICRO_SCORE_PER_HIT = 0.5

# ATR Pullback(가점) — 수정: 진짜 되돌림만 가점
ATR_PB_MIN = 0.5
ATR_PB_MAX = 0.8
ATR_PB_SCORE = 0.5

# ========= SL/TP/트레일 =========
TP1_RATIO = 0.50
RR_TP1, RR_TP2 = 1.0, 1.6
ATR_MIN_MULT = 0.6
ATR_MAX_MULT = 1.2

SL_ATR_K_SQZ = 0.45
SL_ATR_K_NEU = 0.30
SL_ATR_K_EXP = 0.25
SL_BUFFER_PCT = 0.004

BREAKEVEN_TICK_DEFAULT = 0.01

# TP1 편향 완화
TP1_BIAS_ATR_K = 0.03
TP1_MIN_BPS = 0.0003
TP1_POSTONLY = False
TP2_POSTONLY = True

# 트레일링(NEUTRAL 포함, 약간 느슨)
USE_TRAILING = True
TRAIL_ATR_K = 1.3
TRAIL_MID_BB_K = 0.3
TRAIL_REGIMES = {"EXPANSION", "NEUTRAL"}

# ========= BTC 앵커 =========
BTC_CORR_WIN = 60
BTC_ANCHOR_SOFT = (0.5, 0.7)
BTC_SOFT_FACTOR = 0.85

# ========= 오토-릴랙스 =========
RELAX_NS_MAX = 6
RELAX_DURATION_SEC = 30 * 60
RELAX_PENALTY_FLOOR = 0.70
PENALTY_CAP_MIN_MULT = 0.50

# ========= 오토-플립(옵션) =========
ALLOW_AUTO_FLIP = True

# ========= 추가: 진입 게이트 & LTF 하단 VETO 파라미터 =========
ENTRY_WSUM_THRESHOLD = 0.6            # LTF 위치 가중 문턱
ENTRY_GATE_ON = True                  # 게이트 사용
LTF_BAND_VETO_ATR_K = 0.25            # 하단/상단 근접 판단용 ATR 배수
VETO_ONLY_FOR_SELL_15M_LOWER = False   

# ========= 상태 =========
state = {
    s: {
        "loop": 0,
        "last_entry_ts": 0.0,
        "tp1_moved": False,
        "full_qty": 0.0,
        "entry_px": 0.0,
        "side": None,
        "peak": None,
        "regime": None,
        "risk_usdt": 0.0,
        "cooldown": 12,
        "nosig_streak": 0,
        "relax_until": 0.0,
    }
    for s in SYMBOLS
}

# ========= 유틸/지표 =========
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def now_ts():
    return time.time()

def log_event(**kv):
    kv["ts"] = now_utc()
    print(kv)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")
    except Exception:
        pass

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def sma(s, n):
    return s.rolling(n).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast)
    s = ema(close, slow)
    line = f - s
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def boll(close, n=20, k=2):
    mid = sma(close, n)
    std = close.rolling(n).std()
    up = mid + k * std
    lo = mid - k * std
    w = (up - lo) / (mid + 1e-12)
    return up, mid, lo, w

def atr(df, n=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    prev = cl.shift(1)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - prev).abs(),
        (lo - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ========= 거래소 =========
def get_ex():
    ex = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "recvWindow": 5000}
    })
    ex.load_markets()

    def _is_not_modified(e: Exception) -> bool:
        msg = str(e)
        return ("retCode" in msg) and ("not modified" in msg.lower())

    for s in SYMBOLS:
        try:
            ex.set_position_mode(False, s)
            log_event(action="SET_POSMODE_OK", symbol=s)
        except Exception as e:
            if _is_not_modified(e):
                log_event(level="INFO", action="SET_POSMODE_UNCHANGED", symbol=s)
            else:
                log_event(level="WARN", action="SET_POSMODE_FAIL", symbol=s, err=str(e))

        try:
            ex.set_margin_mode(MARGIN_MODE, s)
            log_event(action="SET_MARGINMODE_OK", symbol=s, mode=MARGIN_MODE)
        except Exception as e:
            if _is_not_modified(e):
                log_event(level="INFO", action="SET_MARGINMODE_UNCHANGED", symbol=s, mode=MARGIN_MODE)
            else:
                log_event(level="WARN", action="SET_MARGINMODE_FAIL", symbol=s, err=str(e))

        try:
            ex.set_leverage(DEFAULT_LEV, s, {"marginMode": MARGIN_MODE})
            log_event(action="SET_LEV_OK", symbol=s, lev=DEFAULT_LEV)
        except Exception as e:
            if _is_not_modified(e):
                log_event(level="INFO", action="SET_LEV_UNCHANGED", symbol=s, lev=DEFAULT_LEV)
            else:
                log_event(level="WARN", action="SET_LEV_FAIL", symbol=s, err=str(e))
        time.sleep(0.1)
    return ex

ex = get_ex()

def round_qty(symbol, q):
    try:
        return float(ex.amount_to_precision(symbol, q))
    except Exception:
        return float(f"{q:.6f}")

def round_price(symbol, p):
    try:
        return float(ex.price_to_precision(symbol, p))
    except Exception:
        return float(f"{p:.6f}")

def market_min_tick(symbol):
    try:
        prec = ex.markets[symbol]['precision']['price']
        if isinstance(prec, int) and prec > 0:
            return 10 ** (-prec)
        return BREAKEVEN_TICK_DEFAULT
    except Exception:
        return BREAKEVEN_TICK_DEFAULT

# ========= 데이터 =========
def fetch_ohlcv(symbol, tf, limit=300):
    o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def build_frame_from(df):
    d = df.copy()
    d['ema20'] = ema(d['close'], 20)
    d['ema50'] = ema(d['close'], 50)
    d['ema100'] = ema(d['close'], 100)
    d['ema200'] = ema(d['close'], 200)
    d['rsi14'] = rsi(d['close'], 14)
    d['atr14'] = atr(d, 14)
    up, mid, lo, w = boll(d['close'], 20, 2)
    d['bb_up'], d['bb_mid'], d['bb_lo'], d['bb_w'] = up, mid, lo, w
    ml, ms, mh = macd(d['close'])
    d['macd'], d['macd_sig'], d['macd_hist'] = ml, ms, mh
    d['vol_sma'] = sma(d['vol'], 20)
    return d

def build_frame(symbol, tf, limit=400):
    return build_frame_from(fetch_ohlcv(symbol, tf, limit))

# ========= 레짐/적응형 =========
def detect_regime_by_width(frame, lookback=REGIME_LOOKBACK):
    w = frame['bb_w'].dropna()
    if len(w) < lookback:
        return "NEUTRAL"
    recent = w.iloc[-lookback:]
    cur = float(w.iloc[-1])
    if cur <= recent.quantile(SQUEEZE_Q):
        return "SQUEEZE"
    if cur >= recent.quantile(EXPAND_Q):
        return "EXPANSION"
    return "NEUTRAL"

def adaptive_k(frame):
    w = frame['bb_w'].dropna()
    if len(w) < REGIME_LOOKBACK:
        return 0.6
    recent = w.iloc[-REGIME_LOOKBACK:]
    cur = float(w.iloc[-1])
    w_min = float(recent.min())
    w_max = float(recent.max())
    z = (cur - w_min) / max(1e-12, (w_max - w_min))
    return max(0.35, min(1.0, 0.35 + 0.8 * z))

def ema50_penalty_mult(h1, side):
    if not USE_EMA_FILTER:
        return 1.0
    e = h1['ema50']
    if len(e) < EMA_SLOPE_N + 2:
        return 1.0
    slope = float(e.iloc[-1] - e.iloc[-EMA_SLOPE_N]) / EMA_SLOPE_N
    ok = (slope >= 0) if side == 'buy' else (slope <= 0)
    return 1.0 if ok else EMA_PENALTY

# ========= 수축→확장/마이크로/ATR Pullback =========
def is_ltf_squeeze_to_expand(frame):
    w = frame['bb_w'].dropna()
    if len(w) < 60:
        return False
    recent = w.iloc[-60:]
    cur = float(recent.iloc[-1])
    lo = recent.quantile(LTF_SQUEEZE_TH)
    mid = recent.quantile(0.5)
    return (cur <= lo) or (cur > mid * LTF_EXPAND_MIN and recent.iloc[-2] <= lo)

def day_vwap(df_15m):
    d = df_15m.copy()
    if d.empty:
        return None
    start = pd.Timestamp(datetime.now(timezone.utc).date(), tz='UTC')
    d = d[d['ts'] >= start]
    if len(d) < 2:
        return None
    tp = (d['high'] + d['low'] + d['close']) / 3.0
    vol = d['vol'].replace(0, 1e-9)
    vwap_series = (tp * vol).cumsum() / vol.cumsum()
    last_vwap = vwap_series.iloc[-1]
    return float(last_vwap) if pd.notna(last_vwap) else None

def is_bullish_engulf(prev, now):
    return (prev['close'] < prev['open']) and (now['close'] > now['open']) and (now['close'] >= prev['open']) and (now['open'] <= prev['close'])

def is_bearish_engulf(prev, now):
    return (prev['close'] > prev['open']) and (now['close'] < now['open']) and (now['close'] <= prev['open']) and (now['open'] >= prev['close'])

def micro_triggers(symbol, side):
    hits = 0.0
    tags = []
    for tf in MICRO_TFS:
        f = build_frame(symbol, tf, 200)
        if f.empty:
            continue
        last = float(f.iloc[-1]['close'])
        prev = f.iloc[-2]
        mid = float(f['bb_mid'].iloc[-1])
        atr1 = float(f['atr14'].iloc[-1])

        if side == 'buy' and last >= mid and prev['close'] < mid and abs(last - mid) <= atr1 * 0.4:
            hits += 1; tags.append(f"{tf}_MID_RETEST_UP")
        if side == 'sell' and last <= mid and prev['close'] > mid and abs(last - mid) <= atr1 * 0.4:
            hits += 1; tags.append(f"{tf}_MID_RETEST_DN")

        if side == 'buy' and f['macd_hist'].iloc[-1] > f['macd_hist'].iloc[-2]:
            hits += 1; tags.append(f"{tf}_MACD_UP")
        if side == 'sell' and f['macd_hist'].iloc[-1] < f['macd_hist'].iloc[-2]:
            hits += 1; tags.append(f"{tf}_MACD_DN")

        if tf == "15m":
            vwap = day_vwap(f)
            if vwap:
                if side == 'buy' and last >= vwap and (last <= vwap + atr1 * 0.3):
                    hits += 0.5; tags.append("VWAP_REC_BULL")
                if side == 'sell' and last <= vwap and (last >= vwap - atr1 * 0.3):
                    hits += 0.5; tags.append("VWAP_REC_BEAR")

        p, c = f.iloc[-2], f.iloc[-1]
        if side == 'buy' and is_bullish_engulf(p, c):
            hits += 0.5; tags.append(f"{tf}_ENG_BULL")
        if side == 'sell' and is_bearish_engulf(p, c):
            hits += 0.5; tags.append(f"{tf}_ENG_BEAR")
    return hits, tags

# 수정: 진짜 되돌림만 가점 (미들 반대쪽에서만)
def atr_pullback_score(h1, side):
    try:
        atr1 = float(h1['atr14'].iloc[-1])
        last = float(h1.iloc[-1]['close'])
        mid = float(h1['bb_mid'].iloc[-1])
    except Exception:
        return 0.0, []
    score = 0.0
    tags = []
    if side == 'buy' and last <= mid:
        pb = (mid - last) / max(1e-12, atr1)
        if ATR_PB_MIN <= pb <= ATR_PB_MAX:
            score += ATR_PB_SCORE; tags.append("ATR_PB_BULL")
    if side == 'sell' and last >= mid:
        pb = (last - mid) / max(1e-12, atr1)
        if ATR_PB_MIN <= pb <= ATR_PB_MAX:
            score += ATR_PB_SCORE; tags.append("ATR_PB_BEAR")
    return score, tags

# ========= VPVR(간단) =========
def vpvr_near(price, frame, bins=30, tol_atr_k=0.6):
    atr1 = float(frame['atr14'].iloc[-1])
    lo = float(frame['low'].rolling(60).min().iloc[-2])
    hi = float(frame['high'].rolling(60).max().iloc[-2])
    if hi <= lo:
        return False
    df = frame.iloc[-200:].copy()
    step = (hi - lo) / bins
    if step <= 0:
        return False
    df['bin'] = ((df['close'] - lo) / step).clip(0, bins - 1).astype(int)
    vol_by_bin = df.groupby('bin')['vol'].sum()
    if vol_by_bin.empty:
        return False
    top_bins = vol_by_bin.sort_values(ascending=False).head(5).index.tolist()
    levels = [lo + (b + 0.5) * step for b in top_bins]
    return any(abs(price - lvl) <= atr1 * tol_atr_k for lvl in levels)

# ========= 컨플루언스 =========
def confluence_score(price, frame, side):
    score = 0
    tags = []
    atr1 = float(frame['atr14'].iloc[-1])

    lo = float(frame['low'].rolling(30).min().iloc[-2])
    hi = float(frame['high'].rolling(30).max().iloc[-2])
    if hi < lo:
        lo, hi = hi, lo
    for lvl in [0.382, 0.5, 0.618]:
        lpx = hi - (hi - lo) * lvl
        if abs(price - lpx) <= atr1 * 0.6:
            score += CORE_W; tags.append(f"FIB_{lvl}"); break

    lo2 = float(frame['low'].rolling(14).min().iloc[-2])
    hi2 = float(frame['high'].rolling(14).max().iloc[-2])
    mid2 = (lo2 + hi2) / 2.0
    if side == 'long' and abs(price - lo2) <= atr1 * 0.6:
        score += CORE_W; tags.append("CHAN_LOW")
    elif side == 'short' and abs(price - hi2) <= atr1 * 0.6:
        score += CORE_W; tags.append("CHAN_UP")
    elif abs(price - mid2) <= atr1 * 0.6:
        score += CORE_W; tags.append("CHAN_MID")

    vp = vpvr_near(price, frame, tol_atr_k=0.6)
    if vp:
        score += CORE_W; tags.append("VPVR")

    aux_extra_allowed = 1 if vp else 2
    aux_taken = 0

    try:
        last20 = frame.iloc[-20:]
        bodies = (last20['close'] - last20['open']).abs()
        k = int(bodies.idxmax())
        row = frame.loc[k]
        bull = row['close'] > row['open']
        ob_lo = float(min(row['open'], row['close']))
        ob_hi = float(max(row['open'], row['close']))
        box_ok = (ob_lo - 0.2 * atr1) <= price <= (ob_hi + 0.2 * atr1)
        if aux_taken < aux_extra_allowed:
            if side == 'long' and bull and box_ok:
                score += AUX_W; tags.append("OB_BULL"); aux_taken += 1
            if side == 'short' and (not bull) and box_ok:
                score += AUX_W; tags.append("OB_BEAR"); aux_taken += 1
    except Exception:
        pass

    try:
        a, b, c = frame.iloc[-3], frame.iloc[-2], frame.iloc[-1]
        if aux_taken < aux_extra_allowed:
            if side == 'long' and b['low'] > a['high']:
                lo_g, hi_g = float(a['high']), float(b['low'])
                if (lo_g - 0.2 * atr1) <= price <= (hi_g + 0.2 * atr1):
                    score += AUX_W; tags.append("FVG_BULL"); aux_taken += 1
            if side == 'short' and b['high'] < a['low']:
                lo_g, hi_g = float(b['high']), float(a['low'])
                if (lo_g - 0.2 * atr1) <= price <= (hi_g + 0.2 * atr1):
                    score += AUX_W; tags.append("FVG_BEAR"); aux_taken += 1
    except Exception:
        pass

    rsi_v = float(frame['rsi14'].iloc[-1])
    if side == 'long' and rsi_v < 35:
        score += AUX_W; tags.append("RSI_OS")
    if side == 'short' and rsi_v > 65:
        score += AUX_W; tags.append("RSI_OB")

    return score, tags

def portion_from_score(score):
    x = 1.0 / (1.0 + math.exp(-SIGMOID_A * (score - SIGMOID_T)))
    return max(BASE_PORTION, min(MAX_PORTION, BASE_PORTION + (MAX_PORTION - BASE_PORTION) * x))

# ========= 포지션/계좌 =========
def get_free_usdt():
    try:
        bal = ex.fetch_balance()
        free = bal.get('free', {}).get('USDT')
        if free is None:
            free = bal.get('USDT', {}).get('free')
        return float(free or 0.0)
    except Exception:
        return 0.0

def get_equity_usdt():
    try:
        bal = ex.fetch_balance()
        total = bal.get('total', {}).get('USDT')
        if total is None:
            free = bal.get('free', {}).get('USDT') or 0.0
            used = bal.get('used', {}).get('USDT') or 0.0
            total = float(free) + float(used)
        return float(total or 0.0)
    except Exception:
        return 0.0

def fetch_positions(symbol):
    try:
        return ex.fetch_positions([symbol])
    except Exception:
        try:
            return ex.fetch_positions_risk([symbol])
        except Exception:
            return []

def in_position(symbol):
    for p in fetch_positions(symbol):
        if p.get('symbol') == symbol and float(p.get('contracts') or p.get('info', {}).get('size') or 0) > 0:
            return True
    return False

def get_position_side_and_size(symbol):
    for p in fetch_positions(symbol):
        if p.get('symbol') == symbol and float(p.get('contracts') or p.get('info', {}).get('size') or 0) > 0:
            side = (p.get('side') or '').lower()
            sz = float(p.get('contracts') or p.get('info', {}).get('size') or 0)
            ep = float(p.get('entryPrice') or 0)
            return side, sz, ep
    return None, 0.0, 0.0

def current_active_symbols():
    return [s for s in SYMBOLS if in_position(s)]

def active_positions_count():
    return len(current_active_symbols())

# ========= 주문/체결 & SL 관리 =========
def _cid(side_code, purpose):
    ms = int(time.time() * 1000)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    base36, x = "", ms
    while x:
        base36 = digits[x % 36] + base36
        x //= 36
    if not base36:
        base36 = "0"
    r2 = "".join(random.choice(digits) for _ in range(2))
    return f"{TAG}-{base36}-{r2}-{side_code}{purpose}"[:45]

def create_order(symbol, side, type_, qty, price=None, reduceOnly=False, tag="E", extra=None):
    params = {
        "reduceOnly": reduceOnly,
        "clientOrderId": _cid("L" if side == 'buy' else "S", tag[:2]),
        "positionIdx": 0
    }
    if extra:
        params.update(extra)
    if DRYRUN:
        log_event(DRYRUN=True, action="CREATE_ORDER", symbol=symbol, side=side, type=type_,
                  qty=qty, price=price, params=params)
        return {"price": price}
    return ex.create_order(
        symbol, type_, side, round_qty(symbol, qty),
        None if type_ == 'market' else round_price(symbol, price), params
    )

def fetch_open_orders(symbol):
    try:
        return ex.fetch_open_orders(symbol)
    except Exception:
        return []

def cancel_tagged_orders(symbol, tag_contains):
    for o in fetch_open_orders(symbol):
        cid = o.get('clientOrderId') or o.get('info', {}).get('orderLinkId') or ''
        if tag_contains in str(cid):
            try:
                ex.cancel_order(o['id'], symbol)
            except Exception as e:
                log_event(level="WARN", action="CANCEL_FAIL", symbol=symbol, err=str(e))

def _trigger_direction(order_side, trigger, last=None):
    # reduce-only stop: sell-stop은 하락(Descending), buy-stop은 상승(Ascending)에서 트리거
    return "descending" if order_side == "sell" else "ascending"

def create_stop_fallback(symbol, side, qty, trigger, tag):
    """Bybit stop 주문 fallback: 'stop' → 'stopMarket' → 'market'+stopLossPrice"""
    params_common = {
        "reduceOnly": True,
        "clientOrderId": _cid("L" if side == 'buy' else "S", tag[:2]),
        "positionIdx": 0
    }
    tdir = _trigger_direction(side, trigger)
    # 1) unified 'stop'
    try:
        return ex.create_order(
            symbol, 'stop', side, round_qty(symbol, qty), None,
            {
                **params_common,
                "triggerPrice": round_price(symbol, trigger),
                "triggerDirection": tdir,
                "slTriggerBy": "MarkPrice",
                "closeOnTrigger": True
            }
        )
    except Exception as e1:
        log_event(level="WARN", action="STOP_FALLBACK1", symbol=symbol, err=str(e1))
    # 2) stopMarket
    try:
        return ex.create_order(
            symbol, 'stopMarket', side, round_qty(symbol, qty), None,
            {
                **params_common,
                "triggerPrice": round_price(symbol, trigger),
                "triggerDirection": tdir,
                "slTriggerBy": "MarkPrice",
                "closeOnTrigger": True
            }
        )
    except Exception as e2:
        log_event(level="WARN", action="STOP_FALLBACK2", symbol=symbol, err=str(e2))
    # 3) best-effort
    try:
        return ex.create_order(
            symbol, 'market', side, round_qty(symbol, qty), None,
            {
                **params_common,
                "stopLossPrice": round_price(symbol, trigger)
            }
        )
    except Exception as e3:
        log_event(level="ERROR", action="STOP_FALLBACK_FAIL", symbol=symbol, err=str(e3))
        raise

def spread_ok(symbol):
    try:
        ob = ex.fetch_order_book(symbol, 5)
        bid = float(ob['bids'][0][0])
        ask = float(ob['asks'][0][0])
        spr = (ask - bid) / max(1e-12, (ask + bid) / 2)
        return spr <= MAX_SPREAD_PCT
    except Exception:
        return True

# ========= 사이징 =========
def size_from_budget(symbol, last, budget_usdt):
    if last <= 0 or budget_usdt <= 0:
        return 0.0
    qty_target = budget_usdt / last
    free = get_free_usdt()
    notional_cap = free * DEFAULT_LEV
    qty_cap = max(0.0, notional_cap / last)
    return round_qty(symbol, min(qty_target, qty_cap))

def size_from_risk(symbol, last, stop_price, risk_usdt):
    if last <= 0 or risk_usdt <= 0 or stop_price <= 0:
        return 0.0
    stop_dist = abs(last - stop_price)
    if stop_dist <= 0:
        return 0.0
    qty_target = risk_usdt / stop_dist
    free = get_free_usdt()
    notional_cap = free * DEFAULT_LEV
    qty_cap = max(0.0, notional_cap / last)
    return round_qty(symbol, max(0.0, min(qty_target, qty_cap)))

# ========= SL/TP/트레일 =========
def regime_sl_k(regime):
    if regime == "SQUEEZE":
        return SL_ATR_K_SQZ
    if regime == "EXPANSION":
        return SL_ATR_K_EXP
    return SL_ATR_K_NEU

def calc_sl_tp(symbol, side, h1, last, regime):
    atr1 = float(h1['atr14'].iloc[-1])
    lo10 = float(h1['low'].rolling(10).min().iloc[-2])
    hi10 = float(h1['high'].rolling(10).max().iloc[-2])
    slk = regime_sl_k(regime)

    if side == 'buy':
        swing = lo10 * (1 - SL_BUFFER_PCT)
        risk_raw = max(1e-8, last - swing)
        risk = max(ATR_MIN_MULT * atr1, min(risk_raw, ATR_MAX_MULT * atr1))
        sl = last - (risk + atr1 * slk)
        tp1 = last + risk * RR_TP1
        tp2 = last + risk * RR_TP2
    else:
        swing = hi10 * (1 + SL_BUFFER_PCT)
        risk_raw = max(1e-8, swing - last)
        risk = max(ATR_MIN_MULT * atr1, min(risk_raw, ATR_MAX_MULT * atr1))
        sl = last + (risk + atr1 * slk)
        tp1 = last - risk * RR_TP1
        tp2 = last - risk * RR_TP2

    return round_price(symbol, sl), round_price(symbol, tp1), round_price(symbol, tp2), atr1

def place_bracket(symbol, side, qty, sl, tp1, tp2, last, atr1):
    bias_abs = min(atr1 * TP1_BIAS_ATR_K, last * TP1_MIN_BPS)
    tp1_adj = round_price(symbol, tp1 - bias_abs) if side == 'buy' else round_price(symbol, tp1 + bias_abs)

    tag = f"{TAG}_{side.upper()}"
    entry = create_order(symbol, side, 'market', qty, reduceOnly=False, tag=f"{tag}_E")

    red = 'sell' if side == 'buy' else 'buy'
    q1 = round_qty(symbol, qty * TP1_RATIO)
    q2 = round_qty(symbol, max(0.0, qty - q1))

    create_order(symbol, red, 'limit', q1, price=tp1_adj, reduceOnly=True, tag=f"{tag}_T1",
                 extra={"postOnly": bool(TP1_POSTONLY)})
    create_order(symbol, red, 'limit', q2, price=tp2, reduceOnly=True, tag=f"{tag}_T2",
                 extra={"postOnly": bool(TP2_POSTONLY)})

    cancel_tagged_orders(symbol, f"{TAG}_{side.upper()}_SL")
    create_stop_fallback(symbol, red, qty, sl, tag=f"{tag}_SL")
    return entry

def replace_stop(symbol, side, qty, new_sl):
    red = 'sell' if side == 'buy' else 'buy'
    cancel_tagged_orders(symbol, f"{TAG}_{side.upper()}_SL")
    create_stop_fallback(symbol, red, qty, new_sl, tag=f"{TAG}_{side.upper()}_SL")
    log_event(action="TRAIL_SL", symbol=symbol, side=side, new_sl=round_price(symbol, new_sl))

def update_trailing(symbol, h1):
    if not USE_TRAILING:
        return
    st = state[symbol]
    side = st["side"]
    regime = st.get("regime") or "NEUTRAL"
    if not side or regime not in TRAIL_REGIMES:
        return

    last = float(h1.iloc[-1]['close'])
    atr1 = float(h1['atr14'].iloc[-1])
    bb_mid = float(h1['bb_mid'].iloc[-1])

    if st["peak"] is None:
        st["peak"] = last
    if side == 'buy':
        st["peak"] = max(st["peak"], last)
    else:
        st["peak"] = min(st["peak"], last)

    ep = st["entry_px"]
    qty = st["full_qty"]
    if qty <= 0 or ep <= 0:
        return

    tick = market_min_tick(symbol)
    if side == 'buy':
        new_sl = max(ep + tick, st["peak"] - atr1 * TRAIL_ATR_K, bb_mid - atr1 * TRAIL_MID_BB_K)
    else:
        new_sl = min(ep - tick, st["peak"] + atr1 * TRAIL_ATR_K, bb_mid + atr1 * TRAIL_MID_BB_K)

    replace_stop(symbol, side, qty, new_sl)

# ========= BTC 앵커/상관성 =========
def get_btc_direction():
    try:
        bf = build_frame(BTC_SYMBOL, "4h", 400)
        if bf.empty:
            return None
        last = float(bf.iloc[-1]['close'])
        mid = float(bf['bb_mid'].iloc[-1])
        ema50 = float(bf['ema50'].iloc[-1])
        if last > mid and last > ema50:
            return 'buy'
        if last < mid and last < ema50:
            return 'sell'
        return None
    except Exception:
        return None

def btc_corr(symbol):
    try:
        a = build_frame(symbol, "1h", 400)
        b = build_frame(BTC_SYMBOL, "1h", 400)
        if a.empty or b.empty:
            return 0.0
        x = a['close'].iloc[-BTC_CORR_WIN:].reset_index(drop=True).pct_change()
        y = b['close'].iloc[-BTC_CORR_WIN:].reset_index(drop=True).pct_change()
        return float(x.corr(y) or 0.0)
    except Exception:
        return 0.0

# ========= 의사결정 =========
def ltf_entry_weights(symbol, side, regime):
    wsum = 0.0
    tags = []
    for tf in ENTRY_TFS:
        f = build_frame(symbol, tf, 200)
        if f.empty or pd.isna(f['atr14'].iloc[-1]):
            continue
        last = float(f.iloc[-1]['close'])
        atr = float(f['atr14'].iloc[-1])
        up = float(f['bb_up'].iloc[-1])
        mid = float(f['bb_mid'].iloc[-1])
        lo = float(f['bb_lo'].iloc[-1])
        k = adaptive_k(f)

        if side == 'sell':
            touch = (last >= up - atr * k)
            midok = (last >= mid)
        else:
            touch = (last <= lo + atr * k)
            midok = (last <= mid)

        if regime == "SQUEEZE":
            if touch: wsum += 0.5; tags.append(f"{tf}_TOUCH+0.5")
            elif midok: wsum += 0.1; tags.append(f"{tf}_MID+0.1")
        else:
            if touch: wsum += 0.5; tags.append(f"{tf}_TOUCH+0.5")
            elif midok: wsum += 0.3; tags.append(f"{tf}_MID+0.3")
    return (wsum if wsum >= ENTRY_WSUM_THRESHOLD else 0.0), tags

def squeeze_bias(symbol, side):
    bonus = 0.0
    tags = []
    for tf in ENTRY_TFS:
        f = build_frame(symbol, tf, 200)
        if f.empty or not is_ltf_squeeze_to_expand(f):
            continue
        last = float(f.iloc[-1]['close'])
        up = float(f['bb_up'].iloc[-1])
        lo = float(f['bb_lo'].iloc[-1])
        if side == 'buy':
            if last >= up: bonus -= 0.0  # (완화) 즉시 +1 제거 가능 지점
            elif last <= lo: bonus -= 1.0; tags.append(f"{tf}_SQZ-1")
        else:
            if last <= lo: bonus += 1.0; tags.append(f"{tf}_SQZ+1")
            elif last >= up: bonus -= 1.0; tags.append(f"{tf}_SQZ-1")
    return bonus, tags

def decide_side(symbol, h4, h1, last, btc_dir, rho):
    regime = detect_regime_by_width(h4)
    bb_up = float(h4['bb_up'].iloc[-1])
    bb_lo = float(h4['bb_lo'].iloc[-1])
    bb_mid = float(h4['bb_mid'].iloc[-1])
    atr1 = float(h1['atr14'].iloc[-1])
    k_reg = adaptive_k(h1)

    if regime == "EXPANSION":
        long_ok = (last <= bb_lo + atr1 * k_reg)
        short_ok = (last >= bb_up - atr1 * k_reg)
    else:
        long_ok = (last > bb_mid)
        short_ok = (last < bb_mid)

    hard_block = False
    if btc_dir and symbol != BTC_SYMBOL and rho >= BTC_ANCHOR_SOFT[1]:
        if btc_dir == 'buy': short_ok = False
        if btc_dir == 'sell': long_ok = False
        hard_block = True

    choice = None
    if long_ok and not short_ok:
        choice = 'buy'
    elif short_ok and not long_ok:
        choice = 'sell'
    elif long_ok and short_ok:
        choice = 'sell' if abs(last - bb_up) < abs(last - bb_lo) else 'buy'

    return regime, choice, hard_block

def cooldown_by_regime(regime):
    if regime == "SQUEEZE": return 20
    if regime == "EXPANSION": return 8
    return 12

def allowed_by_cooldown(symbol):
    return (now_ts() - state[symbol]["last_entry_ts"]) >= state[symbol].get("cooldown", 12)

# === NEW: 15m 하단 숏 VETO (예외: 수축→아래방향 확장) ===
def ltf_band_veto(symbol, side):
    try:
        f = build_frame(symbol, "15m", 160)
        if f.empty or pd.isna(f['atr14'].iloc[-1]):
            return False
        last = float(f.iloc[-1]['close'])
        prev_close = float(f.iloc[-2]['close'])
        atr1 = float(f['atr14'].iloc[-1])
        up = float(f['bb_up'].iloc[-1])
        lo = float(f['bb_lo'].iloc[-1])

        near_lower = last <= lo + atr1 * LTF_BAND_VETO_ATR_K
        near_upper = last >= up - atr1 * LTF_BAND_VETO_ATR_K

        if VETO_ONLY_FOR_SELL_15M_LOWER:
            if side == 'sell' and near_lower:
                # 예외 허용: 수축→확장 상태 + 하단 이탈 + 현재 캔들 하락
                sqz_expand = is_ltf_squeeze_to_expand(f)
                downside = (last <= lo) and (last < prev_close)
                if sqz_expand and downside:
                    return False  # 허용
                return True       # VETO
            return False
        else:
            # (대칭 적용하고 싶을 때 사용)
            if side == 'sell' and near_lower:
                sqz_expand = is_ltf_squeeze_to_expand(f)
                downside = (last <= lo) and (last < prev_close)
                if sqz_expand and downside: return False
                return True
            if side == 'buy' and near_upper:
                sqz_expand = is_ltf_squeeze_to_expand(f)
                upside = (last >= up) and (last > prev_close)
                if sqz_expand and upside: return False
                return True
            return False
    except Exception:
        return False

# ========= 엔트리 게이트 =========
def pass_entry_gate(entry_w, micro_hits, conf_score):
    return (entry_w > 0) or (micro_hits >= 1) or (conf_score >= 2)

# ========= 메인 루프 =========
def run_once(symbol):
    h4 = build_frame(symbol, "4h", 400)
    h1 = build_frame(symbol, "1h", 400)
    if h4.empty or h1.empty:
        log_event(level="WARN", symbol=symbol, msg="NO_DATA")
        return

    last = float(h1.iloc[-1]['close'])
    st = state[symbol]
    st["loop"] += 1

    if st["loop"] % 6 == 0:
        log_event(hb=True, symbol=symbol, px=last, regime=detect_regime_by_width(h4), active=current_active_symbols())

    if in_position(symbol):
        update_trailing(symbol, h1)
        st["nosig_streak"] = 0
        return

    if active_positions_count() >= MAX_ACTIVE_POS:
        return
    if not spread_ok(symbol):
        log_event(action="SPREAD_GUARD", symbol=symbol)
        st["nosig_streak"] += 1
        return

    btc_dir = get_btc_direction()
    rho = btc_corr(symbol)
    regime, choice, hard_block = decide_side(symbol, h4, h1, last, btc_dir, rho)
    st["cooldown"] = cooldown_by_regime(regime)
    if not allowed_by_cooldown(symbol):
        return
    if not choice:
        log_event(action="NO_SIGNAL", symbol=symbol, reason="no_choice", btc_dir=btc_dir, rho=round(rho, 3), regime=regime)
        st["nosig_streak"] += 1
        return

    # 15m 하단 숏 VETO (요청사항)
    if ltf_band_veto(symbol, choice):
        log_event(action="VETO_LTF_BAND_EXTREME", symbol=symbol, side=choice)
        st["nosig_streak"] += 1
        return

    if not auto_flip_if_needed(symbol, choice):
        log_event(action="BLOCK_SAME_DIR", symbol=symbol)
        st["nosig_streak"] += 1
        return

    # ==== 소프트 가중/가점 ====
    soft_mult = 1.0
    soft_tags = []

    if btc_dir and symbol != BTC_SYMBOL and (BTC_ANCHOR_SOFT[0] <= rho < BTC_ANCHOR_SOFT[1]):
        soft_mult *= BTC_SOFT_FACTOR
        soft_tags.append(f"BTC_SOFTx{BTC_SOFT_FACTOR}")

    em_mult = ema50_penalty_mult(h1, choice)
    if em_mult < 1.0:
        soft_mult *= em_mult
        soft_tags.append(f"EMA50slopex{em_mult:.2f}")

    entry_w, entry_tags = ltf_entry_weights(symbol, choice, regime)
    sqz_bonus, sqz_tags = squeeze_bias(symbol, choice)
    micro_hits, micro_tags = micro_triggers(symbol, choice)
    pb_score, pb_tags = atr_pullback_score(h1, choice)
    conf_score, conf_tags = confluence_score(last, h1, 'long' if choice == 'buy' else 'short')

    total_score = conf_score + entry_w + sqz_bonus + micro_hits * MICRO_SCORE_PER_HIT + pb_score
    portion = portion_from_score(total_score)

    # 엔트리 게이트 (문턱 미충족시 진입 안함)
    if ENTRY_GATE_ON and not pass_entry_gate(entry_w, micro_hits, conf_score):
        log_event(action="NO_SIGNAL", symbol=symbol, reason="ENTRY_GATE")
        st["nosig_streak"] += 1
        return

    relaxing = now_ts() < st.get("relax_until", 0.0)
    if relaxing:
        soft_mult = max(soft_mult, RELAX_PENALTY_FLOOR)
        soft_tags.append("RELAX")

    soft_mult = max(soft_mult, PENALTY_CAP_MIN_MULT)
    final_portion = max(BASE_PORTION, min(MAX_PORTION, portion * soft_mult))

    # ==== 사이징/브래킷 ====
    sl, tp1, tp2, atr1 = calc_sl_tp(symbol, choice, h1, last, regime)

    if RISK_SIZING_ENABLED:
        equity = get_equity_usdt()
        base_risk = max(RISK_MIN_USDT, equity * RISK_PCT)
        risk_usdt = base_risk * final_portion
        qty = size_from_risk(symbol, last, sl, risk_usdt)
        st["risk_usdt"] = risk_usdt
    else:
        budget_usdt = PER_SYMBOL_BUDGET_USDT * final_portion
        qty = size_from_budget(symbol, last, budget_usdt)
        st["risk_usdt"] = budget_usdt

    if qty <= 0:
        log_event(action="SIZE_ZERO", symbol=symbol,
                  budget=st["risk_usdt"], sizing="risk" if RISK_SIZING_ENABLED else "budget")
        st["nosig_streak"] += 1
        if st["nosig_streak"] >= RELAX_NS_MAX:
            st["relax_until"] = now_ts() + RELAX_DURATION_SEC
            st["nosig_streak"] = 0
            log_event(action="RELAX_ON", symbol=symbol, until=int(st["relax_until"]))
        return

    if DRYRUN:
        log_event(
            DRYRUN=True, action="ENTER_PLAN", symbol=symbol, side=choice, qty=qty,
            portion=round(final_portion, 3), sl=sl, tp1=tp1, tp2=tp2, score=round(total_score, 2),
            reason=f"REG:{regime}; BTC:{btc_dir}/ρ={rho:.2f}{'(HARD)' if hard_block else ''}; "
                   f"soft={'|'.join(soft_tags) if soft_tags else 'none'}; "
                   f"entryW:{entry_w:+.2f}; sqz:{sqz_bonus:+.2f}; micro:{micro_hits:.1f}; pb:{pb_score:.1f}; conf:{conf_score:.2f}",
            tags=list(set(conf_tags + entry_tags + micro_tags + sqz_tags + pb_tags)),
            sizing="risk" if RISK_SIZING_ENABLED else "budget",
            risk_usdt=round(st["risk_usdt"], 2)
        )
        st["last_entry_ts"] = now_ts()
        st["nosig_streak"] = 0
        return

    entry = place_bracket(symbol, choice, qty, sl, tp1, tp2, last, atr1)
    st["last_entry_ts"] = now_ts()
    st["tp1_moved"] = False
    st["full_qty"] = qty
    st["entry_px"] = float(entry.get('price') or last)
    st["side"] = choice
    st["peak"] = last
    st["regime"] = regime
    st["nosig_streak"] = 0

    log_event(
        action="ENTER", symbol=symbol, side=choice, qty=qty, portion=round(final_portion, 3),
        sl=sl, tp1=tp1, tp2=tp2, score=round(total_score, 2),
        reason=f"REG:{regime}; BTC:{btc_dir}/ρ={rho:.2f}{'(HARD)' if hard_block else ''}; "
               f"soft={'|'.join(soft_tags) if soft_tags else 'none'}; "
               f"entryW:{entry_w:+.2f}; sqz:{sqz_bonus:+.2f}; micro:{micro_hits:.1f}; pb:{pb_score:.1f}; conf:{conf_score:.2f}",
        tags=list(set(conf_tags + entry_tags + micro_tags + sqz_tags + pb_tags)),
        sizing="risk" if RISK_SIZING_ENABLED else "budget",
        risk_usdt=round(st["risk_usdt"], 2)
    )

def auto_flip_if_needed(symbol, desired_side):
    if not ALLOW_AUTO_FLIP:
        return True
    cur_side, sz, _ = get_position_side_and_size(symbol)
    if sz <= 0:
        return True
    if (cur_side == 'long' and desired_side == 'buy') or (cur_side == 'short' and desired_side == 'sell'):
        return False
    red = 'sell' if cur_side == 'long' else 'buy'
    try:
        create_order(symbol, red, 'market', sz, reduceOnly=True, tag="CLS")
        log_event(action="AUTO_FLIP_CLOSE", symbol=symbol, closed_side=cur_side, size=sz)
        time.sleep(0.2)
        return True
    except Exception as e:
        log_event(level="ERROR", action="AUTO_FLIP_FAIL", symbol=symbol, err=str(e))
        return False

def main_loop():
    log_event(
        action="START", symbols=SYMBOLS,
        budget_per_symbol=PER_SYMBOL_BUDGET_USDT, btc_anchor_soft=BTC_ANCHOR_SOFT,
        risk_sizing=RISK_SIZING_ENABLED, risk_pct=RISK_PCT
    )
    backoff = 3
    while True:
        try:
            for sym in SYMBOLS:
                run_once(sym)
                time.sleep(0.4)
            time.sleep(LOOP_SLEEP_SEC)
            backoff = 3
        except KeyboardInterrupt:
            log_event(action="STOP_BY_USER")
            break
        except Exception as e:
            log_event(level="ERROR", action="MAIN_ERR", err=str(e))
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

if __name__ == "__main__":
    main_loop()
