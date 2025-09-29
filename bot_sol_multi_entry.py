# bot_sol_multi_entry.py
# Bybit USDT Perp (SOL/USDT:USDT) — One-way(단방향) 전용
# - ATR 기반 SL/TP + 분할익절, TP1 체결 시 SL 본절(+틱) 이동
# - 피보 되돌림/채널 + 간단 OB/FVG는 "가점(컨플루언스)"로만 사용(없어도 동작)
# - positionIdx 사용하지 않음 → 10001 오류 회피
# - recvWindow 적용, reduceOnly 정리 등 안전장치

import time, json, math, ccxt
import pandas as pd
from datetime import datetime, timezone

# =============== 사용자 설정 ===============
DRYRUN         = 0                      # 0=실거래
API_KEY        = "eb2aY5kC775hqNDYO2"                     
API_SECRET     = "lcBdmA34s6yrtB3N1GRpu8Uo8I6NYL6NnQLz"                    
SYMBOL         = "SOL/USDT:USDT"

MAX_LEV        = 10
MARGIN_MODE    = "cross"
ORDER_QTY_COIN = 35

# 리스크/목표(완화)
SL_BUFFER_PCT  = 0.003
ATR_MIN_MULT   = 0.6
ATR_MAX_MULT   = 1.2
RR_TP1, RR_TP2 = 1.0, 1.6
TP1_RATIO      = 0.50
BREAKEVEN_TICK = 0.01

# 컨플루언스(가점)
FIB_LEVELS     = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_TOL_ATR    = 0.6
CHAN_TOL_ATR   = 0.6
CONF_REQ       = 2

# 루프
LOOP_SLEEP_SEC = 10
TF_4H, TF_1H, TF_15M = "4h", "1h", "15m"
LOG_PATH       = "./trade_log_sol.jsonl"

state = {"tp1_moved": False, "entry_px": None, "full_qty": 0.0}

# =============== 유틸 ===============
def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
def log_event(**kv):
    kv["ts"] = now_utc()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")
    except Exception: pass
    print(kv)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))
def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast); s = ema(close, slow)
    line = f - s; sig = ema(line, signal); hist = line - sig
    return line, sig, hist
def boll(close, n=20, k=2):
    mid = sma(close, n); std = close.rolling(n).std()
    up = mid + k*std; lo = mid - k*std
    width = (up - lo) / (mid + 1e-12)
    return up, mid, lo, width
def atr(df, n=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    prev = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =============== 거래소 연결 ===============
def get_ex():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API_KEY/API_SECRET 입력 필요")
    ex = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "recvWindow": 5000}
    })
    ex.load_markets()
    # 단방향(One-way) 시도 — 실패해도 경고만
    try: ex.set_position_mode(False, SYMBOL)
    except Exception as e: log_event(level="WARN", msg="set_position_mode(False)", err=str(e))
    try: ex.set_margin_mode(MARGIN_MODE, SYMBOL)
    except Exception as e: log_event(level="WARN", msg="set_margin_mode", err=str(e))
    try: ex.set_leverage(int(MAX_LEV), SYMBOL, {"marginMode": MARGIN_MODE})
    except Exception as e: log_event(level="WARN", msg="set_leverage", err=str(e))
    return ex
ex = get_ex()

# =============== 라운딩 ===============
def round_qty(q):
    try: return float(ex.amount_to_precision(SYMBOL, q))
    except Exception: return float(f"{q:.6f}")
def round_price(p):
    try: return float(ex.price_to_precision(SYMBOL, p))
    except Exception: return float(f"{p:.6f}")

# =============== 데이터 ===============
def fetch_ohlcv(tf, limit=300):
    o = ex.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    # 인덱스는 정수(0..N-1) 유지 — 피보 채널에서 그대로 좌표로 사용
    return df

def build_frame(tf, limit):
    df = fetch_ohlcv(tf, limit)
    df['ema20']  = ema(df['close'], 20)
    df['ema50']  = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14']  = rsi(df['close'], 14)
    df['atr14']  = atr(df, 14)
    up,mid,lo,w  = boll(df['close'], 20, 2)
    df['bb_up'], df['bb_mid'], df['bb_lo'], df['bb_w'] = up, mid, lo, w
    ml, ms, mh = macd(df['close'])
    df['macd'], df['macd_sig'], df['macd_hist'] = ml, ms, mh
    return df

# =============== 스윙 포인트/피보 ===============
def swing_points(series, left=3, right=3, kind='low'):
    s = series
    idxs = []
    for i in range(left, len(s)-right):
        win = s.iloc[i-left:i+right+1]
        if kind=='low'  and s.iloc[i] == win.min(): idxs.append(i)  # 정수 인덱스 반환
        if kind=='high' and s.iloc[i] == win.max(): idxs.append(i)
    return idxs

def fib_levels(low, high):
    diff = high - low
    return {lvl: high - diff*lvl for lvl in FIB_LEVELS}

def build_fib_retracement(df):
    lows  = swing_points(df['low'],  left=3, right=3, kind='low')
    highs = swing_points(df['high'], left=3, right=3, kind='high')
    if len(lows) < 1 or len(highs) < 1: return None
    lo_idx = lows[-1]; hi_idx = highs[-1]
    lo = float(df.loc[lo_idx, 'low']); hi = float(df.loc[hi_idx, 'high'])
    if hi < lo: lo, hi = hi, lo
    return {"low": lo, "high": hi, "levels": fib_levels(lo, hi)}

def build_fib_channel(df):
    # ⚠️ 인덱스는 정수 좌표를 그대로 사용 — reset_index로 'ts' 중복 생성하지 않음
    lows  = swing_points(df['low'],  left=3, right=3, kind='low')
    highs = swing_points(df['high'], left=3, right=3, kind='high')
    if len(lows) < 2 or len(highs) < 1: return None
    xa, xb, xh = lows[-2], lows[-1], highs[-1]
    ya, yb, yh = float(df.loc[xa,'low']), float(df.loc[xb,'low']), float(df.loc[xh,'high'])
    if xb == xa: return None
    m = (yb - ya) / (xb - xa)
    xN = len(df) - 1
    low_now = m*(xN - xb) + yb
    up_now  = m*(xN - xh) + yh
    mid_now = (low_now + up_now) / 2.0
    return {"lower_now": float(low_now), "upper_now": float(up_now), "mid_now": float(mid_now)}

# =============== 간단 OB/FVG (가점용) ===============
def detect_ob_side(df):
    try:
        last20 = df.iloc[-20:]
        bodies = (last20['close'] - last20['open']).abs()
        k = int(bodies.idxmax())
        row = df.loc[k]
        bull = row['close'] > row['open']
        ob_low  = float(min(row['open'], row['close']))
        ob_high = float(max(row['open'], row['close']))
        return {"side": "bull" if bull else "bear", "low": ob_low, "high": ob_high}
    except Exception:
        return None

def detect_fvg(df):
    try:
        a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if b['low'] > a['high']:   # 상승 FVG
            return {"type":"bull", "gap_low": float(a['high']), "gap_high": float(b['low'])}
        if b['high'] < a['low']:   # 하락 FVG
            return {"type":"bear", "gap_low": float(b['high']), "gap_high": float(a['low'])}
        return None
    except Exception:
        return None

# =============== 컨플루언스 점수 ===============
def confluence_score(px, frame, fib_ret, fib_chan, ob, fvg, side):
    score = 0
    atr1 = float(frame['atr14'].iloc[-1])

    if fib_ret:
        pref = [0.5, 0.618, 0.786] if side=='long' else [0.382, 0.5, 0.236]
        for lvl, lvl_px in fib_ret['levels'].items():
            if abs(px - lvl_px) <= atr1*FIB_TOL_ATR:
                score += 2 if lvl in pref else 1

    if fib_chan:
        if side=='long'  and abs(px - fib_chan['lower_now']) <= atr1*CHAN_TOL_ATR: score += 2
        if side=='short' and abs(px - fib_chan['upper_now']) <= atr1*CHAN_TOL_ATR: score += 2
        if abs(px - fib_chan['mid_now']) <= atr1*CHAN_TOL_ATR: score += 1

    if ob:
        if side=='long'  and ob['side']=='bull' and ob['low']-atr1*0.2 <= px <= ob['high']+atr1*0.2: score += 1
        if side=='short' and ob['side']=='bear' and ob['low']-atr1*0.2 <= px <= ob['high']+atr1*0.2: score += 1

    if fvg:
        if side=='long'  and fvg['type']=='bull' and fvg['gap_low']-atr1*0.2 <= px <= fvg['gap_high']+atr1*0.2: score += 1
        if side=='short' and fvg['type']=='bear' and fvg['gap_low']-atr1*0.2 <= px <= fvg['gap_high']+atr1*0.2: score += 1

    return score

# =============== 포지션/주문 ===============
def fetch_position():
    try:
        pos = ex.fetch_positions([SYMBOL])
    except Exception:
        pos = ex.fetch_positions_risk([SYMBOL])
    for p in pos:
        if p.get('symbol') == SYMBOL and abs(float(p.get('contracts') or 0)) > 0:
            return p
    return None

def in_position(): return fetch_position() is not None

def cancel_reduce_orders():
    try:
        for o in ex.fetch_open_orders(SYMBOL):
            info = o.get('info', {})
            ro = o.get('reduceOnly') or info.get('reduceOnly') or info.get('isReduceOnly')
            if ro:
                try: ex.cancel_order(o['id'], SYMBOL)
                except Exception: pass
    except Exception: pass

def calc_sl_tp(side, frame, last, atr_min=ATR_MIN_MULT, atr_max=ATR_MAX_MULT, rr1=RR_TP1, rr2=RR_TP2):
    atr1 = float(frame['atr14'].iloc[-1])
    lo10 = float(frame['low'].rolling(10).min().iloc[-2])
    hi10 = float(frame['high'].rolling(10).max().iloc[-2])
    if side=='buy':
        swing = lo10 * (1 - SL_BUFFER_PCT)
        risk_raw = max(1e-8, last - swing)
        risk = max(atr_min*atr1, min(risk_raw, atr_max*atr1))
        sl  = last - risk; tp1 = last + risk*rr1; tp2 = last + risk*rr2
    else:
        swing = hi10 * (1 + SL_BUFFER_PCT)
        risk_raw = max(1e-8, swing - last)
        risk = max(atr_min*atr1, min(risk_raw, atr_max*atr1))
        sl  = last + risk; tp1 = last - risk*rr1; tp2 = last - risk*rr2
    return round_price(sl), round_price(tp1), round_price(tp2)

def place_entry_bracket(side, qty, sl, tp1, tp2):
    qty = round_qty(qty)
    q1  = round_qty(qty * TP1_RATIO)
    q2  = round_qty(qty - q1)
    red_side = 'sell' if side=='buy' else 'buy'

    if DRYRUN:
        log_event(DRYRUN=True, action="ORDER_SKIP", side=side, qty=qty, sl=sl, tp1=tp1, tp2=tp2)
        return None

    cancel_reduce_orders()

    entry = ex.create_order(
        SYMBOL, 'market', side, qty, None,
        {"reduceOnly": False, "stopLoss": sl, "slTriggerBy": "MarkPrice"}
    )
    ex.create_order(SYMBOL, 'limit', red_side, q1, tp1, {"reduceOnly": True, "postOnly": True})
    ex.create_order(SYMBOL, 'limit', red_side, q2, tp2, {"reduceOnly": True, "postOnly": True})

    try:
        ex.create_order(SYMBOL, 'stop', red_side, qty, None,
                        {"reduceOnly": True, "triggerPrice": sl, "slTriggerBy": "MarkPrice"})
    except Exception:
        pass

    state.update({"tp1_moved": False, "entry_px": float(entry.get('price') or 0.0), "full_qty": float(qty)})
    log_event(action="ORDER_PLACED", side=side, qty=qty, sl=sl, tp1=tp1, tp2=tp2, entry_id=entry.get('id'))
    return entry

def move_sl_to_breakeven_if_tp1_done():
    if state["tp1_moved"] or state["full_qty"] <= 0: return
    p = fetch_position()
    if not p: return
    remain = abs(float(p.get('contracts') or 0))
    if remain <= state["full_qty"] * (1 - TP1_RATIO*0.95):
        try: ep = float(p.get('entryPrice') or 0) or (state["entry_px"] or 0.0)
        except Exception: ep = state["entry_px"] or 0.0
        if ep == 0.0: return
        side = (p.get('side') or '').lower()  # long/short
        red_side = "sell" if side=="long" else "buy"
        new_sl = round_price(ep + BREAKEVEN_TICK if side=="long" else ep - BREAKEVEN_TICK)
        try:
            ex.create_order(SYMBOL, 'stop', red_side, round_qty(remain), None,
                            {"reduceOnly": True, "triggerPrice": new_sl, "slTriggerBy": "MarkPrice"})
            state["tp1_moved"] = True
            log_event(action="MOVE_SL_TO_BREAKEVEN", new_sl=new_sl, remain_qty=remain)
        except Exception as e:
            log_event(level="WARN", action="MOVE_SL_TO_BREAKEVEN_FAIL", err=str(e))

# =============== 시그널(완화) ===============
def bias_4h(h4):
    c, p = h4.iloc[-1], h4.iloc[-2]
    up = (c['close'] > c['bb_mid'] and p['close'] <= p['bb_mid']) or \
         (c['close'] > c['ema20']  and p['close'] <= p['ema20'])  or \
         (c['macd_hist'] > 0 and p['macd_hist'] <= 0)
    dn = (c['close'] < c['bb_mid'] and p['close'] >= p['bb_mid']) or \
         (c['close'] < c['ema20']  and p['close'] >= p['ema20'])  or \
         (c['macd_hist'] < 0 and p['macd_hist'] >= 0)
    if up and not dn: return "long"
    if dn and not up: return "short"
    return "long" if c['close'] >= c['ema20'] else "short"

def entry_ok(side, h1, h4):
    c1, p1 = h1.iloc[-1], h1.iloc[-2]
    if side == 'buy':
        return (c1['close'] > c1['ema20']) or (c1['rsi14'] > 48) or (c1['macd_hist'] > p1['macd_hist'])
    else:
        return (c1['close'] < c1['ema20']) or (c1['rsi14'] < 52) or (c1['macd_hist'] < p1['macd_hist'])

# =============== 메인 루프 ===============
def run_once():
    h4  = build_frame(TF_4H, 400)
    h1  = build_frame(TF_1H,  400)
    m15 = build_frame(TF_15M, 300)  # 현재는 점수 가점에 직접 사용 X

    last  = float(h1.iloc[-1]['close'])
    trend = bias_4h(h4)

    fib_ret_h1  = build_fib_retracement(h1)
    fib_chan_h1 = build_fib_channel(h1)
    ob_h1       = detect_ob_side(h1)
    fvg_h1      = detect_fvg(h1)

    side     = 'buy' if trend == 'long' else 'sell'
    side_key = 'long' if side=='buy' else 'short'
    score    = confluence_score(last, h1, fib_ret_h1, fib_chan_h1, ob_h1, fvg_h1, side_key)

    has_pos = in_position()
    print(f"[HB] {now_utc()} SOL px={last:.3f} trend(4H)={trend} conf={score} inPos={has_pos}")

    if has_pos:
        move_sl_to_breakeven_if_tp1_done()
        return

    if entry_ok(side, h1, h4) and score >= CONF_REQ:
        sl, tp1, tp2 = calc_sl_tp(side, h1, last)
        place_entry_bracket(side, ORDER_QTY_COIN, sl, tp1, tp2)
        log_event(action="ENTER", side=side, price=last, sl=sl, tp1=tp1, tp2=tp2,
                  conf=score, ob=(ob_h1 or {}).get("side"), fvg=(fvg_h1 or {}).get("type"))

def main_loop():
    log_event(action="START", live=(DRYRUN==0), symbol=SYMBOL, dryrun=bool(DRYRUN))
    while True:
        try:
            run_once()
        except Exception as e:
            log_event(level="ERROR", msg=str(e))
            time.sleep(5)
        time.sleep(LOOP_SLEEP_SEC)

if __name__ == "__main__":
    main_loop()
