# bot_alt_multi_entry_live_final.py
# Bybit USDT Perp — ETH / SOL / BNB (BTC는 취급하지 않음)
# 실전용: 컨플루언스 비중(10/20/30) + 2분할(6:4)/단발, 트레일링 스톱/익절, 동시진입 캡, 공격형(반대포지션 자동청산)
# 키: 환경변수 BYBIT_KEY / BYBIT_SECRET
# 필요 라이브러리: ccxt, pandas

import os, time, json, ccxt
import pandas as pd
from datetime import datetime, timezone
from ccxt.base.errors import NetworkError, ExchangeError

# ======================= 설정 =======================
DRYRUN = 0  # 0=실거래, 1=모의
API_KEY    = os.getenv("BYBIT_KEY")
API_SECRET = os.getenv("BYBIT_SECRET")

SYMBOLS = ["ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]

# 공격형: 신호와 반대 포지션 존재 시 자동 청산 후 재진입 (BTC는 대상 아님)
AGGRESSIVE_MODE = True

MAX_LEV     = 10
MARGIN_MODE = "cross"

# 컨플루언스 4(=30%) 기준 목표 증거금에 맞춘 기본 수량(예시)
ORDER_QTY = {
    "ETH/USDT:USDT": 12,    # conf4 → 3.6 ETH
    "SOL/USDT:USDT": 230,   # conf4 → 69 SOL
    "BNB/USDT:USDT": 67,    # conf4 → 20 BNB
}

# 트레일링/익절
TRAILING_STOP_PCT = 0.02   # 2%
TAKE_PROFIT_PCT   = 0.05   # 5%

# 컨플루언스
FIB_LEVELS   = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_TOL_ATR  = 0.6
CHAN_TOL_ATR = 0.6
CONF_REQ     = 2

# 루프/가드
LOOP_SLEEP_SEC = 10
COOLDOWN_SEC   = 15
HB_EVERY       = 6
TF_1D, TF_4H, TF_1H, TF_15M = "1d", "4h", "1h", "15m"

# 분할/비중
SPLIT_FIRST_RATIO    = 0.60
SPLIT_SECOND_RATIO   = 0.40
SECOND_LEG_OFFSET_ATR = 0.4  # 2차: entry_px ± ATR*0.4

# 동시 포지션 하드캡
MAX_ACTIVE_POSITIONS = 2

LOG_PATH = "./trade_log_multi.jsonl"

# 상태
state = {s: {"position": None, "entry_px": 0.0,
             "trail_max": 0.0, "trail_min": 0.0,
             "last_entry_ts": 0.0, "loop_cnt": 0}
         for s in SYMBOLS}

# ======================= 유틸/지표 =======================
def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
def log_event(**kv):
    kv["ts"] = now_utc()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG_WRITE_FAIL] {e}")
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

# ======================= 거래소 연결 =======================
def get_ex():
    if not API_KEY or not API_SECRET:
        log_event(level="CRITICAL", msg="환경변수 BYBIT_KEY/BYBIT_SECRET 필요")
        raise RuntimeError("BYBIT_KEY/BYBIT_SECRET 필요")
    ex = ccxt.bybit({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "recvWindow": 5000}
    })
    try:
        ex.load_markets()
    except (NetworkError, ExchangeError) as e:
        log_event(level="ERROR", msg="load_markets 실패", err=str(e)); raise
    for sym in SYMBOLS:
        try: ex.set_position_mode(False, sym)   # one-way
        except Exception as e: log_event(level="WARN", symbol=sym, msg="set_position_mode 실패", err=str(e))
        try: ex.set_margin_mode(MARGIN_MODE, sym)
        except Exception as e: log_event(level="WARN", symbol=sym, msg="set_margin_mode 실패", err=str(e))
        try: ex.set_leverage(int(MAX_LEV), sym, {"marginMode": MARGIN_MODE})
        except Exception as e: log_event(level="WARN", symbol=sym, msg="set_leverage 실패", err=str(e))
    return ex

ex = get_ex()

# ======================= 라운딩/데이터 =======================
def round_qty(symbol, q):
    try: return float(ex.amount_to_precision(symbol, q))
    except Exception: return float(f"{q:.6f}")
def round_price(symbol, p):
    try: return float(ex.price_to_precision(symbol, p))
    except Exception: return float(f"{p:.6f}")

def fetch_ohlcv(symbol, tf, limit=300):
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        return df
    except (NetworkError, ExchangeError) as e:
        log_event(level="ERROR", symbol=symbol, msg=f"OHLCV 로드 실패({tf})", err=str(e))
        return pd.DataFrame()

def build_frame(symbol, tf, limit):
    df = fetch_ohlcv(symbol, tf, limit)
    if df.empty:
        return pd.DataFrame()
    df['ema20']  = ema(df['close'], 20)
    df['ema50']  = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14']  = rsi(df['close'], 14)
    df['atr14']  = atr(df, 14)
    up,mid,lo,w  = boll(df['close'], 20, 2)
    df['bb_up'], df['bb_mid'], df['bb_lo'], df['bb_w'] = up, mid, lo, w
    ml, ms, mh = macd(df['close'])
    df['macd'], df['macd_sig'], df['macd_hist'] = ml, ms, mh
    df['vol_sma'] = sma(df['vol'], 20)
    return df

# ======================= 구조/컨플루언스 =======================
def swing_points(series, left=3, right=3, kind='low'):
    s, idxs = series, []
    for i in range(left, len(s)-right):
        win = s.iloc[i-left:i+right+1]
        if kind=='low'  and s.iloc[i] == win.min(): idxs.append(i)
        if kind=='high' and s.iloc[i] == win.max(): idxs.append(i)
    return idxs

def fib_levels(low, high):
    diff = high - low
    return {lvl: high - diff*lvl for lvl in FIB_LEVELS}

def build_fib_retracement(df):
    if df.empty or len(df) < 10: return None
    lows  = swing_points(df['low'],  left=3, right=3, kind='low')
    highs = swing_points(df['high'], left=3, right=3, kind='high')
    if len(lows) < 1 or len(highs) < 1: return None
    lo_idx = lows[-1]; hi_idx = highs[-1]
    lo = float(df.loc[lo_idx, 'low']); hi = float(df.loc[hi_idx, 'high'])
    if hi < lo: lo, hi = hi, lo
    return {"low": lo, "high": hi, "levels": fib_levels(lo, hi)}

def build_fib_channel(df):
    if df.empty or len(df) < 10: return None
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

def detect_ob_side(df):
    if df.empty or len(df) < 20: return None
    try:
        last20 = df.iloc[-20:]
        bodies = (last20['close'] - last20['open']).abs()
        if bodies.empty: return None
        k = int(bodies.idxmax())
        row = df.loc[k]
        bull = row['close'] > row['open']
        ob_low  = float(min(row['open'], row['close']))
        ob_high = float(max(row['open'], row['close']))
        return {"side": "bull" if bull else "bear", "low": ob_low, "high": ob_high}
    except Exception:
        return None

def detect_fvg(df):
    if df.empty or len(df) < 3: return None
    try:
        a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if b['low'] > a['high'] and c['low'] < b['low']:
            return {"type":"bull", "gap_low": float(a['high']), "gap_high": float(b['low'])}
        if b['high'] < a['low'] and c['high'] > b['high']:
            return {"type":"bear", "gap_low": float(b['high']), "gap_high": float(a['low'])}
        return None
    except Exception:
        return None

def confluence_score(px, frame, fib_ret, fib_chan, ob, fvg, side_key):
    if frame.empty or 'atr14' not in frame.columns or pd.isna(frame['atr14'].iloc[-1]): return 0
    score = 0
    atr1 = float(frame['atr14'].iloc[-1])

    if fib_ret:
        pref = [0.5, 0.618, 0.786] if side_key=='long' else [0.382, 0.5, 0.236]
        for lvl, lvl_px in fib_ret['levels'].items():
            if abs(px - lvl_px) <= atr1*FIB_TOL_ATR: score += 2 if lvl in pref else 1

    if fib_chan:
        if side_key=='long'  and abs(px - fib_chan['lower_now']) <= atr1*CHAN_TOL_ATR: score += 2
        if side_key=='short' and abs(px - fib_chan['upper_now']) <= atr1*CHAN_TOL_ATR: score += 2
        if abs(px - fib_chan['mid_now']) <= atr1*CHAN_TOL_ATR: score += 1

    if ob:
        if side_key=='long'  and ob['side']=='bull' and ob['low']-atr1*0.2 <= px <= ob['high']+atr1*0.2: score += 1
        if side_key=='short' and ob['side']=='bear' and ob['low']-atr1*0.2 <= px <= ob['high']+atr1*0.2: score += 1

    if fvg:
        if side_key=='long'  and fvg['type']=='bull' and fvg['gap_low']-atr1*0.2 <= px <= fvg['gap_high']+atr1*0.2: score += 1
        if side_key=='short' and fvg['type']=='bear' and fvg['gap_low']-atr1*0.2 <= px <= fvg['gap_high']+atr1*0.2: score += 1

    return score

# ======================= 포지션/주문 =======================
def fetch_position(symbol):
    try:
        pos = ex.fetch_positions([symbol])
    except (NetworkError, ExchangeError):
        try: pos = ex.fetch_positions_risk([symbol])
        except Exception as e:
            log_event(level="ERROR", symbol=symbol, msg="fetch_positions 실패", err=str(e)); return None
    for p in pos:
        if p.get('symbol') == symbol and abs(float(p.get('contracts') or 0)) > 1e-8:
            return p
    return None

def in_position(symbol):
    try: return fetch_position(symbol) is not None
    except Exception: return False

def active_positions_count():
    c = 0
    for s in SYMBOLS:
        try:
            if in_position(s): c += 1
        except Exception: pass
    return c

def close_position(symbol, qty, side):
    red_side = 'sell' if side=='buy' else 'buy'
    if DRYRUN:
        log_event(DRYRUN=True, action="CLOSE_SKIP", symbol=symbol, side=red_side, qty=qty); return
    try:
        ex.create_order(symbol, 'market', red_side, round_qty(symbol, qty), None, {'reduceOnly': True})
        log_event(action="POSITION_CLOSED", symbol=symbol, side=red_side, qty=qty)
    except (NetworkError, ExchangeError) as e:
        log_event(level="ERROR", action="CLOSE_FAIL", symbol=symbol, err=str(e))

def close_position_if_opposite(symbol, desired_side):
    p = fetch_position(symbol)
    if not p: return True
    cur_side = (p.get('side') or '').lower()  # long/short
    # 같은 방향이면 유지
    if (cur_side == 'long' and desired_side == 'buy') or (cur_side == 'short' and desired_side == 'sell'):
        return True
    # 반대면 청산
    qty = abs(float(p.get('contracts') or 0))
    if qty <= 0: return True
    try:
        try: ex.cancel_all_orders(symbol)
        except Exception: pass
        if DRYRUN:
            log_event(DRYRUN=True, action="CLOSE_POS_SIM", symbol=symbol, close_side=('sell' if cur_side=='long' else 'buy'), qty=qty)
            return True
        ex.create_order(symbol, 'market', ('sell' if cur_side=='long' else 'buy'),
                        round_qty(symbol, qty), None, {"reduceOnly": True})
        time.sleep(0.5)
        return fetch_position(symbol) is None
    except Exception as e:
        log_event(level="ERROR", action="CLOSE_POS_FAIL", symbol=symbol, err=str(e))
        return False

def place_entry_order(symbol, side, qty):
    qty = round_qty(symbol, qty)
    if qty <= 0: return None
    if DRYRUN:
        entry = {"id": "dryrun", "price": ex.fetch_ticker(symbol).get("last")}
        entry_px = float(entry.get('price') or 0.0)
        state[symbol].update({"position": side, "entry_px": entry_px,
                              "trail_max": entry_px, "trail_min": entry_px,
                              "last_entry_ts": time.time()})
        log_event(DRYRUN=True, action="ORDER_SKIP", symbol=symbol, side=side, qty=qty)
        return entry
    try:
        entry = ex.create_order(symbol, 'market', side, qty, None, {"reduceOnly": False})
        entry_px = float(entry.get('price') or 0.0)
        state[symbol].update({"position": side, "entry_px": entry_px,
                              "trail_max": entry_px, "trail_min": entry_px,
                              "last_entry_ts": time.time()})
        log_event(action="ORDER_PLACED", symbol=symbol, side=side, qty=qty, entry_id=entry.get('id'))
        return entry
    except (NetworkError, ExchangeError) as e:
        log_event(level="ERROR", action="ENTRY_ORDER_FAIL", symbol=symbol, err=str(e))
        return None

def manage_position(symbol, df):
    st = state[symbol]
    p = fetch_position(symbol)
    if not p:
        st.update({"position": None, "entry_px": 0.0, "trail_max": 0.0, "trail_min": 0.0})
        return

    last_px = float(p.get('info', {}).get('markPrice') or 0.0)
    if last_px == 0.0:
        try: last_px = float(ex.fetch_ticker(symbol)['last'] or 0.0)
        except Exception: return

    pos_side = (p.get('side') or '').lower()  # long/short
    qty = abs(float(p.get('contracts') or 0))
    entry_px = st.get('entry_px', 0.0)

    # 익절
    if pos_side == 'long':
        if entry_px > 0 and (last_px / entry_px - 1) >= TAKE_PROFIT_PCT:
            log_event(action="TAKE_PROFIT", symbol=symbol, target=TAKE_PROFIT_PCT)
            close_position(symbol, qty, 'buy'); return
    elif pos_side == 'short':
        if entry_px > 0 and (entry_px / last_px - 1) >= TAKE_PROFIT_PCT:
            log_event(action="TAKE_PROFIT", symbol=symbol, target=TAKE_PROFIT_PCT)
            close_position(symbol, qty, 'short'); return

    # 트레일링
    if pos_side == 'long':
        st["trail_max"] = max(st.get("trail_max", last_px), last_px)
        if st["trail_max"] > 0 and (last_px / st["trail_max"] - 1) <= -TRAILING_STOP_PCT:
            log_event(action="TRAILING_STOP", symbol=symbol, drop_pct=TRAILING_STOP_PCT)
            close_position(symbol, qty, 'buy'); return
    elif pos_side == 'short':
        st["trail_min"] = min(st.get("trail_min", last_px), last_px)
        if st["trail_min"] > 0 and ((last_px / st["trail_min"]) - 1) >= TRAILING_STOP_PCT:
            log_event(action="TRAILING_STOP", symbol=symbol, drop_pct=TRAILING_STOP_PCT)
            close_position(symbol, qty, 'short'); return

    # 간단한 반전 시그널 종료
    if not df.empty and len(df) >= 2:
        if pos_side == 'long' and (df.iloc[-1]['macd_hist'] < 0 and df.iloc[-2]['macd_hist'] >= 0):
            if df.iloc[-1]['rsi14'] < 70:
                log_event(action="SELL_SIGNAL_EXIT", symbol=symbol)
                close_position(symbol, qty, 'buy'); return
        if pos_side == 'short' and (df.iloc[-1]['macd_hist'] > 0 and df.iloc[-2]['macd_hist'] <= 0):
            if df.iloc[-1]['rsi14'] > 50:
                log_event(action="BUY_SIGNAL_EXIT", symbol=symbol)
                close_position(symbol, qty, 'short'); return

# ======================= 방향/트리거 =======================
def entry_ok(side, h1, h4, h1d):
    if h1.empty or h4.empty or h1d.empty: return False
    c1, p1 = h1.iloc[-1], h1.iloc[-2]

    if side == 'buy':
        is_h1_ok = (c1['macd_hist'] > 0 and p1['macd_hist'] <= 0 and
                    50 < c1['rsi14'] < 70 and c1['close'] > c1['ema200'])
    else:
        is_h1_ok = (c1['macd_hist'] < 0 and p1['macd_hist'] >= 0 and
                    c1['rsi14'] < 50 and c1['close'] < c1['ema200'])

    trend_1d_ok = (h1d.iloc[-1]['close'] > h1d.iloc[-1]['ema200']) if side == 'buy' else (h1d.iloc[-1]['close'] < h1d.iloc[-1]['ema200'])
    trend_4h_ok = (h4.iloc[-1]['close']  > h4.iloc[-1]['ema200']) if side == 'buy' else (h4.iloc[-1]['close']  < h4.iloc[-1]['ema200'])
    is_trend_aligned = trend_1d_ok and trend_4h_ok

    is_vol_ok = c1['vol'] > c1['vol_sma'] * 1.5
    return is_h1_ok and is_trend_aligned and is_vol_ok

def bias_trend(df):
    if df.empty or 'ema200' not in df.columns: return "unknown"
    c = df.iloc[-1]
    if c['close'] > c['ema200']: return "long"
    if c['close'] < c['ema200']: return "short"
    return "long" if c['close'] >= c['ema20'] else "short"

# ======================= 비중/분할 =======================
def size_scale_from_conf(conf):
    if conf <= 1: return 0.0
    if conf == 2: return 0.10
    if conf == 3: return 0.20
    return 0.30

def decide_entry_plan(symbol, conf, base_qty):
    scale = size_scale_from_conf(conf)
    q_total = round_qty(symbol, base_qty * scale)
    if q_total <= 0:
        return {"mode":"none","q_total":0.0,"q1":0.0,"q2":0.0}
    if conf in (2, 3):
        q1 = round_qty(symbol, q_total * SPLIT_FIRST_RATIO)
        q2 = round_qty(symbol, max(0.0, q_total - q1))
        return {"mode":"split","q_total":q_total,"q1":q1,"q2":q2}
    else:
        return {"mode":"single","q_total":q_total,"q1":0.0,"q2":0.0}

# ======================= 메인 루프 =======================
def run_once(symbol):
    h1d = build_frame(symbol, TF_1D, 400)
    h4  = build_frame(symbol, TF_4H, 400)
    h1  = build_frame(symbol, TF_1H, 400)

    if h1.empty or h4.empty or h1d.empty:
        log_event(level="WARN", symbol=symbol, msg="프레임 데이터 부족"); return

    last  = float(h1.iloc[-1]['close'])
    trend = bias_trend(h4)
    if trend == "unknown":
        log_event(level="WARN", symbol=symbol, msg="추세 불명"); return

    fib_ret_h1  = build_fib_retracement(h1)
    fib_chan_h1 = build_fib_channel(h1)
    ob_h1       = detect_ob_side(h1)
    fvg_h1      = detect_fvg(h1)

    side     = 'buy' if trend == 'long' else 'sell'
    side_key = 'long' if side=='buy' else 'short'
    score    = confluence_score(last, h1, fib_ret_h1, fib_chan_h1, ob_h1, fvg_h1, side_key)

    st = state[symbol]; st["loop_cnt"] += 1
    if st["loop_cnt"] % HB_EVERY == 0:
        log_event(hb=True, symbol=symbol, px=last, trend=trend, conf=score, inPos=in_position(symbol))

    # 포지션 있으면 관리
    if in_position(symbol):
        manage_position(symbol, h1); return

    # 쿨다운
    remain = max(0, COOLDOWN_SEC - (time.time() - st.get("last_entry_ts", 0.0)))
    if remain > 0:
        log_event(action="COOLDOWN", symbol=symbol, remaining=round(remain,2)); return

    # 동시 포지션 제한
    if active_positions_count() >= MAX_ACTIVE_POSITIONS:
        log_event(action="MAX_POSITIONS_REACHED", symbol=symbol); return

    # 신호 확인
    if not (entry_ok(side, h1, h4, h1d) and score >= CONF_REQ):
        log_event(action="NO_SIGNAL", symbol=symbol, trend=trend, conf=score); return

    # 공격형: 반대 포지션 자동 정리 (BTC와 무관)
    if AGGRESSIVE_MODE:
        desired_side = 'buy' if trend == 'long' else 'sell'
        if not close_position_if_opposite(symbol, desired_side):
            log_event(level="WARN", action="OPPOSITE_CLOSE_FAILED", symbol=symbol); return

    # 진입 플랜
    plan = decide_entry_plan(symbol, score, ORDER_QTY.get(symbol, 0))
    if plan["mode"] == "none" or plan["q_total"] <= 0:
        log_event(action="PLAN_ABORTED", symbol=symbol, msg="컨플루언스 낮음/수량 0"); return

    if plan["mode"] == "single":
        place_entry_order(symbol, side, plan["q_total"])
        log_event(action="ENTER_SINGLE", symbol=symbol, side=side, q=plan["q_total"], conf=score)

    elif plan["mode"] == "split":
        # 1차 시장
        entry = place_entry_order(symbol, side, plan["q1"])
        if entry:
            log_event(action="ENTER_SPLIT_1", symbol=symbol, side=side, q=plan["q1"], conf=score)
            # 2차 리밋(유리한 가격)
            if plan["q2"] > 0:
                entry_price = float(entry.get('price') or 0.0)
                if entry_price > 0 and 'atr14' in h1.columns:
                    atr1 = float(h1.iloc[-1]['atr14'])
                    if side == 'buy':
                        limit_price = round_price(symbol, entry_price - atr1 * SECOND_LEG_OFFSET_ATR)
                    else:
                        limit_price = round_price(symbol, entry_price + atr1 * SECOND_LEG_OFFSET_ATR)
                    try:
                        if DRYRUN:
                            log_event(DRYRUN=True, action="PLACE_SPLIT_2_LIMIT_SIM", symbol=symbol, q=plan["q2"], limit_px=limit_price)
                        else:
                            ex.create_order(symbol, 'limit', side, plan['q2'], limit_price, {"postOnly": True})
                        log_event(action="PLACE_SPLIT_2_LIMIT", symbol=symbol, side=side, q=plan["q2"], limit_px=limit_price)
                    except Exception as e:
                        log_event(level="ERROR", action="SPLIT_2_ORDER_FAIL", symbol=symbol, err=str(e))
                else:
                    log_event(level="WARN", action="SPLIT_2_ABORT", symbol=symbol, msg="entry_px/ATR 미확인")

def main_loop():
    log_event(action="START", live=(DRYRUN==0), symbols=SYMBOLS, dryrun=bool(DRYRUN))
    backoff = 3
    while True:
        try:
            for sym in SYMBOLS:
                run_once(sym)
            time.sleep(LOOP_SLEEP_SEC)
            backoff = 3
        except KeyboardInterrupt:
            log_event(action="STOP_BY_USER"); break
        except Exception as e:
            log_event(level="CRITICAL", action="MAIN_LOOP_ERROR", err=str(e))
            time.sleep(backoff); backoff = min(backoff*2, 60)

if __name__ == "__main__":
    main_loop()
