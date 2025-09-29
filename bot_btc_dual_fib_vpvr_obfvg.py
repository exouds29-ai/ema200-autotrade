# bot_btc_dual_core_scalp.py
# BTC 전용: 중장기 Core(3분할+TP1/TP2/트레일SL) + 단타 Satellite(해지방향)
# - HEDGE_MODE(양방향) 우선. 불가 시, Core와 충돌 방지 위해 Satellite 자동 중지.
# - 수동 Core 포지션 존중: 봇 주문만 태그로 구분/관리(BOT_*), 수동 주문/포지션은 절대 변경X.
# - Core: 3분할(10/10/10=총 30%), TP1 50% 고정익절, TP2 추세 추종(EMA60/ATR 트레일), TP1 체결 시 SL 본절 이동
# - Satellite: Core 반대 방향만 짧게(스캘프/단타), TP1→본절 이동, RR 구조
# - 디버그/판단 로그 상세(ENTRY_OK/FAIL, DECISION BLOCK 등)

import os, time, json, ccxt
import pandas as pd
from datetime import datetime, timezone
from ccxt.base.errors import NetworkError, ExchangeError

# ============== 기본 설정 ==============
DRYRUN = 0  # 0=실거래, 1=모의
API_KEY    = "eb2aY5kC775hqNDYO2"
API_SECRET = "lcBdmA34s6yrtB3N1GRpu8Uo8I6NYL6NnQLz"

SYMBOL = "BTC/USDT:USDT"

# 포지션 모드
HEDGE_MODE = True   # True=양방향(롱/숏 동시보유 가능). False=단방향(충돌 방지 수행)
MARGIN_MODE = "cross"
MAX_LEV = 10

# 태그(봇이 낸 주문만 관리)
TAG_CORE   = "BOT_CORE"
TAG_SCALP  = "BOT_SCALP"

# ===== Core(중장기) 설정 =====
CORE_ENABLED              = True
CORE_MODE                 = "manual"   # "manual"(수동 코어 존재 가정), "auto"(봇이 코어 진입도 수행), "off"
CORE_MANAGE_MANUAL        = False      # 수동 Core가 있을 때 SL/TP를 봇이 건드리지 않음(False 권장)
CORE_SIDE_MANUAL_HINT     = "long"     # manual일 때 방향 힌트("long"/"short"/None) — 충돌방지용
CORE_TOTAL_QTY            = 0.30       # 총 계좌의 몇 %를 코어로 쓸지(지정 수량 기반 운용 권장, 여기선 코인 수량으로 지정)
CORE_BASE_QTY_COIN        = 3.0        # 코어 총 목표 수량(BTC코인) — 3분할이면 각 10%→ 여기선 1진입당 0.1*3=0.3BTC
CORE_SPLIT_RATIOS         = [0.10, 0.10, 0.10]  # 10%+10%+10% = 총 30%
CORE_ZONE_METHOD          = "fib_ob"   # "fib_ob": D1 스윙기반 fib + 간단 OB/FVG로 세 구간 산출
CORE_TP1_RATIO            = 0.50       # 코어 포지션의 50% 고정 익절
CORE_TP2_TRAIL_EMA        = 60         # TP2 추세 추종 기준 EMA
CORE_TP2_TRAIL_ATR_K      = 0.8        # 트레일 SL ATR 계수
CORE_BREAKEVEN_TICK       = 5.0        # TP1 체결 후 본절 이동(+틱)

# ===== Satellite(단타) 설정 =====
SCALP_ENABLED             = True
SCALP_ONLY_OPPOSITE_CORE  = True  # Core와 반대방향만 진입(해지)
SCALP_QTY_COIN            = 0.05   # 1회 단타 코인 수량(예: 0.05 BTC)
SCALP_TP1_RATIO           = 0.50
SCALP_RR_TP1              = 1.0
SCALP_RR_TP2              = 1.6
SCALP_ATR_MIN_MULT        = 0.5
SCALP_ATR_MAX_MULT        = 1.2
SCALP_BREAKEVEN_TICK      = 2.5
SCALP_COOLDOWN_SEC        = 20

# ===== 데이터/프레임 =====
TF_1W, TF_1D, TF_4H, TF_1H, TF_15M, TF_3M = "1w", "1d", "4h", "1h", "15m", "3m"
HB_EVERY = 6
LOOP_SLEEP = 10
LOG_PATH = "./trade_log_btc_dual.jsonl"

# 상태
state = {
    "loop": 0,
    "last_scalp_entry_ts": 0.0,
    "core": {
        "active": False,          # 봇이 만든 코어 활성 여부
        "side": None,             # long/short
        "filled_qty": 0.0,        # 현재 체결 코어 수량
        "avg_entry": 0.0,
        "tp1_done": False
    }
}

# ============== 유틸/로그/지표 ==============
def now_utc(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
def log_event(**kv):
    kv["ts"] = now_utc()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(kv)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / (rd + 1e-12)
    return 100 - (100 / (1 + rs))
def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast); s = ema(close, slow)
    line = f - s; sig = ema(line, signal); hist = line - sig
    return line, sig, hist
def boll(close, n=20, k=2):
    mid = sma(close, n); std = close.rolling(n).std()
    up, lo = mid + k*std, mid - k*std
    width = (up - lo) / (mid + 1e-12)
    return up, mid, lo, width
def atr(df, n=14):
    hi, lo, cl = df['high'], df['low'], df['close']
    prev = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ============== 거래소 연결 ==============
def get_ex():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("환경변수 BYBIT_KEY/BYBIT_SECRET 필요")
    ex = ccxt.bybit({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "recvWindow": 5000}
    })
    ex.load_markets()
    # 포지션 모드/레버리지
    try:
        ex.set_position_mode(HEDGE_MODE, SYMBOL)   # True=Hedge(양방향), False=One-way
    except Exception as e:
        log_event(level="WARN", msg="set_position_mode 실패 — One-way로 동작", err=str(e))
    try:
        ex.set_margin_mode(MARGIN_MODE, SYMBOL)
    except Exception as e:
        log_event(level="WARN", msg="set_margin_mode 실패", err=str(e))
    try:
        ex.set_leverage(int(MAX_LEV), SYMBOL, {"marginMode": MARGIN_MODE})
    except Exception as e:
        log_event(level="WARN", msg="set_leverage 실패", err=str(e))
    return ex

ex = get_ex()

# ============== 라운딩/데이터 ==============
def round_qty(q):
    try: return float(ex.amount_to_precision(SYMBOL, q))
    except Exception: return float(f"{q:.6f}")
def round_price(p):
    try: return float(ex.price_to_precision(SYMBOL, p))
    except Exception: return float(f"{p:.2f}")

def fetch_ohlcv(tf, limit=300):
    o = ex.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def build_frame(tf, limit):
    df = fetch_ohlcv(tf, limit)
    df['ema20']  = ema(df['close'], 20)
    df['ema60']  = ema(df['close'], 60)
    df['ema120'] = ema(df['close'], 120)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14']  = rsi(df['close'], 14)
    df['atr14']  = atr(df, 14)
    up,mid,lo,w  = boll(df['close'], 20, 2)
    df['bb_up'], df['bb_mid'], df['bb_lo'], df['bb_w'] = up, mid, lo, w
    ml, ms, mh = macd(df['close'])
    df['macd'], df['macd_sig'], df['macd_hist'] = ml, ms, mh
    return df

# ============== 스윙 포인트/피보/간단 OB/FVG ==============
def swing_points(series, left=5, right=5, kind='low'):
    s = series; idxs = []
    for i in range(left, len(s)-right):
        w = s.iloc[i-left:i+right+1]
        if kind=='low'  and s.iloc[i] == w.min(): idxs.append(i)
        if kind=='high' and s.iloc[i] == w.max(): idxs.append(i)
    return idxs

def build_fib_retracement(df):
    lows  = swing_points(df['low'],  left=5, right=5, kind='low')
    highs = swing_points(df['high'], left=5, right=5, kind='high')
    if not lows or not highs: return None
    lo_idx, hi_idx = lows[-1], highs[-1]
    lo = float(df.loc[lo_idx,'low']); hi = float(df.loc[hi_idx,'high'])
    if hi < lo: lo, hi = hi, lo
    diff = hi - lo
    levels = {
        0.382: hi - diff*0.382,
        0.5:   hi - diff*0.5,
        0.618: hi - diff*0.618,
        0.786: hi - diff*0.786
    }
    return {"low":lo, "high":hi, "levels":levels}

def detect_ob_d1(df):
    # 일봉 기준 가장 큰 몸통 30봉 내
    try:
        lastN = df.iloc[-30:]
        bodies = (lastN['close'] - lastN['open']).abs()
        k = int(bodies.idxmax())
        row = df.loc[k]
        bull = row['close'] > row['open']
        lo  = float(min(row['open'], row['close']))
        hi  = float(max(row['open'], row['close']))
        return {"side":"bull" if bull else "bear", "low":lo, "high":hi}
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

# ============== 포지션/주문 공통 ==============
def position_idx_for(side):
    if not HEDGE_MODE: return None  # one-way
    return 2 if side=='buy' else 3  # 2=long, 3=short (Bybit)

def fetch_positions():
    try:
        return ex.fetch_positions([SYMBOL])
    except Exception:
        try: return ex.fetch_positions_risk([SYMBOL])
        except Exception as e:
            log_event(level="ERROR", action="FETCH_POS_FAIL", err=str(e)); return []

def get_net_position():
    # one-way일 때 유효: 전체 순포지션
    for p in fetch_positions():
        if p.get('symbol')==SYMBOL and abs(float(p.get('contracts') or 0))>0:
            return p
    return None

def get_hedged_positions():
    # hedge 모드에서 롱/숏 따로 반환
    longs, shorts = None, None
    for p in fetch_positions():
        if p.get('symbol')!=SYMBOL: continue
        side = (p.get('side') or '').lower()  # long/short
        if side=='long' and abs(float(p.get('contracts') or 0))>0: longs = p
        if side=='short' and abs(float(p.get('contracts') or 0))>0: shorts = p
    return longs, shorts

def fetch_open_orders(tag_prefix=None):
    try:
        oo = ex.fetch_open_orders(SYMBOL)
        if tag_prefix is None: return oo
        res = []
        for o in oo:
            link = o.get('clientOrderId') or o.get('info',{}).get('orderLinkId') or ""
            if str(link).startswith(tag_prefix):
                res.append(o)
        return res
    except Exception:
        return []

def cancel_tagged_orders(tag_prefix):
    for o in fetch_open_orders(tag_prefix):
        try: ex.cancel_order(o['id'], SYMBOL)
        except Exception: pass

def create_order(side, type_, qty, price=None, reduceOnly=False, tag=None, extra=None):
    params = {"reduceOnly": reduceOnly}
    if tag: params["clientOrderId"] = tag
    idx = position_idx_for(side)
    if idx is not None: params["positionIdx"] = idx
    if extra: params.update(extra)
    return ex.create_order(SYMBOL, type_, side, round_qty(qty), None if type_=='market' else round_price(price), params)

# ============== Core 진입/관리 ==============
def build_core_zones(d1):
    fib = build_fib_retracement(d1)
    ob  = detect_ob_d1(d1)
    # 롱 기준 예시(바닥 매집): A(공격)=fib 0.5, B(중립)=0.618, C(보수)=0.786(또는 bull OB 영역 하단)
    if not fib:
        return None
    L = fib['levels']
    a = L.get(0.5); b = L.get(0.618); c = L.get(0.786)
    if ob and ob['side']=='bull':
        c = min(c, ob['low'])
    zones = [a, b, c]
    zones = [z for z in zones if z is not None]
    zones = sorted(zones, reverse=False)
    return zones  # [공격, 중립, 보수] 낮은가격→높은가격(롱기준)

def core_place_split_limits(side, total_qty, zones, avg_px_ref):
    # 3분할 limit 세팅 (reduceOnly=False)
    if len(zones) < 3: return
    parts = [total_qty*r for r in CORE_SPLIT_RATIOS]
    tags  = [f"{TAG_CORE}_A1", f"{TAG_CORE}_A2", f"{TAG_CORE}_A3"]
    pxs   = zones if side=='buy' else list(reversed(zones))  # 숏이면 위에서부터 체결
    for q, px, tag in zip(parts, pxs, tags):
        try:
            create_order(side, 'limit', q, price=px, reduceOnly=False, tag=tag, extra={"postOnly": True})
            log_event(action="CORE_LIMIT_SET", side=side, qty=q, px=px, tag=tag)
        except Exception as e:
            log_event(level="WARN", action="CORE_LIMIT_FAIL", err=str(e), tag=tag)

def core_manage_trailing_and_tp(d1, pos_side, qty, avg_entry):
    # TP1 고정: 직전 D1 swing high(롱) / swing low(숏)
    # TP2: EMA60 기반 트레일 + ATR*k 버퍼
    d1_last = d1.iloc[-1]
    atr1    = float(d1['atr14'].iloc[-1])
    ema60   = float(d1['ema60'].iloc[-1])
    # swing 목표
    lows  = swing_points(d1['low'],  left=5, right=5, kind='low')
    highs = swing_points(d1['high'], left=5, right=5, kind='high')
    if pos_side=='long':
        tp1_level = float(d1.loc[highs[-1],'high']) if highs else avg_entry*1.03
        trail_sl  = ema60 - atr1*CORE_TP2_TRAIL_ATR_K
    else:
        tp1_level = float(d1.loc[lows[-1],'low']) if lows else avg_entry*0.97
        trail_sl  = ema60 + atr1*CORE_TP2_TRAIL_ATR_K

    # 이미 열려있는 봇 코어용 reduceOnly 주문만 갱신
    cancel_tagged_orders(f"{TAG_CORE}_TP")
    cancel_tagged_orders(f"{TAG_CORE}_SL")

    # TP1: 잔량의 50% 한정. TP1 체결 후 SL 본절 이동은 fill 모니터링으로 처리(간소화: 가격 도달 시 생성)
    half = round_qty(qty * CORE_TP1_RATIO)
    red_side = 'sell' if pos_side=='long' else 'buy'
    try:
        create_order(red_side, 'limit', half, price=tp1_level, reduceOnly=True, tag=f"{TAG_CORE}_TP1")
        log_event(action="CORE_TP1_SET", level=tp1_level, qty=half)
    except Exception as e:
        log_event(level="WARN", action="CORE_TP1_FAIL", err=str(e))

    # TP2 트레일은 SL로 운영(가격이 추세 이탈 시 청산)
    try:
        create_order(red_side, 'stop', qty, price=None, reduceOnly=True, tag=f"{TAG_CORE}_SL",
                     extra={"triggerPrice": round_price(trail_sl), "slTriggerBy": "MarkPrice"})
        log_event(action="CORE_TRAIL_SL_SET", trail_sl=round_price(trail_sl))
    except Exception as e:
        log_event(level="WARN", action="CORE_TRAIL_SL_FAIL", err=str(e))

def core_detect_and_move_to_breakeven_if_tp1_hit(mark_px, pos_side, avg_entry):
    # 가격이 TP1에 도달했는지 단순 판단 후, 본절(+틱)로 SL 재배치
    # (TP1 채결 여부 정확 추적은 거래소 fills 조회 필요 — 여기선 가격도달/남은수량으로 근사)
    try:
        if pos_side=='long' and mark_px >= avg_entry:  # 최소 본절 근처
            new_sl = round_price(avg_entry + CORE_BREAKEVEN_TICK)
        elif pos_side=='short' and mark_px <= avg_entry:
            new_sl = round_price(avg_entry - CORE_BREAKEVEN_TICK)
        else:
            return
        cancel_tagged_orders(f"{TAG_CORE}_SL")
        red_side = 'sell' if pos_side=='long' else 'buy'
        create_order(red_side, 'stop', 1e-8, price=None, reduceOnly=True, tag=f"{TAG_CORE}_SL",  # qty는 아래서 대체
                     extra={"triggerPrice": new_sl, "slTriggerBy": "MarkPrice"})
        log_event(action="CORE_MOVE_SL_TO_BE", new_sl=new_sl)
    except Exception as e:
        log_event(level="WARN", action="CORE_BE_FAIL", err=str(e))

# ============== Satellite(단타) 진입/관리 ==============
def calc_scalp_sl_tp(side, h1, last):
    atr1 = float(h1['atr14'].iloc[-1])
    lo10 = float(h1['low'].rolling(10).min().iloc[-2])
    hi10 = float(h1['high'].rolling(10).max().iloc[-2])
    if side == 'buy':
        swing = lo10 * (1 - 0.003)
        risk_raw = max(1e-8, last - swing)
        risk = max(SCALP_ATR_MIN_MULT*atr1, min(risk_raw, SCALP_ATR_MAX_MULT*atr1))
        sl  = last - risk; tp1 = last + risk*SCALP_RR_TP1; tp2 = last + risk*SCALP_RR_TP2
    else:
        swing = hi10 * (1 + 0.003)
        risk_raw = max(1e-8, swing - last)
        risk = max(SCALP_ATR_MIN_MULT*atr1, min(risk_raw, SCALP_ATR_MAX_MULT*atr1))
        sl  = last + risk; tp1 = last - risk*SCALP_RR_TP1; tp2 = last - risk*SCALP_RR_TP2
    return round_price(sl), round_price(tp1), round_price(tp2)

def scalp_signal(side, h1, h15, h4):
    # 단타: 4H 방향 보조, 1H/15m 트리거(간단)
    c1, p1 = h1.iloc[-1], h1.iloc[-2]
    ema_ok = (c1['close'] > c1['ema200']) if side=='buy' else (c1['close'] < c1['ema200'])
    rsi_ok = (c1['rsi14'] >= 48) if side=='buy' else (c1['rsi14'] <= 52)
    macd_ok= (c1['macd_hist'] > p1['macd_hist']) if side=='buy' else (c1['macd_hist'] < p1['macd_hist'])
    h4_ok  = (h4.iloc[-1]['close'] > h4.iloc[-1]['ema60']) if side=='buy' else (h4.iloc[-1]['close'] < h4.iloc[-1]['ema60'])
    # 완화: EMA+RSI가 맞으면 통과, MACD/4H는 보조(가점) → 여기선 단순화해 TRUE 비중 ↑
    ok = ema_ok and rsi_ok
    score = (1 if ema_ok else 0) + (1 if rsi_ok else 0) + (1 if macd_ok else 0) + (1 if h4_ok else 0)
    return ok, score

def scalp_place_bracket(side, qty, sl, tp1, tp2):
    if DRYRUN:
        log_event(DRYRUN=True, action="SCALP_ORDER_SKIP", side=side, qty=qty, sl=sl, tp1=tp1, tp2=tp2); return
    # 엔트리
    entry = create_order(side, 'market', qty, reduceOnly=False, tag=f"{TAG_SCALP}_ENTRY")
    # TP1/TP2
    red_side = 'sell' if side=='buy' else 'buy'
    q1 = round_qty(qty * SCALP_TP1_RATIO)
    q2 = round_qty(max(0.0, qty - q1))
    try:
        create_order(red_side, 'limit', q1, price=tp1, reduceOnly=True, tag=f"{TAG_SCALP}_TP1", extra={"postOnly": True})
        create_order(red_side, 'limit', q2, price=tp2, reduceOnly=True, tag=f"{TAG_SCALP}_TP2", extra={"postOnly": True})
        # 보호 SL
        create_order(red_side, 'stop', qty, price=None, reduceOnly=True, tag=f"{TAG_SCALP}_SL",
                     extra={"triggerPrice": sl, "slTriggerBy": "MarkPrice"})
    except Exception as e:
        log_event(level="WARN", action="SCALP_BRACKET_FAIL", err=str(e))
    return entry

def scalp_move_be_if_tp1_done():
    # 간소: TP1 체결 후 남은 수량에 대해 SL 본절(+틱)
    # 정확체결 확인은 fills 필요 — 여기선 남은계약/가격으로 근사
    longs, shorts = get_hedged_positions() if HEDGE_MODE else (get_net_position(), None)
    p = None
    if HEDGE_MODE:
        p = longs or shorts
    else:
        p = get_net_position()
    if not p: return
    side = (p.get('side') or '').lower()  # long/short
    ep = float(p.get('entryPrice') or 0.0)
    if ep<=0: return
    remain = abs(float(p.get('contracts') or 0))
    # 남은 reduceOnly 오더 점검은 생략 — 단순히 SL 교체
    red_side = 'sell' if side=='long' else 'buy'
    new_sl = round_price(ep + SCALP_BREAKEVEN_TICK if side=='long' else ep - SCALP_BREAKEVEN_TICK)
    cancel_tagged_orders(f"{TAG_SCALP}_SL")
    try:
        create_order(red_side, 'stop', remain, price=None, reduceOnly=True, tag=f"{TAG_SCALP}_SL",
                     extra={"triggerPrice": new_sl, "slTriggerBy": "MarkPrice"})
        log_event(action="SCALP_MOVE_BE", new_sl=new_sl, remain=remain)
    except Exception as e:
        log_event(level="WARN", action="SCALP_MOVE_BE_FAIL", err=str(e))

# ============== 메인 루프 ==============
def main_loop():
    log_event(action="START", symbol=SYMBOL, hedge=HEDGE_MODE, live=(DRYRUN==0))
    backoff = 3
    while True:
        try:
            loop_once()
            time.sleep(LOOP_SLEEP)
            backoff = 3
        except KeyboardInterrupt:
            log_event(action="STOP_BY_USER"); break
        except Exception as e:
            log_event(level="ERROR", action="MAIN_ERR", err=str(e))
            time.sleep(backoff); backoff = min(backoff*2, 60)

def loop_once():
    state["loop"] += 1

    # 프레임
    d1  = build_frame(TF_1D, 400)
    h4  = build_frame(TF_4H, 400)
    h1  = build_frame(TF_1H, 400)
    h15 = build_frame(TF_15M, 300)

    last = float(h1.iloc[-1]['close'])
    mark_px = last

    # 현재 보유 포지션 파악(충돌 방지)
    if HEDGE_MODE:
        long_pos, short_pos = get_hedged_positions()
        net_has = (long_pos is not None) or (short_pos is not None)
        core_dir_manual = CORE_SIDE_MANUAL_HINT
    else:
        net_p = get_net_position()
        net_has = net_p is not None
        core_dir_manual = (net_p.get('side') or '').lower() if net_p else CORE_SIDE_MANUAL_HINT

    if state["loop"] % HB_EVERY == 0:
        log_event(hb=True, px=last, hasPos=net_has, manual_core_hint=core_dir_manual)

    # ===== Core (중장기) =====
    if CORE_ENABLED:
        # 수동 Core 존중: manual + manage_manual=False 면 진입/관리 X (충돌 방지)
        if CORE_MODE == "manual" and not CORE_MANAGE_MANUAL:
            # 단지 코어 방향 힌트만 반영해 단타 해지 여부에 사용
            pass
        elif CORE_MODE == "auto":
            # 자동 코어 진입(옵션) — 기본은 off. 필요하면 켜세요.
            if not state["core"]["active"]:
                # 롱 예시: D1 볼밴 하단 복귀 + RSI 다이버전스 등 고급 룰 가능. 여기선 간소화: fib/ob 기반 구간에 3분할 예약.
                zones = build_core_zones(d1)
                if zones and len(zones)>=3:
                    side = "buy"  # 예시: 바닥 매집용. 고점 숏 매집은 반대로 설계 가능.
                    total_qty = sum(CORE_SPLIT_RATIOS) * CORE_BASE_QTY_COIN
                    core_place_split_limits(side, total_qty, zones, avg_px_ref=last)
                    state["core"].update({"active": True, "side": side})
                    log_event(action="CORE_SPLIT_ORDERS_PLACED", side=side, zones=zones, total_qty=total_qty)
        # 봇이 만든 코어만 추세 추종/TP 관리 (수동은 손대지 않음)
        if state["core"]["active"]:
            pos_long, pos_short = get_hedged_positions() if HEDGE_MODE else (get_net_position(), None)
            core_pos = pos_long if state["core"]["side"]=="long" else pos_short
            if core_pos:
                qty = abs(float(core_pos.get('contracts') or 0.0))
                ep  = float(core_pos.get('entryPrice') or 0.0)
                if qty>0 and ep>0:
                    core_manage_trailing_and_tp(d1, state["core"]["side"], qty, ep)
                    core_detect_and_move_to_breakeven_if_tp1_hit(mark_px, state["core"]["side"], ep)

    # ===== Satellite (단타) — 해지 운용 =====
    if SCALP_ENABLED:
        # One-way 모드면 해지 불가 → 단타 비활성화해 Core와 충돌 방지
        if not HEDGE_MODE:
            log_event(action="SCALP_DISABLED_ONEWAY", reason="hedge off → core와 충돌 위험"); return

        # Core 방향 판단(수동 코어 가정)
        core_dir = core_dir_manual  # "long"/"short"/None
        if SCALP_ONLY_OPPOSITE_CORE and core_dir in ("long","short"):
            scalp_side = "sell" if core_dir=="long" else "buy"
        else:
            # 코어 없으면 자유 — 간단히 4H 기준 방향
            scalp_side = "buy" if h4.iloc[-1]['close'] > h4.iloc[-1]['ema60'] else "sell"

        ok, score = scalp_signal(scalp_side, h1, h15, h4)
        if not ok:
            log_event(action="SCALP_FAIL_DETAIL", scalp_side=scalp_side, score=score); return
        # 쿨다운
        if time.time() - state["last_scalp_entry_ts"] < SCALP_COOLDOWN_SEC:
            return

        sl, tp1, tp2 = calc_scalp_sl_tp(scalp_side, h1, last)
        entry = scalp_place_bracket(scalp_side, SCALP_QTY_COIN, sl, tp1, tp2)
        if entry:
            state["last_scalp_entry_ts"] = time.time()
            log_event(action="SCALP_ENTER", side=scalp_side, qty=SCALP_QTY_COIN, sl=sl, tp1=tp1, tp2=tp2)
            # TP1 체결 후 본절 이동 루틴(간소화)
            scalp_move_be_if_tp1_done()

if __name__ == "__main__":
    main_loop()
