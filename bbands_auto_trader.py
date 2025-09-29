# -*- coding: utf-8 -*-
"""
bbands_auto_trader.py  (FULL)

- 거래소: Bybit Perp (USDT, category=linear)
- 주 전략 프레임: 15m 확정봉
- 모드: trend(수축→확장 추종), range(30m 레인지 하단/상단 매수·매도, 2/3 청산)
- 필터/게이트:
  * 거래량: 15m 확정 이탈/돌파봉 Vol ≥ 최근 5개 평균 × 1.5
  * 손익비(R:R) 게이트: ≥ 1.8 (gross)
  * 밴드 가장자리 금지(예외: 스퀴즈→확장)
  * 컨제스천(OB/FVG/스윙 저·고점 중첩) 회피
  * 유동성 게이트(호가 스프레드/깊이 훅 자리, 미사용시 off)
  * 스타트업 늦진입 방지: 프로그램 시작 이전에 이미 진행 중인 확장 신호 무시
  * 리더(BTC 4h) 강구조 존 근접 필터: BTC의 강구조 존에 "수렴" 방향 진입 차단
- 포지션 운영:
  * 2분할(40/60), TP1 체결 시 SL 본절 이동
  * TP2/TP3는 상위 프레임 확장 동기화(30m/1h 중심선 이탈 전까지 트레일링)
  * 하이브리드: 전략익절 이후 재신호 자동 재진입 가능
- 자금/심볼:
  * 총 운용자금 대비 메인/서브 비중, 레버리지, 동시에 보유할 최대 심볼 수
  * 심볼 리스트 중 이미 포지션 보유 심볼 제외하여 신규 진입
"""

import math, time, datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP


# =========================
# ===== 사용자 설정 ========
# =========================
API_KEY    = "eb2aY5kC775hqNDYO2"
API_SECRET = "lcBdmA34s6yrtB3N1GRpu8Uo8I6NYL6NnQLz"

# 자금/포지션
TOTAL_BALANCE = 5000.0     # 총 운용 USDT (수정하면 즉시 반영)
LEVERAGE      = 10         # 거래소 레버리지(바이비트에서 10배로 설정되어 있어야 함)
MAIN_PORTION  = 0.20       # 메인 시그널 증거금 비중
SUB_PORTION   = 0.12       # 서브(컨플루언스 3~4점) 비중
MAX_POSITIONS = 3          # 동시에 보유 가능한 최대 심볼 수

# 심볼 세팅
SYMBOLS       = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]  # 모니터링 심볼
LEADER_SYMBOL = "BTCUSDT"                          # 리더 차트(필터용)

# 전략 공통
TIMEFRAME_TRADE = "15m"   # 주 전략 프레임
TIMEFRAME_RANGE = "30m"   # 횡보 레인지 판정 프레임
TIMEFRAME_SYNC  = "1h"    # 상위 확장 동기화 프레임
LOOKBACK        = 600     # 캔들 개수

# 볼밴/ATR/거래량
BB_PERIOD  = 20
BB_STD     = 2.0
ATR_LEN    = 14
VOL_LEN    = 5
VOL_MULT   = 1.5    # 최근5평균 × 1.5

# 손익비 및 SL 버퍼
RR_GATE    = 1.8
SL_MIN_PCT = 0.002
SL_ATR_K   = 0.8

# 모드 스위칭(30m BBWidth)
BBWIDTH_THR_RANGE = 0.006  # 0.6% 이하면 range

# 컨제스천(혼잡) 회피
CONGESTION_TOL_PCT   = 0.004
CONGESTION_MIN_HITS  = 2
SWING_LOOKBACK_BARS  = 120

# 분할/비중
SPLIT_2 = (0.4, 0.6)

# 컨플루언스(간단 점수) → 비중 분기
CONF_MAIN = 5
CONF_SUB  = 3

# 스타트업 늦진입 방지: 프로그램 시작 이후 첫 확정 이탈/돌파만 허용
LATE_ENTRY_GUARD = True

# 리더(BTC) 4시간 강구조 존 필터
LEADER_GATE_ON        = True
LEADER_ZONE_TF        = "4h"
LEADER_ZONE_TOL_PCT   = 0.0015  # 0.15% 근접
LEADER_ZONE_MIN_SCORE = 2       # 2개 이상 중첩 시 strong 존

POLL_SEC = 3


# =========================
# ===== Bybit API Layer ===
# =========================
class Bybit:
    def __init__(self, api_key, api_secret):
        self.sess = HTTP(api_key=api_key, api_secret=api_secret)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit:int=600) -> pd.DataFrame:
        tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","4h":"240"}
        res = self.sess.get_kline(category="linear", symbol=symbol, interval=tf_map[timeframe], limit=limit)
        rows = list(reversed(res["result"]["list"]))
        # 열 개수 방어
        if len(rows[0]) == 7:
            df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
        elif len(rows[0]) == 6:
            df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume"])
            df["turnover"] = 0.0
        else:
            df = pd.DataFrame(rows).iloc[:,:7]
            df.columns = ["start","open","high","low","close","volume","turnover"]
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        # timestamp 처리(초/밀리초 자동 판별)
        start = pd.to_numeric(df["start"], errors="coerce")
        unit = "ms" if start.iloc[-1] > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(df["start"].astype(float), unit=unit, utc=True)
        return df[["timestamp","open","high","low","close","volume"]]

    def get_filters(self, symbol: str):
        res = self.sess.get_instruments_info(category="linear", symbol=symbol)
        info = res["result"]["list"][0]
        lot = info["lotSizeFilter"]
        return {
            "qtyStep": float(lot["qtyStep"]),
            "minQty": float(lot["minOrderQty"]),
            "priceStep": float(info["priceFilter"]["tickSize"])
        }

    def positions(self, symbols: List[str]) -> Dict[str, Optional[Dict]]:
        out = {}
        for s in symbols:
            res = self.sess.get_positions(category="linear", symbol=s)
            lst = res["result"]["list"]
            if lst:
                p = lst[0]
                out[s] = {
                    "side": p["side"].lower(),   # "buy"/"sell"
                    "size": float(p["size"]),
                    "entry": float(p["avgPrice"])
                }
            else:
                out[s] = None
        return out

    def place_market(self, symbol: str, side: str, qty: float, reduceOnly: bool=False):
        # side: "buy"/"sell"
        return self.sess.place_order(
            category="linear", symbol=symbol, side=side.capitalize(),
            orderType="Market", qty=str(qty),
            timeInForce="GoodTillCancel", reduceOnly=reduceOnly
        )

    def place_take_profit(self, symbol: str, side: str, qty: float, price: float):
        # TP → 반대 방향 리듀스온리
        side_tp = "Sell" if side.lower()=="buy" else "Buy"
        return self.sess.place_order(
            category="linear", symbol=symbol, side=side_tp,
            orderType="Limit", qty=str(qty), price=str(price),
            reduceOnly=True, timeInForce="GoodTillCancel"
        )

    def place_stop_market(self, symbol: str, side: str, qty: float, stop_price: float):
        # SL → 반대 방향 StopMarket, reduceOnly
        side_sl = "Sell" if side.lower()=="buy" else "Buy"
        return self.sess.place_order(
            category="linear", symbol=symbol, side=side_sl,
            orderType="Market", qty=str(qty),
            reduceOnly=True, timeInForce="GoodTillCancel",
            triggerPrice=str(stop_price), triggerDirection=1,  # pybit v5: TP/SL 트리거 주문은 별도 엔드포인트가 더 안전하지만
        )


# =========================
# ===== 인디케이터/유틸 ====
# =========================
def ema(series: pd.Series, length: int=20) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def bollinger_bands(close: pd.Series, period: int=20, std: float=2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    up = ma + std*sd
    lo = ma - std*sd
    w = (up - lo) / ma
    return ma, up, lo, w

def atr(df: pd.DataFrame, length: int=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(high-low), (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def slope(series: pd.Series, n: int=5) -> float:
    if len(series) < n: return 0.0
    y = series.tail(n).values; x = np.arange(n)
    xm, ym = x.mean(), y.mean()
    num = ((x-xm)*(y-ym)).sum(); den = ((x-xm)**2).sum() or 1.0
    return float(num/den)

def recent_avg_vol(vol: pd.Series, n:int=5) -> float:
    return float(vol.tail(n).mean())

# --- 구조 레벨(간이) ---
def detect_ob_levels(df15: pd.DataFrame) -> List[float]:
    body = (df15["close"] - df15["open"]).abs()
    th = body.tail(200).mean()*1.2
    idx = body[body>th].index
    levels = [float((df15.loc[i,"open"] + df15.loc[i,"close"])/2.0) for i in idx]
    return sorted(list(set(levels)))

def detect_fvg_levels(df15: pd.DataFrame) -> List[Tuple[float,float]]:
    lv = []
    for i in range(2, len(df15)):
        prev_low = df15.iloc[i-2]["low"]; prev_high = df15.iloc[i-2]["high"]
        cur_low  = df15.iloc[i]["low"];   cur_high  = df15.iloc[i]["high"]
        if cur_low > prev_high: lv.append((prev_high, cur_low))   # 상승FVG
        if cur_high < prev_low: lv.append((cur_high, prev_low))   # 하락FVG
    return lv

def detect_swing_points(df: pd.DataFrame, lookback: int=120):
    lows, highs = [], []
    start = max(0, len(df)-lookback)
    for i in range(start+2, len(df)-2):
        win = df.iloc[i-2:i+3]
        if df.iloc[i]["low"]  == win["low"].min():  lows.append(float(df.iloc[i]["low"]))
        if df.iloc[i]["high"] == win["high"].max(): highs.append(float(df.iloc[i]["high"]))
    return lows, highs

def is_congestion_near(price: float, side: str, df15: pd.DataFrame,
                       ob_levels: List[float], fvg_levels: List[Tuple[float,float]],
                       tol_pct: float, min_hits: int, lookback: int) -> bool:
    band_low = price*(1.0 - tol_pct); band_high = price*(1.0 + tol_pct)
    hits = 0
    for lvl in ob_levels:
        if band_low <= lvl <= band_high: hits += 1
    for a,b in fvg_levels:
        c = (a+b)/2.0
        if band_low <= c <= band_high: hits += 1
    lows, highs = detect_swing_points(df15, lookback)
    if side=="long":  hits += sum(1 for x in lows  if band_low <= x <= band_high)
    else:             hits += sum(1 for x in highs if band_low <= x <= band_high)
    return hits >= min_hits

def get_tp1_from_structure(price: float, side: str,
                           ob_levels: List[float], fvg_levels: List[Tuple[float,float]]) -> Optional[float]:
    cands = []
    if side=="long":
        cands += [lvl for lvl in ob_levels if lvl>price]
        cands += [rng[0] for rng in fvg_levels if rng[0]>price]
        return min(cands) if cands else None
    else:
        cands += [lvl for lvl in ob_levels if lvl<price]
        cands += [rng[1] for rng in fvg_levels if rng[1]<price]
        return max(cands) if cands else None

def calc_rr(entry: float, sl: float, tp: float, side: str) -> float:
    if side=="long":
        risk = max(entry - sl, 1e-9); reward = max(tp - entry, 1e-9)
    else:
        risk = max(sl - entry, 1e-9); reward = max(entry - tp, 1e-9)
    return reward / risk

def round_step(x: float, step: float) -> float:
    return math.floor(x/step)*step


# =========================
# ===== 리더(BTC) 존 ======
# =========================
def detect_leader_strong_zones(df4h: pd.DataFrame) -> List[Tuple[float,int]]:
    """
    간이 강구조 존: OB(mid), VPVR proxy(mode), SR(최근 최고/최저 평균) 3종을 중첩 계산.
    점수(겹침수)와 중앙가를 반환.
    """
    close = df4h["close"].astype(float)
    ob = close.rolling(50).mean().iloc[-200:]      # OB proxy
    mode = close.iloc[-400:].round(2).mode()
    vp = float(mode.iloc[0]) if len(mode) else float(close.iloc[-1])
    sr_hi = close.rolling(200).max().iloc[-1]
    sr_lo = close.rolling(200).min().iloc[-1]
    seeds = [float(ob.iloc[-1]), vp, sr_hi, sr_lo]

    # 같은 가격대(±0.15%)에 모이는 개수로 점수화
    zones = []
    for z in seeds:
        score = sum(1 for s in seeds if abs(s-z)/max(z,1e-9) <= LEADER_ZONE_TOL_PCT)
        zones.append((z, score))
    # 중첩 높은 것만 unique로 추림
    uniq=[]
    for z,sc in zones:
        if sc >= LEADER_ZONE_MIN_SCORE:
            if all(abs(z-u)/u > LEADER_ZONE_TOL_PCT for u,_ in uniq):
                uniq.append((z,sc))
    return sorted(uniq, key=lambda x: -x[1])

def is_blocked_by_leader(sym_price: float, side: str, leader_zones: List[Tuple[float,int]]) -> bool:
    """
    BTC 강구조 존과 '수렴' 방향이면 차단:
      - 롱: 현재가 < 존(아래에서 위로 수렴) => 차단
      - 숏: 현재가 > 존(위에서 아래로 수렴) => 차단
    """
    for z,_ in leader_zones:
        if abs(sym_price - z)/z <= LEADER_ZONE_TOL_PCT:
            if (side=="long" and sym_price<z) or (side=="short" and sym_price>z):
                return True
    return False


# =========================
# ===== 엔진 ==============
# =========================
class Engine:
    def __init__(self, api: Bybit):
        self.api = api
        self.state: Dict[str, Dict] = {}  # 심볼별 포지션/주문 상태 메모리
        self.program_start = pd.Timestamp.utcnow()
        self.program_start_bar: Dict[str, pd.Timestamp] = {}  # 심볼별 15m 바 시작 시각 기록

    def _prep_frames(self, symbol: str) -> Dict[str, pd.DataFrame]:
        d15 = self.api.fetch_ohlcv(symbol, TIMEFRAME_TRADE, LOOKBACK)
        d30 = self.api.fetch_ohlcv(symbol, TIMEFRAME_RANGE, LOOKBACK)
        d1h = self.api.fetch_ohlcv(symbol, TIMEFRAME_SYNC,  LOOKBACK)
        for d in (d15, d30, d1h):
            d["ma"], d["bb_up"], d["bb_lo"], d["bb_w"] = bollinger_bands(d["close"], BB_PERIOD, BB_STD)
            d["atr"]   = atr(d.rename(columns={"timestamp":"ts","open":"open","high":"high","low":"low","close":"close"}), ATR_LEN)
            d["ema20"] = ema(d["close"], 20)
        return {"15m":d15, "30m":d30, "1h":d1h}

    def _mode(self, d30: pd.DataFrame) -> str:
        bbw = float(d30["bb_w"].iloc[-1])
        return "range" if bbw <= BBWIDTH_THR_RANGE else "trend"

    def _first_break_flags(self, d15: pd.DataFrame):
        i, prev = -2, -3  # 확정봉
        c  = float(d15["close"].iloc[i])
        o  = float(d15["open"].iloc[i])
        up = float(d15["bb_up"].iloc[i]);   up_prev = float(d15["bb_up"].iloc[prev])
        lo = float(d15["bb_lo"].iloc[i]);   lo_prev = float(d15["bb_lo"].iloc[prev])
        c_prev = float(d15["close"].iloc[prev])

        firstL = (c > up) and (c_prev <= up_prev)
        firstS = (c < lo) and (c_prev >= lo_prev)
        bull = c > o; bear = c < o
        return firstL, firstS, bull, bear, c, up, lo

    def _late_entry_guard_ok(self, symbol: str, d15: pd.DataFrame) -> bool:
        if not LATE_ENTRY_GUARD:
            return True
        # 프로그램 시작 이후의 첫 번째 15m 확정봉부터만 신호 허용
        bar_open = d15["timestamp"].iloc[-2]  # 확정봉의 open 시각
        first = self.program_start_bar.get(symbol)
        if first is None:
            # 프로그램 시작 이후 첫 확정봉 오픈 타임 기록
            self.program_start_bar[symbol] = bar_open
            return False  # 첫 루프에서는 신호 무시 (늦진입 방지)
        # 프로그램 시작 후 새 바가 열렸는지 확인
        return bar_open > first

    def compute_confluence(self, frames: Dict[str,pd.DataFrame]) -> int:
        d15,d30,d1h = frames["15m"], frames["30m"], frames["1h"]
        score=0
        s30 = slope(d30["ema20"], 10); s1h = slope(d1h["ema20"], 10)
        if s30>0 and s1h>0: score+=1
        if s30<0 and s1h<0: score+=1
        e15,e30,e1h = float(d15["ema20"].iloc[-1]), float(d30["ema20"].iloc[-1]), float(d1h["ema20"].iloc[-1])
        if e15>e30>e1h or e15<e30<e1h: score+=1
        bbw_now=float(d30["bb_w"].iloc[-1]); bbw_prev=float(d30["bb_w"].iloc[-6]) if len(d30)>6 else bbw_now
        if bbw_now>bbw_prev: score+=1
        v5=float(d15["volume"].tail(5).mean()); v20=float(d15["volume"].tail(20).mean()) if len(d15)>=20 else v5
        if v5>v20: score+=1
        rng_now=float((d1h["high"].iloc[-1]-d1h["low"].iloc[-1])) if "high" in d1h.columns else 0
        rng_avg=float((d1h["high"]-d1h["low"]).tail(20).mean()) if "high" in d1h.columns else rng_now
        if rng_now>rng_avg: score+=1
        return score

    def build_signal(self, symbol: str, frames: Dict[str,pd.DataFrame]) -> Optional[Dict]:
        d15,d30,d1h = frames["15m"], frames["30m"], frames["1h"]
        mode = self._mode(d30)
        # 스타트업 늦진입 방지
        if not self._late_entry_guard_ok(symbol, d15):
            return None

        # 확정봉 기준
        i, prev = -2, -3
        c=float(d15["close"].iloc[i]); o=float(d15["open"].iloc[i])
        up=float(d15["bb_up"].iloc[i]); lo=float(d15["bb_lo"].iloc[i])
        atr15=float(d15["atr"].iloc[i]); sl_buf=max(SL_MIN_PCT*c, SL_ATR_K*atr15)
        vol_last=float(d15["volume"].iloc[i]); vol_avg5=recent_avg_vol(d15["volume"].iloc[:i+1], VOL_LEN)
        vol_ok = (vol_last >= VOL_MULT*vol_avg5)

        ob_lvls=detect_ob_levels(d15.iloc[:i+1]); fvg_lvls=detect_fvg_levels(d15.iloc[:i+1])

        firstL, firstS, bull, bear, _, _, _ = self._first_break_flags(d15)

        # 컨제스천 회피
        congL = is_congestion_near(c,"long", d15.iloc[:i+1], ob_lvls,fvg_lvls, CONGESTION_TOL_PCT, CONGESTION_MIN_HITS, SWING_LOOKBACK_BARS)
        congS = is_congestion_near(c,"short",d15.iloc[:i+1], ob_lvls,fvg_lvls, CONGESTION_TOL_PCT, CONGESTION_MIN_HITS, SWING_LOOKBACK_BARS)

        # ===== 모드별 =====
        if mode=="trend":
            # 상단 돌파 롱
            if firstL and bull and vol_ok and not congL:
                tp1 = get_tp1_from_structure(c,"long",ob_lvls,fvg_lvls) or (c + 2*atr15)
                sl  = c - sl_buf
                rr  = calc_rr(c, sl, tp1, "long")
                if rr >= RR_GATE:
                    return {"side":"long","entry":c,"sl":sl,"tp1":tp1,"mode":"trend","rr":rr}
            # 하단 이탈 숏
            if firstS and bear and vol_ok and not congS:
                tp1 = get_tp1_from_structure(c,"short",ob_lvls,fvg_lvls) or (c - 2*atr15)
                sl  = c + sl_buf
                rr  = calc_rr(c, sl, tp1, "short")
                if rr >= RR_GATE:
                    return {"side":"short","entry":c,"sl":sl,"tp1":tp1,"mode":"trend","rr":rr}
        else:
            # range: 30m 밴드
            up30=float(d30["bb_up"].iloc[-1]); lo30=float(d30["bb_lo"].iloc[-1])
            two_thirds_up = lo30 + (up30-lo30)*(2/3)
            two_thirds_lo = up30 - (up30-lo30)*(2/3)
            # 하단 매수
            if c <= lo30 and not congL:
                tp1=two_thirds_up; sl=c - sl_buf
                rr=calc_rr(c,sl,tp1,"long")
                if rr >= RR_GATE:
                    return {"side":"long","entry":c,"sl":sl,"tp1":tp1,"mode":"range","rr":rr}
            # 상단 매도
            if c >= up30 and not congS:
                tp1=two_thirds_lo; sl=c + sl_buf
                rr=calc_rr(c,sl,tp1,"short")
                if rr >= RR_GATE:
                    return {"side":"short","entry":c,"sl":sl,"tp1":tp1,"mode":"range","rr":rr}
        return None

    def size_and_round(self, symbol: str, entry: float, sl: float, portion: float) -> Tuple[float, Dict]:
        filt = self.api.get_filters(symbol)
        risk_perc = abs(entry-sl)/entry
        margin = TOTAL_BALANCE * portion
        notional = margin * LEVERAGE
        qty_raw = notional / entry
        qty = max(round_step(qty_raw, filt["qtyStep"]), filt["minQty"])
        return qty, filt

    def manage_trailing(self, symbol: str, frames: Dict[str,pd.DataFrame]):
        """간단 관리: 30m/1h 확장 동기화에 맞춘 SL 상향/하향"""
        st = self.state.get(symbol); 
        if not st or st.get("status")!="open": return
        d30,d1h = frames["30m"], frames["1h"]
        side = st["side"]; price_now = frames["15m"]["close"].iloc[-1]
        changed=False
        # TP1 체결 체크
        if not st.get("tp1_done"):
            if (side=="long" and price_now>=st["tp1"]) or (side=="short" and price_now<=st["tp1"]):
                st["tp1_done"]=True
                # 본절 이동
                st["sl_live"]=st["entry"]
                # SL 주문 갱신(간단: 기존 SL 취소 후 신규 — 여기선 생략, 실전은 cancel→재발주 필요)
                changed=True
        # 확장 동기화 트레일링
        if side=="long":
            if price_now >= float(d30["bb_up"].iloc[-1]):
                st["sl_live"]=max(st["sl_live"], float(d30["ema20"].iloc[-1] or st["sl_live"]))
            if price_now >= float(d1h["bb_up"].iloc[-1]):
                st["sl_live"]=max(st["sl_live"], float(d1h["ema20"].iloc[-1] or st["sl_live"]))
        else:
            if price_now <= float(d30["bb_lo"].iloc[-1]):
                st["sl_live"]=min(st["sl_live"], float(d30["ema20"].iloc[-1] or st["sl_live"]))
            if price_now <= float(d1h["bb_lo"].iloc[-1]):
                st["sl_live"]=min(st["sl_live"], float(d1h["ema20"].iloc[-1] or st["sl_live"]))
        if changed:
            print(f"[MANAGE] {symbol} TP1 done → move SL to BE={st['entry']}")
        self.state[symbol]=st

    def loop(self):
        print(f"[{pd.Timestamp.utcnow()}] START symbols={SYMBOLS}, max={MAX_POSITIONS}")

        # 리더(BTC) 강구조 존 미리 계산
        leader_zones=[]
        if LEADER_GATE_ON:
            df_leader = self.api.fetch_ohlcv(LEADER_SYMBOL, LEADER_ZONE_TF, LOOKBACK)
            zones = detect_leader_strong_zones(df_leader)
            leader_zones = zones
            print(f"[DEBUG] leader strong zones({LEADER_ZONE_TF}): {[z for z,_ in zones]}")

        while True:
            try:
                # 현재 보유 포지션 확인
                live_pos = self.api.positions(SYMBOLS)
                holding_syms = [s for s,p in live_pos.items() if p and p["size"]>0]
                # 진입 가능한 슬롯
                slots = max(0, MAX_POSITIONS - len(holding_syms))

                enter_results = {}
                manage_results = {}

                # 관리 루프 (보유 심볼)
                for sym in holding_syms:
                    frames = self._prep_frames(sym)
                    self.manage_trailing(sym, frames)
                    manage_results[sym] = {"status":"holding", "side": live_pos[sym]["side"]}

                # 신규 진입 탐색
                for sym in SYMBOLS:
                    if slots<=0: break
                    if sym in holding_syms: 
                        enter_results[sym] = {"status":"skip_holding"}
                        continue

                    frames = self._prep_frames(sym)
                    sig = self.build_signal(sym, frames)
                    if not sig:
                        enter_results[sym] = {"status":"no_signal"}
                        continue

                    # 리더(BTC) 필터: strong zone 수렴 차단
                    price_now = float(frames["15m"]["close"].iloc[-1])
                    if LEADER_GATE_ON and is_blocked_by_leader(price_now, sig["side"], leader_zones):
                        print(f"[BLOCK] {sym} {sig['side']} near BTC zone → blocked")
                        enter_results[sym] = {"status":"blocked_leader"}
                        continue

                    # 컨플루언스 점수 → 비중 결정
                    conf = self.compute_confluence(frames)
                    portion = MAIN_PORTION if conf>=CONF_MAIN else (SUB_PORTION if conf>=CONF_SUB else 0.0)
                    if portion<=0:
                        enter_results[sym] = {"status":"no_conf", "conf":conf}
                        continue

                    qty, filt = self.size_and_round(sym, sig["entry"], sig["sl"], portion)
                    if qty < filt["minQty"]:
                        enter_results[sym] = {"status":"qty_too_small"}
                        continue

                    # 실 주문 (시장진입 + TP1 + SL 각각)
                    side_mkt = "buy" if sig["side"]=="long" else "sell"
                    self.api.place_market(sym, side_mkt, qty, reduceOnly=False)

                    # 분할 비중
                    qty1 = round_step(qty*SPLIT_2[0], filt["qtyStep"])
                    qty2 = max(qty - qty1, filt["minQty"])

                    # TP1: 전체 중 qty1 우선
                    self.api.place_take_profit(sym, side_mkt, qty1, sig["tp1"])
                    # SL: 전체 수량에 대해 StopMarket(표시 보장)
                    self.api.place_stop_market(sym, side_mkt, qty, sig["sl"])

                    # 상태 기록
                    self.state[sym] = {
                        "status":"open", "side":sig["side"], "entry":sig["entry"],
                        "tp1":sig["tp1"], "sl_init":sig["sl"], "sl_live":sig["sl"],
                        "tp1_done":False, "ported_qty":qty, "conf":conf, "mode":sig["mode"]
                    }
                    enter_results[sym] = {"status":"entered", "tier":"main" if portion==MAIN_PORTION else "sub", "rr":sig["rr"]}
                    slots -= 1

                print("ENTER:", enter_results)
                print("MANAGE:", manage_results)
                time.sleep(POLL_SEC)

            except Exception as e:
                print("[ERROR]", e)
                time.sleep(POLL_SEC)


# =========================
# ===== 실행 ==============
# =========================
def main():
    api = Bybit(API_KEY, API_SECRET)
    eng = Engine(api)
    eng.loop()

if __name__ == "__main__":
    main()
