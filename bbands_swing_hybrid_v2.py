# -*- coding: utf-8 -*-
"""
bbands_swing_hybrid_full_vNEXT_PLUS.py

Bybit V5 — 4H 스윙 + 15m 볼린저 하이브리드 (개선 통합본)

필수 안정성 패치 반영:
- 주문 성공判定: resp.get("retCode")==0 (DRYRUN은 특별 처리)
- 쿨다운 bar_id 길이 체크(확정봉 iloc[-2] 접근 전 len(df)≥2 필수)
- 예산 캡 vs 최소수량 상충: quantize_qty(raw, step, minQty) (raw<minQty→0)
- VIRTUAL_PAPER에서도 동일봉 쿨다운 등록(register_exec)
- 손실 틸트 연동: EXIT 시 PnL<0 → record_loss()
- API 호출 절감: 메인 빌드에서 받은 d1h 재사용 (중복 fetch 제거)
- 로깅 수치 안전 캐스팅
- 변수 그림자 제거

서브(15m) 전략 보강:
- 진입 확인: 2연속 밴드 이탈(옵션) + 강봉/강거래 1봉 예외 허용
- SL 모드 추가: "BAND_RECOVERY" (밴드 재진입 시 컷) + 최소 SL 거리 보장
- TP1/TP2: TP1 고정(부분익절 후 BE), TP2 힌트값 로깅 + 트레일링 운영(ATR 또는 N봉 극값)
- TP1/TP2/SL 라인 로깅 강화(차트 오버레이용)

주의: 실제 배포 전 TESTNET/DRY_RUN로 충분히 검증하세요.
"""

import os, csv, math, time, zipfile, traceback, uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ===========================
# ========= CONFIG ==========
# ===========================

API_KEY    = "6D7H5kEMR5uGOlzlzA"   # <-- 본인 키
API_SECRET = "lT6XEk2Mn1xj2vcZIMS5LGxYlviQgbiuXY7p"   # <-- 본인 시크릿
TESTNET    = False
BASE_URL   = "https://api.bybit.com"

LEADER   = "BTCUSDT"
SYMBOLS  = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]

MAX_MAIN_POS = 1
MAX_SUB_POS  = 1

STARTUP_RECONCILE = False
RUNTIME_REVERSE   = True

# ===== 심볼별 기본 수량 =====
MAIN_FIXED_QTY = {"ETHUSDT": 0.7, "SOLUSDT": 2.0, "BNBUSDT": 0.8, "DEFAULT": 0.5}
SUB_FIXED_QTY  = {"ETHUSDT": 0.4, "SOLUSDT": 1.2, "BNBUSDT": 0.6, "DEFAULT": 0.4}

# ===== (옵션) 명목 예산 캡 =====
USE_NOTIONAL_BUDGET = True
PER_SYMBOL_BUDGET_USDT = {"ETHUSDT": 20000.0, "SOLUSDT": 10000.0, "BNBUSDT": 20000.0, "DEFAULT": 15000.0}

# ===== 컨플루언스 가중 =====
CONF_WEIGHT_STEP_MAIN = 0.5
CONF_MAX_WEIGHT_MAIN  = 2.0
CONF_WEIGHT_STEP_SUB  = 0.5
CONF_MAX_WEIGHT_SUB   = 2.0

# ===== 프리셋(균형형 기본) =====
PRESET = "Balanced"  # "Conservative" | "Balanced" | "Active"

# ===== 메인(4H) =====
MAIN_EMA_LEN         = 200
MAIN_RR_GATE         = 1.8
MAIN_ATR_LEN_4H      = 14
MAIN_SL_ATR_K        = 0.5
MAIN_TP_MODE         = "tp1_be"     # "tp1_be" | "stair"
MAIN_TP1_RATIO       = 0.4

# EMA200 가드 파라미터
EMA_NEUTRAL_PCT        = 0.002
EMA_NEUTRAL_ATR_K      = 0.3
SWITCH_UP_PCT          = 0.0015
SWITCH_DOWN_PCT        = 0.0025
EMA_SLOPE_MIN          = 0.0
EMA_CROSS_CONFIRM_BARS = 1

# ===== 메인(4H) 확장 초입 옵션 =====
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

# ==== 30m 확장-횡보 Range 전략 게이트 ====
RANGE_EXPANDED_BBW_MIN   = 0.020   # 30m BBW가 이 값 이상이면 "확장"으로 간주
SIDEWAYS_SLOPE_ABS_MAX   = 0.0004  # 30m EMA20 기울기 절대값이 이 이하이면 "횡보"
SIDEWAYS_LOOKBACK_EMA    = 6       # EMA 기울기 확인 간격(봉수)
SIDEWAYS_PIVOT_LOOK      = 24      # pivot_trend 중립(0) 확인용 lookback

# Range 진입/청산 파라미터
RANGE_TOUCH_EPS          = 0.0005  # 밴드 터치 허용 오차(0.05%)
RANGE_SL_PAD_PCT         = 0.0008  # SL를 밴드 바깥으로 약간 여유(0.08%)
MIN_RR_RANGE30           = 1.20    # Range 시 최소 RR


# 거래량 멀티(동적 완화)
VOL_MULT_SUB_MIN = 1.05
VOL_MULT_SUB_MAX = 1.20

SL_MIN_PCT       = 0.0025
TP1_RATIO_SUB    = 0.40

# ▶ 수축/확장 판단
SQUEEZE_LOOKBACK = 40
SQUEEZE_K        = 0.7
EXPAND_MULT      = 1.25

# ▶ 베토(일반 확장 구간) — ATR 기반
LTF_BAND_VETO_ATR_K = 0.20  # Balanced (보수:0.25, 공격:0.18)

# ▶ 분할 진입
SPLIT_ENTRY_ENABLE    = True
SPLIT_ENTRY_FIRST_PCT = 0.5
RETEST_ENABLE         = True
RETEST_TIMEOUT_BARS   = 8
RETEST_MAX_DRIFT_PCT  = 0.004
RETEST_TOUCH_MODE     = "band_or_mid"  # "band" | "mid" | "band_or_mid"
RETEST_TOUCH_EPS      = 0.0020

# ▶ 손절 모드 (서브)
# "ATR" | "MIDLINE" | "BAND_RECOVERY"
SUB_STOP_MODE     = "BAND_RECOVERY"
STOP_ATR_K        = 0.5
MID_BUF_PCT       = 0.0015
# BAND 복귀형 SL 파라미터
BAND_RECOVERY_EPS_PCT = 0.0008    # 밴드선 대비 여유
SUB_MIN_STOP_DIST_PCT = 0.0012    # 엔트리 대비 최소 SL 거리 보장(12bp)
SUB_SL_CHOICE     = "NEAR"        # MIDLINE 모드에서만 사용 ("FAR" | "NEAR")

# ▶ 30m 레인지
RANGE_BBWIDTH_THR = 0.007
RANGE_TGT_RATIO   = 2.0/3.0
FORBID_MID_RANGE  = True

# ▶ 상단 롱 예외(대칭)
ALLOW_LONG_AT_UPPER_BAND = True
LBU_REQUIRE_TREND_30M_UP = True
LBU_VOL_MULT              = 1.25
LBU_MAX_PULLIN_PCT        = 0.0020

# ▶ 15m Early Expansion(초입) 보조 트리거
USE_15M_EE_TRIGGER        = True
EE15M_BBW_EXPAND_K        = 0.9
EE15M_MIN_RR              = 2.0

# ▶ 하단 숏 예외(기존)
ALLOW_SHORT_AT_LOWER_BAND   = True
SLB_REQUIRE_TREND_30M_DOWN  = True
SLB_VOL_MULT                = 1.25
SLB_MAX_BOUNCE_PCT          = 0.0015

# ▶ 최소 변동성 바닥
ATR_15M_MIN  = 0.0   # 필요시 세팅(예: 심볼가×0.0006)
BBW_15M_MIN  = 0.0

# ▶ 진입 확인(첫봉/2연속/강봉예외)
SUB_ENTRY_CONFIRM_MODE   = "FIRST_OR_2"   # "FIRST_ONLY" | "TWO_ONLY" | "FIRST_OR_2"
STRONG_BAR_BODY_PCT      = 0.60           # (|close-open|)/(high-low)
STRONG_BAR_VOL_MULT      = 1.50           # 1.5배 이상 거래량이면 강봉 예외

# ▶ TP2 트레일링
SUB_TP2_TRAIL_MODE       = "ATR"          # "ATR" | "SWING"
SUB_TP2_TRAIL_ATR_K      = 0.8
SUB_TP2_TRAIL_LOOKBACK   = 5              # SWING 모드: 최근 N봉 저/고

# ▶ 쿨다운/재진입
TILT_MAX_LOSSES_DIR = 2   # 같은 방향 연속 손절 N회 → 데드타임
TILT_DEAD_BARS      = 6

# ===== 리더 필터(옵션 유지) =====
BTC_STRONG_ZONES = [108500, 109000, 111200]
BTC_STRONG_TOL   = 50
BTC_FILTER_MODE  = "off"
BTC_SOFT_ENABLE  = False
SOFT_RR_GATE     = 2.0
SOFT_SIZE_FACTOR = 0.5
SOFT_SL_BOOST    = 1.10

# ===== 재이탈 허용 윈도(15m 봉) =====
REBREAK_DEADLINE_BARS = 8

# ===== 추격 진입 =====
CHASE_ENABLE         = True
CHASE_SIZE_RATIO     = 0.5
CHASE_MAX_DRIFT_PCT  = 0.006

# ===== 루프/로깅/테스트 =====
VIRTUAL_PAPER = False
DRY_RUN       = False
POLL_SEC      = 5
HEARTBEAT_EVERY = 6

LOG_TO_CSV       = True
LOG_DIR          = "logs"
LOG_FILE         = f"{LOG_DIR}/trade_log.csv"
LOG_SKIP_REASONS = True
SKIP_LOG_FILE    = f"{LOG_DIR}/skip_log.csv"
LOG_RETENTION_DAYS = 7
LOG_MAX_BYTES      = 5_000_000

print(f"[CONFIG] PRESET={PRESET}  MAIN_FIXED_QTY={MAIN_FIXED_QTY}  SUB_FIXED_QTY={SUB_FIXED_QTY}")

LOOPS_PER_15M = max(1, int((15*60)//POLL_SEC))

# 동일봉 쿨다운(엔진×심볼×방향) & 재진입 틸트 제어
LAST_EXEC_BAR: Dict[Tuple[str,str,str], int] = {}
LAST_LOSS_DIR: Dict[Tuple[str,str], Dict[str,int]] = {}  # (engine,symbol)->{"dir": "long"/"short","losses":n,"dead":bars}

# ===========================
# ===== 로깅/회전 유틸 ======
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
        "event": event,
        "symbol": kw.get("symbol"),
        "engine": kw.get("engine"),
        "side": kw.get("side"),
        "qty": kw.get("qty"),
        "entry": kw.get("entry"),
        "sl": kw.get("sl"),
        "tp1": kw.get("tp1"),
        "tp2": kw.get("tp2"),
        "rr":  kw.get("rr"),
        "mode": kw.get("mode"),
        "extra": kw.get("extra"),
        "exit_price": kw.get("exit_price"),
        "bbw": kw.get("bbw"),
        "came_from_squeeze": kw.get("came_from_squeeze"),
        "range_mode": kw.get("range_mode"),
        "rebreak_ok": kw.get("rebreak_ok"),
        "ee_flag": kw.get("ee_flag"),
    }
    _csv_write(LOG_FILE, header, row)

def log_skip_reason(symbol: str, reason: str, extra: str=""):
    if not LOG_SKIP_REASONS: return
    ensure_log_dir()
    header = ["ts_utc","symbol","reason","extra"]
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol,
        "reason": reason,
        "extra": extra
    }
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
    for _ in range(max_try):
        try:
            r = func()
            return r
        except Exception as e:
            print("[WARN retry]", e)
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
                if df.shape[1] < 6:
                    raise ValueError(f"kline columns <6 (got {df.shape[1]})")
                std_cols = ["start","open","high","low","close","volume","turnover"]
                use_n = min(df.shape[1], len(std_cols))
                df = df.iloc[:, :use_n].copy()
                df.columns = std_cols[:use_n]
                for c in ["open","high","low","close","volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                start = pd.to_numeric(df["start"], errors="coerce")
                unit = "ms" if start.iloc[-1] > 1e12 else "s"
                df["timestamp"] = pd.to_datetime(start, unit=unit, utc=True)
                if "high" not in df.columns: df["high"] = df["close"]
                if "low"  not in df.columns: df["low"]  = df["close"]
                use_cols = [c for c in ["timestamp","open","high","low","close","volume"] if c in df.columns]
                return df[use_cols]
            except Exception as e:
                print(f"[WARN fetch_ohlcv] {symbol} {timeframe} attempt={attempt}/{retries} err={e}")
                if attempt == retries:
                    print(f"[ERROR fetch_ohlcv] {symbol} {timeframe} giving up.")
                    return pd.DataFrame()
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
        # V5: timeInForce="GTC"
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
            print("[ERROR set_stop_loss_mark]", symbol, stop_price, e)
            return {"status":"error","error":str(e)}

# ===========================
# ===== Indicators/Helpers ==
# ===========================
def is_sideways_expanded_30m(d30: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """
    30m가 '확장된 횡보'인지 판정:
      - 확장: BBW >= RANGE_EXPANDED_BBW_MIN
      - 횡보: EMA20 기울기 거의 0 (abs(slope) <= SIDEWAYS_SLOPE_ABS_MAX)
      - 추세 배제: pivot_trend(d30, look=SIDEWAYS_PIVOT_LOOK) == 0
      - 스퀴즈 배제: BBW <= RANGE_BBWIDTH_THR 는 False 여야 함
    """
    if d30 is None or len(d30) < max(25, SIDEWAYS_LOOKBACK_EMA+1): 
        return False, {"bbw":0.0,"slope":0.0,"pivot":0}

    bbw_now = float(d30.get("bb_w", pd.Series([0])).iloc[-1] or 0.0)
    # EMA20 기울기
    ema20 = ema(d30["close"], 20)
    denom = max(abs(float(ema20.iloc[-SIDEWAYS_LOOKBACK_EMA])), 1e-9)
    slope = float((ema20.iloc[-1] - ema20.iloc[-SIDEWAYS_LOOKBACK_EMA]) / denom)

    # 피벗 트렌드(상/하 추세 배제)
    pv = pivot_trend(d30, look=SIDEWAYS_PIVOT_LOOK)

    # 스퀴즈(축소) 배제: 기존 임계로 배제
    is_squeeze = bbw_now <= RANGE_BBWIDTH_THR

    ok = (bbw_now >= RANGE_EXPANDED_BBW_MIN) and (abs(slope) <= SIDEWAYS_SLOPE_ABS_MAX) and (pv == 0) and (not is_squeeze)
    return ok, {"bbw":bbw_now, "slope":slope, "pivot":pv}


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

def round_qty(qty: float, step: float, min_qty: float) -> float:
    if step <= 0: return qty
    q = math.floor(qty / step) * step
    return max(q, min_qty)

def quantize_qty(raw: float, step: float, min_qty: float) -> float:
    # 예산<minQty 충돌 방지: minQty 미만이면 0
    if raw < min_qty: return 0.0
    if step <= 0: return raw
    q = math.floor(raw / step) * step
    return q if q >= min_qty else 0.0

def round_price(px: float, tick: float) -> float:
    if tick <= 0: return px
    n = round(px / tick)
    return round(n * tick, 8)

def rr(entry: float, sl: float, tp: float, side: str) -> float:
    if side == "long":
        risk = max(entry - sl, 1e-9); reward = max(tp - entry, 1e-9)
    else:
        risk = max(sl - entry, 1e-9); reward = max(entry - tp, 1e-9)
    return reward / risk

# 간단 OB/FVG + 간이 VPVR
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

def simple_vpvr_poc(df: pd.DataFrame, bins: int=50) -> Optional[float]:
    px = ((df["high"] + df["low"]) / 2.0).values
    vol = df["volume"].values
    if len(px) < 10: return None
    hist, edges = np.histogram(px, bins=bins, weights=vol)
    idx = int(np.argmax(hist))
    poc = (edges[idx] + edges[idx+1]) / 2.0
    return float(poc)

def is_btc_strong_near(leader_px: float) -> bool:
    return any(abs(leader_px - z) <= BTC_STRONG_TOL for z in BTC_STRONG_ZONES)

def passes_ema_guard(side: str, d1h: pd.DataFrame, d4: Optional[pd.DataFrame]=None) -> Tuple[bool, Dict]:
    if d1h is None or len(d1h) < 10:
        return True, {"dist_pct":0.0,"slope":0.0,"neutral_band":0.0}
    d1h = d1h.copy()
    d1h["ema200"] = ema(d1h["close"], 200)
    price = float(d1h["close"].iloc[-1]); ema200 = float(d1h["ema200"].iloc[-1])
    if ema200 == 0 or price == 0:
        return True, {"dist_pct":0.0,"slope":0.0,"neutral_band":0.0}
    dist_pct = (price - ema200) / ema200
    atr4 = 0.0
    if d4 is not None and len(d4)>=50:
        try: atr4 = float(atr(d4, 14).iloc[-1] or 0)
        except: atr4 = 0.0
    slope = float((d1h["ema200"].iloc[-1] - d1h["ema200"].iloc[-5]) / max(abs(d1h["ema200"].iloc[-5]),1e-9))
    neutral_band = max(EMA_NEUTRAL_PCT, (atr4/price if price else 0)*EMA_NEUTRAL_ATR_K)
    if abs(dist_pct) <= neutral_band and slope >= EMA_SLOPE_MIN:
        return True, {"dist_pct":dist_pct,"slope":slope,"neutral_band":neutral_band}
    if side=="long":
        ok = (dist_pct >= SWITCH_UP_PCT) and (slope >= EMA_SLOPE_MIN)
    else:
        ok = (dist_pct <= -SWITCH_DOWN_PCT) and (slope >= EMA_SLOPE_MIN)
    return ok, {"dist_pct":dist_pct,"slope":slope,"neutral_band":neutral_band}

# PATCH: pivot 동시 참 케이스 타이브레이크
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
        if abs(price - lv) <= tol:
            add += 1; notes.append(f"OB≈{lv:.0f}"); break
    for lo, hi in fvg:
        if (lo - tol) <= price <= (hi + tol):
            add += 1; notes.append("FVG zone"); break
    if vpoc and abs(price - vpoc) <= tol:
        add += 1; notes.append("VPOC zone")
    return add, notes

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PATCH START (BBEngine 전체 교체: BTC 15m 동조 "필수") >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ===== BTC 15m 동조 "필수" 옵션 (서브 전용) =====
BTC_SYNC_REQUIRED  = True     # 켜면 BTC 동조 불충족 시 서브 진입 금지
BTC_SYNC_EPS_PCT   = 0.0008   # 밴드선 접촉 여유 (0.08%)

class BBEngine:
    def __init__(self, api: Bybit):
        self.api = api
        self.state: Dict[str, Dict] = {}
        self.rebreak: Dict[str, Dict] = {}

    # === 심볼 프레임 ===
    def _frames(self, sym: str):
        d15 = self.api.fetch_ohlcv(sym,"15m",300)
        d30 = self.api.fetch_ohlcv(sym,"30m",300)
        d1h = self.api.fetch_ohlcv(sym,"1h",300)
        d4  = self.api.fetch_ohlcv(sym,"4h",240)  # 컨플루언스 가산용
        if d15.empty or d30.empty or d1h.empty or d4.empty: return None
        d15 = bollinger(d15.copy(), BB_PERIOD, BB_STD)
        d30 = bollinger(d30.copy(), BB_PERIOD, BB_STD)
        d1h["atr"] = atr(d1h.copy(), ATR_LEN_1H)
        d30["ema20"] = ema(d30["close"], 20)
        return (d15, d30, d1h, d4)

    # === BTC 15분 프레임(볼밴 포함) ===
    def _btc_frames_15m(self) -> Optional[pd.DataFrame]:
        """BTC 15m 프레임 + 볼밴"""
        try:
            d15_btc = self.api.fetch_ohlcv(LEADER, "15m", 200)
            if d15_btc.empty or len(d15_btc) < 30:
                return None
            d15_btc = bollinger(d15_btc.copy(), BB_PERIOD, BB_STD)
            return d15_btc
        except Exception:
            return None

    # === BTC 동조 확인 ===
    def _btc_in_sync(self, side: str, d15_btc: Optional[pd.DataFrame]) -> bool:
        """BTC 15m이 같은 방향으로 '무너짐/돌파' 확정인지 판정(최근 확정봉 기준)"""
        if d15_btc is None or len(d15_btc) < 2:
            return False
        b  = d15_btc.iloc[-2]   # 최근 확정봉
        c  = float(b["close"])
        up = float(b["bb_up"]); lo = float(b["bb_lo"])
        eps = BTC_SYNC_EPS_PCT
        if side == "long"  and (c >= up * (1.0 - eps)): return True   # 상단 돌파/근접 확정
        if side == "short" and (c <= lo * (1.0 + eps)): return True   # 하단 이탈/근접 확정
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

    # ▼ Early Expansion(초입) — 15m 보조 트리거
    def _early_expansion_15m(self, d15: pd.DataFrame, d30: pd.DataFrame, d1h: pd.DataFrame) -> Optional[Dict]:
        if not USE_15M_EE_TRIGGER or len(d15)<40 or len(d30)<25: return None
        j = -2
        bbw = d15["bb_w"]
        avg = float(bbw.iloc[-20:].mean() or 0)
        bbw_prev = float(bbw.iloc[-3] or 0)
        bbw_now  = float(bbw.iloc[-2] or 0)
        if not (bbw_prev <= avg*SQUEEZE_K and bbw_now >= avg*EE15M_BBW_EXPAND_K):
            return None
        prev = d15.iloc[j]
        price = float(prev["close"]); up=float(prev["bb_up"]); lo=float(prev["bb_lo"])
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
            risk = max(price-sl, 1e-9); tp1 = price + 2.0*risk
        else:
            sl = max(float(prev["high"]), price + 0.5*atr15)
            risk = max(sl-price, 1e-9); tp1 = price - 2.0*risk
        R = rr(price, sl, tp1, side)
        if R < EE15M_MIN_RR: return None
        tp2 = (price + 3.0*risk) if side=="long" else (price - 3.0*risk)
        return {"side":side,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":R,"ee_flag":True,
                "came_from_squeeze":True,"range_mode":self.mode_range(d30), "bbw":float(d15["bb_w"].iloc[-2])}

    def _polarity_veto(self, price: float, up: float, lo: float, side: str, atr1h: float) -> bool:
        tol = max(price * 0.0005, atr1h * LTF_BAND_VETO_ATR_K)
        if side=="long"  and (price - up) <= tol:  return True
        if side=="short" and (lo - price) <= tol:  return True
        return False

    def _long_upper_band_exception(self, d30: pd.DataFrame, price: float, up: float, atr1h: float, prev_vol: float, vol_avg5: float) -> bool:
        if not ALLOW_LONG_AT_UPPER_BAND: return False
        ema20_30 = d30["ema20"]
        slope_ok = len(ema20_30) >= 6 and (ema20_30.iloc[-1] - ema20_30.iloc[-5]) / max(abs(ema20_30.iloc[-5]),1e-9) > 0
        if LBU_REQUIRE_TREND_30M_UP and not slope_ok: return False
        atr_tol = max(price*0.0005, atr1h * LTF_BAND_VETO_ATR_K)
        near_up = (price - up) <= max(atr_tol, up * LBU_MAX_PULLIN_PCT)
        vol_ok  = prev_vol >= (LBU_VOL_MULT * max(vol_avg5,1e-9))
        return near_up and vol_ok

    def _short_lower_band_exception(self, d15: pd.DataFrame, d30: pd.DataFrame, price: float, lo: float, atr1h: float,
                                    prev_vol: float, vol_avg5: float) -> bool:
        if not ALLOW_SHORT_AT_LOWER_BAND: return False
        atr_tol = max(price * 0.0005, atr1h * LTF_BAND_VETO_ATR_K)
        near_lo = (lo - price) <= max(atr_tol, lo * SLB_MAX_BOUNCE_PCT)
        slope_ok = True
        if SLB_REQUIRE_TREND_30M_DOWN:
            ema20_30 = d30["ema20"]
            slope_ok = len(ema20_30) >= 6 and ((ema20_30.iloc[-1] - ema20_30.iloc[-5]) / max(abs(ema20_30.iloc[-5]), 1e-9) < 0)
        vol_ok = prev_vol >= (SLB_VOL_MULT * max(vol_avg5, 1e-9))
        return near_lo and slope_ok and vol_ok

    def _entry_confirmation_ok(self, d15: pd.DataFrame, side: str) -> bool:
        if len(d15) < 3: return False
        b1 = d15.iloc[-3]; b2 = d15.iloc[-2]
        if side=="long":
            c1 = float(b1["close"]) > float(b1["bb_up"])
            c2 = float(b2["close"]) > float(b2["bb_up"])
        else:
            c1 = float(b1["close"]) < float(b1["bb_lo"])
            c2 = float(b2["close"]) < float(b2["bb_lo"])
        if SUB_ENTRY_CONFIRM_MODE == "FIRST_ONLY":
            return c2
        if SUB_ENTRY_CONFIRM_MODE == "TWO_ONLY":
            return (c1 and c2)
        # FIRST_OR_2: 2연속 or 강봉·강거래 예외
        if c1 and c2:
            return True
        rng = max(float(b2["high"]) - float(b2["low"]), 1e-9)
        body = abs(float(b2["close"]) - float(b2["open"])) / rng
        vol_avg5 = d15["volume"].iloc[-7:-2].mean()
        strong = (body >= STRONG_BAR_BODY_PCT) and (float(b2["volume"]) >= STRONG_BAR_VOL_MULT * max(vol_avg5,1e-9))
        return c2 and strong

    # === 신호 생성 (BTC 동조 "필수") ===
    def build_signal(self, sym: str, leader_px: float, main_active: bool) -> Optional[Dict]:
        pack = self._frames(sym)
        if pack is None:
            print(f"[DEBUG] {sym} no data"); return None
        d15, d30, d1h, d4 = pack
        if len(d15) < 50:
            log_skip_reason(sym, "insufficient_bars"); return None

        # BTC 15m 프레임(볼밴) 준비 — 한 번만 호출
        d15_btc = self._btc_frames_15m()

        # 1) 15m Early Expansion 보조 트리거(우선)
        ee = self._early_expansion_15m(d15, d30, d1h)
        if ee:
            ee["symbol"]=sym; ee["range_mode"]=self.mode_range(d30)
            add_conf, _ = _near_confluence_4h(ee["entry"], d4, tol_k=0.6)
            ee["conf_count"] = add_conf + 1  # 기본 1 가점

            # 🔴 BTC 동조 "필수"
            if BTC_SYNC_REQUIRED and not self._btc_in_sync(ee["side"], d15_btc):
                log_skip_reason(sym, "btc_sync_fail_ee", f"side={ee['side']}")
                ee = None
            if ee:
                return ee

        # 2) 일반 신호 생성 (기존 로직)
        j = -2
        prev = d15.iloc[j]
        price = float(prev["close"])
        up, lo = float(prev["bb_up"]), float(prev["bb_lo"])
        mid = float(prev["bb_ma"])
        bbw_now = float(d15["bb_w"].iloc[j] or 0)

             # === (추가) 30m 확장-횡보일 때만 Range 되돌림 전략 가동 ===
        ok_sideways, sdiag = is_sideways_expanded_30m(d30)
        if ok_sideways:
            # 밴드 터치/근접 시 기계적 되돌림: 하단≈롱 / 상단≈숏
            band_span = max(up - lo, 1e-9)

            side = None
            # 하단 근접 → 롱
            if price <= lo * (1.0 + RANGE_TOUCH_EPS):
                side = "long"
                # SL: 하단 밴드 바깥으로 소폭
                sl  = lo * (1.0 - RANGE_SL_PAD_PCT)
                # TP1: 중앙선, TP2: 하단→상단 75% 지점
                tp1 = mid
                tp2 = lo + 0.75 * band_span

            # 상단 근접 → 숏
            elif price >= up * (1.0 - RANGE_TOUCH_EPS):
                side = "short"
                sl  = up * (1.0 + RANGE_SL_PAD_PCT)
                tp1 = mid
                tp2 = up - 0.75 * band_span

            if side is not None:
                R = rr(price, sl, tp1, side)
                if R >= MIN_RR_RANGE30:
                    # 컨플루언스는 낮게(횡보 전략이므로 과한 가중 금지)
                    conf_count = 0
                    return {
                        "symbol": sym, "side": side, "entry": price, "sl": sl,
                        "tp1": tp1, "tp2": tp2, "rr": R,
                        "size_factor": 1.0,  # 필요 시 별도 가중
                        "range_mode": True, "mid": mid, "up": up, "lo": lo,
                        "came_from_squeeze": False, "retest_px": None,
                        "conf_count": conf_count,
                        "bbw": float(bbw_now),
                        "rebreak_ok": False, "ee_flag": False
                    }
                else:
                    log_skip_reason(sym, "range30_rr_low", f"R={R:.2f}")
            else:
                log_skip_reason(sym, "range30_no_touch",
                                f"price={price:.4f}, up={up:.4f}, lo={lo:.4f}, bbw={sdiag['bbw']:.4f}")
        else:
            # 확장-횡보가 아니면 Range 전략은 완전히 비활성 (추세/스퀴즈 구간 진입 금지)
            pass


        # 거래량 멀티(동적): 수축구간 완화/확장구간 강화
        if bbw_now < 0.01:   mult = max(1.05, VOL_MULT_SUB_MIN)
        elif bbw_now < 0.02: mult = 1.10
        else:                mult = min(1.20, VOL_MULT_SUB_MAX)
        vol_window = d15["volume"].iloc[j-4:j+1]
        vol_avg5 = float(vol_window.mean()) if len(vol_window)==5 else float(d15["volume"].iloc[:j+1].tail(5).mean())
        vol_ok = float(prev["volume"]) >= mult * (vol_avg5 or 1e-9)
        if not vol_ok:
            log_skip_reason(sym, "vol_fail", f"bbw={bbw_now:.4f}, mult={mult:.2f}"); return None

        came_from_squeeze = self._came_from_squeeze(d15, len(d15)+j)

        # 30m 레인지 중앙 금지(조건부 해제)
        if self.mode_range(d30):
            up30 = float(d30["bb_up"].iloc[-1]); lo30 = float(d30["bb_lo"].iloc[-1])
            upper_third = lo30 + (up30-lo30)*RANGE_TGT_RATIO
            lower_third = up30 - (up30-lo30)*RANGE_TGT_RATIO
            in_mid = FORBID_MID_RANGE and (lower_third < price < upper_third)

            # late_break: 최근 N봉 내 스퀴즈→확장 감지 시 예외 허용
            late_break = False
            for back in range(3,6):
                if self._came_from_squeeze(d15, len(d15)-back):
                    late_break = True; break

            if in_mid and not (came_from_squeeze or late_break):
                log_skip_reason(sym, "mid_range_forbid"); return None

        # 후보: 상단 돌파 롱 / 하단 돌파 숏
        cand=[]
        atr1h = float(d1h["atr"].iloc[-1] or 0)
        if price>up: cand.append(("long", price))
        if price<lo: cand.append(("short", price))
        if not cand:
            log_skip_reason(sym, "no_candidate"); return None

        # 진입 확인(첫봉/2연속/강봉예외)
        side_try = "long" if price>up else "short"
        if not self._entry_confirmation_ok(d15, side_try):
            log_skip_reason(sym, "entry_confirm_fail", SUB_ENTRY_CONFIRM_MODE); return None

        # 재이탈 허용 관리
        rebreak_ok = False
        rb = getattr(self, "rebreak", {}).get(sym, {"side":None, "tries":0, "deadline":0})
        if came_from_squeeze:
            if price>up: rb = {"side":"long","tries":0,"deadline":REBREAK_DEADLINE_BARS}
            elif price<lo: rb = {"side":"short","tries":0,"deadline":REBREAK_DEADLINE_BARS}
            self.rebreak[sym] = rb
        else:
            if rb["deadline"] > 0 and rb["tries"] < 1:
                if (rb["side"]=="long" and price>up) or (rb["side"]=="short" and price<lo):
                    rebreak_ok = True; rb["tries"] += 1; self.rebreak[sym] = rb

        # 폴라리티 베토(+ 대칭/하단 예외)
        filtered=[]
        for side, e in cand:
            if came_from_squeeze or rebreak_ok:
                filtered.append((side, e)); continue
            veto = self._polarity_veto(price, up, lo, side, atr1h)
            if not veto:
                filtered.append((side, e))
            else:
                if (side=="long") and self._long_upper_band_exception(
                    d30=d30, price=price, up=up, atr1h=atr1h,
                    prev_vol=float(prev["volume"]), vol_avg5=vol_avg5):
                    filtered.append((side, e))
                    log_skip_reason(sym, "long_upper_band_exception", f"atr={atr1h:.4f}")
                elif (side=="short") and self._short_lower_band_exception(
                    d15=d15, d30=d30, price=price, lo=lo, atr1h=atr1h,
                    prev_vol=float(prev["volume"]), vol_avg5=vol_avg5):
                    filtered.append((side, e))
                    log_skip_reason(sym, "short_lower_band_exception", f"atr={atr1h:.4f}")
        cand = filtered
        if not cand:
            log_skip_reason(sym, "polarity_veto"); return None

        # 손절/TP 산출
        best=None; best_rr=-1
        for side, e in cand:
            # SL 산출
            if SUB_STOP_MODE.upper()=="BAND_RECOVERY":
                if side=="long":
                    band_sl = up * (1.0 - BAND_RECOVERY_EPS_PCT)
                    atr_sl  = e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)
                    sl = min(band_sl, atr_sl)
                    sl = min(sl, e*(1.0 - SUB_MIN_STOP_DIST_PCT))  # 최소 거리 보장
                else:
                    band_sl = lo * (1.0 + BAND_RECOVERY_EPS_PCT)
                    atr_sl  = e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)
                    sl = max(band_sl, atr_sl)
                    sl = max(sl, e*(1.0 + SUB_MIN_STOP_DIST_PCT))
            elif SUB_STOP_MODE.upper()=="MIDLINE":
                mid_sl_long  = mid*(1.0 - MID_BUF_PCT)
                mid_sl_short = mid*(1.0 + MID_BUF_PCT)
                if side=="long":
                    cand_far  = min(e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid_sl_long)
                    cand_near = max(e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid_sl_long)
                    sl = cand_far if SUB_SL_CHOICE=="FAR" else cand_near
                else:
                    cand_far  = max(e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid_sl_short)
                    cand_near = min(e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K), mid_sl_short)
                    sl = cand_far if SUB_SL_CHOICE=="FAR" else cand_near
            else:  # "ATR"
                sl = (e - max(SL_MIN_PCT*e, atr1h*STOP_ATR_K)) if side=="long" else (e + max(SL_MIN_PCT*e, atr1h*STOP_ATR_K))

            # TP1/TP2
            risk = abs(e - sl)
            if not self.mode_range(d30):
                tp1 = (e + 2*risk) if side=="long" else (e - 2*risk)
            else:
                up30 = float(d30["bb_up"].iloc[-1]); lo30 = float(d30["bb_lo"].iloc[-1])
                tp1 = (lo30 + (up30-lo30)*RANGE_TGT_RATIO) if side=="long" else (up30 - (up30-lo30)*RANGE_TGT_RATIO)
            tp2 = (e + 3*risk) if side=="long" else (e - 3*risk)  # 힌트값(트레일링 운영)

            R = rr(e, sl, tp1, side)
            if R > best_rr:
                best_rr = R; best = (side, e, sl, tp1, tp2)

        if (best is None) or (best_rr < RR_GATE_SUB):
            log_skip_reason(sym, "rr_gate_fail", f"rr={best_rr:.2f}"); return None

        side, entry, sl, tp1, tp2 = best

        # 🔴 BTC 동조 "필수" — 일반 신호에서도 강제
        if BTC_SYNC_REQUIRED and not self._btc_in_sync(side, d15_btc):
            log_skip_reason(sym, "btc_sync_fail", f"side={side}")
            return None

        # 컨플루언스 카운트(서브)
        conf_count = 0
        if came_from_squeeze or rebreak_ok: conf_count += 1
        if not self.mode_range(d30): conf_count += 1
        if best_rr >= (RR_GATE_SUB + 0.3): conf_count += 1
        add4h, _ = _near_confluence_4h(entry, d4, tol_k=0.6); conf_count += add4h

        retest_px = None
        if RETEST_ENABLE:
            if RETEST_TOUCH_MODE=="band": retest_px = up if side=="long" else lo
            elif RETEST_TOUCH_MODE=="mid": retest_px = mid
            else: retest_px = max(up, mid) if side=="long" else min(lo, mid)

        return {"symbol":sym,"side":side,"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":best_rr,
                "size_factor":1.0,"range_mode": self.mode_range(d30),
                "mid": mid, "up":up, "lo":lo, "came_from_squeeze":came_from_squeeze,
                "retest_px": retest_px, "conf_count": conf_count,
                "bbw": bbw_now, "rebreak_ok": rebreak_ok, "ee_flag": False}

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

    # --- 체결 후 상태(가상) ---
    def after_entry_virtual(self, sym, side, entry, tp1, tp2, sl, qty, sub_mode, retest_px, meta):
        self.state[sym] = {"side":side,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"tp1_done":False,
                           "init_size":qty,"breakeven_set":False,"mode":sub_mode,
                           "split_left": (SPLIT_ENTRY_FIRST_PCT < 1.0), "retest_px":retest_px,
                           "retest_deadline": RETEST_TIMEOUT_BARS, "add_done": False,
                           "ee_flag": meta.get("ee_flag", False), "trail_on": False}

    # --- 체결 후 상태(실계좌) + TP1 주문 ---
    def after_entry(self, sym, side, entry, tp1, tp2, sl, qty, filters, sub_mode, retest_px, meta):
        pos = self.api.get_position(sym)
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
        if sub_mode=="scalp":
            self.api.place_reduce_limit(sym, opp, qty, px1)
        else:
            tp_qty = quantize_qty(qty*TP1_RATIO_SUB, filters["qtyStep"], filters["minQty"])
            if tp_qty > 0:
                self.api.place_reduce_limit(sym, opp, tp_qty, px1)

    # --- 트레일링 SL (TP1 이후) ---
    def _maybe_trail_after_tp1(self, sym: str, api: Bybit):
        st = self.state[sym]
        if not st.get("trail_on", False): return
        df = api.fetch_ohlcv(sym, "15m", max(20, SUB_TP2_TRAIL_LOOKBACK+5))
        if df.empty: return
        atr15 = float(atr(df, ATR_LEN_15M).iloc[-1] or 0)
        price = float(df["close"].iloc[-1])
        cur_sl = st["sl"]
        if st["side"]=="long":
            if SUB_TP2_TRAIL_MODE.upper()=="ATR":
                new_sl = max(cur_sl, price - SUB_TP2_TRAIL_ATR_K*atr15)
            else:  # SWING
                new_sl = max(cur_sl, df["low"].iloc[-SUB_TP2_TRAIL_LOOKBACK:].min())
        else:
            if SUB_TP2_TRAIL_MODE.upper()=="ATR":
                new_sl = min(cur_sl, price + SUB_TP2_TRAIL_ATR_K*atr15)
            else:
                new_sl = min(cur_sl, df["high"].iloc[-SUB_TP2_TRAIL_LOOKBACK:].max())
        if (st["side"]=="long" and new_sl>cur_sl) or (st["side"]=="short" and new_sl<cur_sl):
            st["sl"] = round_price(new_sl, api.get_filters(sym)["tickSize"])
            api.set_stop_loss_mark(sym, st["sl"])

    # --- 리테스트 추가 진입 (평단/SL/BE 재계산 포함) ---
    def _maybe_add_on_retest(self, sym: str, api: Bybit):
        st = self.state[sym]
        if (not SPLIT_ENTRY_ENABLE) or st["add_done"] or (not st["split_left"]) or (not RETEST_ENABLE):
            return
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
            if px > st["entry"]*(1.0 + RETEST_MAX_DRIFT_PCT): 
                st["retest_deadline"] -= 1
            if px <= st["retest_px"]*(1.0 + RETEST_TOUCH_EPS):
                add_qty = quantize_qty(st["init_size"]*(1.0 - SPLIT_ENTRY_FIRST_PCT), f["qtyStep"], f["minQty"])
                if add_qty > 0: _commit_add("long", add_qty)
                st["add_done"]=True
        else:
            if px < st["entry"]*(1.0 - RETEST_MAX_DRIFT_PCT):
                st["retest_deadline"] -= 1
            if px >= st["retest_px"]*(1.0 - RETEST_TOUCH_EPS):
                add_qty = quantize_qty(st["init_size"]*(1.0 - SPLIT_ENTRY_FIRST_PCT), f["qtyStep"], f["minQty"])
                if add_qty > 0: _commit_add("short", add_qty)
                st["add_done"]=True

        # (옵션) 리테스트 실패 시 추격
        if st["retest_deadline"] <= 0 and (not st["add_done"]) and CHASE_ENABLE:
            if st["side"] == "long":
                ok = px <= st["entry"]*(1.0 + CHASE_MAX_DRIFT_PCT)
            else:
                ok = px >= st["entry"]*(1.0 - CHASE_MAX_DRIFT_PCT)
            if ok:
                add_qty = quantize_qty(st["init_size"]*(1.0 - SPLIT_ENTRY_FIRST_PCT)*CHASE_SIZE_RATIO, f["qtyStep"], f["minQty"])
                if add_qty > 0:
                    api.place_market(sym, "Buy" if st["side"]=="long" else "Sell", add_qty, reduce_only=False)
                    new_entry = (st["entry"]*st["init_size"] + px*add_qty) / (st["init_size"]+add_qty)
                    st["entry"] = new_entry; st["init_size"] += add_qty
                    if st.get("breakeven_set", False):
                        be = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                        api.set_stop_loss_mark(sym, be); st["sl"] = be
            st["add_done"] = True
        if st["retest_deadline"] <= 0:
            st["add_done"]=True

    # --- 관리 루프(가상/실계좌) ---
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
                st["sl"] = st["entry"]; st["breakeven_set"]=True
            log_trade_event("TP1", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"]*TP1_RATIO_SUB,
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], rr="", mode=st["mode"],
                            extra="virtual SL→BE", bbw="", came_from_squeeze="", range_mode="", rebreak_ok="",
                            ee_flag=st.get("ee_flag",False))
            print(f"[SUB-V] {sym} TP1→BE")
        if st.get("trail_on", False):
            self._maybe_trail_after_tp1(sym, api)

        if hit_sl or ((st["tp1_done"]) and (price<=st["sl"] if st["side"]=="long" else price>=st["sl"])):
            pnl = (price - st["entry"]) * st["init_size"] if st["side"]=="long" else (st["entry"] - price) * st["init_size"]
            if pnl < 0: record_loss("sub", sym, st["side"])
            log_trade_event("EXIT", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], rr="", mode=st["mode"],
                            exit_price=price, extra="virtual_exit", bbw="", came_from_squeeze="",
                            range_mode="", rebreak_ok="", ee_flag=st.get("ee_flag",False))
            print(f"[SUB-V] {sym} EXIT @ {price}")
            self.state.pop(sym, None)

    def poll_manage(self, sym: str, api: Bybit):
        if sym not in self.state: return
        st = self.state[sym]
        pos = api.get_position(sym)
        if not pos or abs(float(pos.get("size",0) or 0))==0:
            df_last = api.fetch_ohlcv(sym,"15m",2)
            exit_px = float(df_last["close"].iloc[-1]) if not df_last.empty else None
            if exit_px is not None:
                pnl = (exit_px - st["entry"]) * st["init_size"] if st["side"]=="long" else (st["entry"] - exit_px) * st["init_size"]
                if pnl < 0: record_loss("sub", sym, st["side"])
            log_trade_event("EXIT", symbol=sym, engine="sub", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st["tp2"], rr="", mode=st["mode"],
                            exit_price=exit_px, extra="flat", bbw="", came_from_squeeze="", range_mode="",
                            rebreak_ok="", ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None); return

        self._maybe_add_on_retest(sym, api)

        cur_sz = abs(float(pos["size"]))
        if (not st["tp1_done"]) and (cur_sz < st["init_size"]):
            st["tp1_done"]=True; st["trail_on"]=True
            if not st["breakeven_set"]:
                be = round_price(st["entry"], api.get_filters(sym)["tickSize"])
                api.set_stop_loss_mark(sym, be)
                st["sl"]=be; st["breakeven_set"]=True
                realized = st["init_size"] - cur_sz
                log_trade_event("TP1", symbol=sym, engine="sub", side=st["side"], qty=realized,
                                entry=st["entry"], sl=be, tp1=st["tp1"], tp2=st["tp2"], rr="", mode=st["mode"],
                                extra="SL→BE", bbw="", came_from_squeeze="", range_mode="", rebreak_ok="",
                                ee_flag=st.get("ee_flag",False))
                print(f"[SUB] {sym} TP1→BE={be}")

        if st.get("trail_on", False):
            self._maybe_trail_after_tp1(sym, api)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PATCH END (BBEngine 전체 교체: BTC 15m 동조 "필수") <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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
        score += 1  # 4H EMA200 존재
        ob = detect_ob_levels(d4.tail(240))
        fvg= detect_fvg_levels(d4.tail(240))
        if ob: score += 1
        if fvg: score += 1
        rng_now = abs(d1["close"].iloc[-1]-d1["open"].iloc[-1])
        rng_avg = abs(d1["close"]-d1["open"]).tail(20).mean()
        if rng_now >= 0.6*(rng_avg or rng_now): score += 1
        hh = d4["high"].iloc[-20:].max(); ll = d4["low"].iloc[-20:].min(); c  = d4["close"].iloc[-1]
        if (c>=hh) or (c<=ll): score += 1
        vpoc = simple_vpvr_poc(d4.tail(240), bins=40)
        if vpoc and abs(float(c)-vpoc) <= max(float(d4["atr"].iloc[-1] or 0)*0.5, float(c)*0.001):
            score += 1
        return score

    # 4H 확장 초입 트리거
    def _early_expansion_4h(self, d4: pd.DataFrame, d1h: pd.DataFrame) -> Optional[Dict]:
        if not MAIN_EE_ENABLE or len(d4) < max(MAIN_EE_BBW_LOOKBACK+5, 30):
            return None
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
        if not (bbw_prev <= avg_bbw * MAIN_EE_SQUEEZE_K and bbw_now >= avg_bbw * MAIN_EE_EXPAND_K):
            return None
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
                risk = max(price - sl, 1e-9); tp1 = price + 2.0*risk
            else:
                sl = max(hi, price + MAIN_EE_SL_ATR_K*atr4)
                risk = max(sl - price, 1e-9); tp1 = price - 2.0*risk
        else:
            if side=="long": sl = price - MAIN_SL_ATR_K*atr4; tp1 = price + 2*atr4
            else:            sl = price + MAIN_SL_ATR_K*atr4; tp1 = price - 2*atr4
        R = rr(price, sl, tp1, side)
        if R < MAIN_EE_MIN_RR: return None
        tp2 = price + 3*(tp1-price) if side=="long" else price - 3*(price-tp1)  # 힌트
        return {"symbol":None,"side":side,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,"rr":R,
                "ee_flag":True, "size_factor": (MAIN_EE_SIZE_FACTOR if MAIN_EE_ALLOW_LOW_CONF else 1.0)}

    def build_signal(self, sym: str) -> Optional[Dict]:
        pack = self._frames(sym)
        if pack is None:
            print(f"[MAIN] {sym} no data"); return None
        d4, d1, d1h = pack
        price = float(d4["close"].iloc[-1]); ema200 = float(d4["ema200"].iloc[-1]); atr4 = float(d4["atr"].iloc[-1] or 0)

        # (A) 4H 확장 초입 우선 트리거
        ee_sig = self._early_expansion_4h(d4, d1h)
        if ee_sig is not None:
            ee_sig["symbol"] = sym
            return ee_sig

        # (B) 기존 로직
        trend = pivot_trend(d4, look=24)
        side = "long" if price>ema200 else "short"
        if side=="long" and trend<0:
            log_skip_reason(sym, "main_pivot_down_trend_block"); return None
        if side=="short" and trend>0:
            log_skip_reason(sym, "main_pivot_up_trend_block"); return None

        ok_guard, metr = passes_ema_guard(side, d1h, d4)  # d1h 재사용
        if not ok_guard:
            dist=float(metr.get('dist_pct',0.0) or 0.0); slope=float(metr.get('slope',0.0) or 0.0); nb=float(metr.get('neutral_band',0.0) or 0.0)
            log_skip_reason(sym, "main_ema_guard",
                            f"dist_pct={dist:.4f}, slope={slope:.4f}, neutral_band={nb:.4f}")
            return None

        pctB, mid_slope = bb_phase(d4)
        if side == "long":
            if (mid_slope < 0) and (pctB < 0.5):
                log_skip_reason(sym, "main_bb_phase_down"); return None
        else:
            if (mid_slope > 0) and (pctB > 0.5):
                log_skip_reason(sym, "main_bb_phase_up"); return None

        score = self.confluence_score(d4,d1)
        if score < 2:
            align = ((side=="long" and mid_slope>0 and pctB>=0.5) or (side=="short" and mid_slope<0 and pctB<=0.5))
            if not align:
                log_skip_reason(sym,"main_score_low", f"score={score}")
                return None
            size_factor = MAIN_EE_SIZE_FACTOR if MAIN_EE_ALLOW_LOW_CONF else 1.0
        else:
            size_factor = 1.0

        sl = price - MAIN_SL_ATR_K*atr4 if side=="long" else price + MAIN_SL_ATR_K*atr4
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

    def after_entry(self, sym, side, entry, tp1, tp2, sl, qty, filters, ee_flag=False):
        pos = self.api.get_position(sym)
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
        if (not st["tp1_done"]) and hit_tp1:
            st["tp1_done"]=True
            if not st["breakeven_set"]:
                st["sl"] = st["entry"]; st["breakeven_set"]=True
            tp_log_qty = st.get("tp1_qty", st["init_size"] * MAIN_TP1_RATIO)
            log_trade_event("TP1", symbol=sym, engine="main", side=st["side"],
                            qty=tp_log_qty, entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"), rr="", mode="main",
                            extra=("virtual SL→BE" if virtual else "SL→BE"),
                            bbw="", came_from_squeeze="", range_mode="", rebreak_ok="", ee_flag=st.get("ee_flag",False))
            print(f"[MAIN{'-V' if virtual else ''}] {sym} TP1→BE")
        if hit_sl or ((st["tp1_done"]) and (price<=st["sl"] if st["side"]=="long" else price>=st["sl"])) :
            pnl = (price - st["entry"]) * st["init_size"] if st["side"]=="long" else (st["entry"] - price) * st["init_size"]
            if pnl < 0: record_loss("main", sym, st["side"])
            log_trade_event("EXIT", symbol=sym, engine="main", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"), rr="", mode="main",
                            exit_price=price, extra=("virtual_exit" if virtual else "flat"),
                            bbw="", came_from_squeeze="", range_mode="", rebreak_ok="", ee_flag=st.get("ee_flag",False))
            print(f"[MAIN{'-V' if virtual else ''}] {sym} EXIT @ {price}")
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
            if exit_px is not None:
                pnl = (exit_px - st["entry"]) * st["init_size"] if st["side"]=="long" else (st["entry"] - exit_px) * st["init_size"]
                if pnl < 0: record_loss("main", sym, st["side"])
            log_trade_event("EXIT", symbol=sym, engine="main", side=st["side"], qty=st["init_size"],
                            entry=st["entry"], sl=st["sl"], tp1=st["tp1"], tp2=st.get("tp2"), rr="", mode="main",
                            exit_price=exit_px, extra="flat",
                            bbw="", came_from_squeeze="", range_mode="", rebreak_ok="", ee_flag=st.get("ee_flag",False))
            self.state.pop(sym, None); return
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
                                mode="main", extra="SL→BE",
                                bbw="", came_from_squeeze="", range_mode="", rebreak_ok="", ee_flag=st.get("ee_flag",False))
                print(f"[MAIN] {sym} TP1→BE={be}")
        df = api.fetch_ohlcv(sym,"15m",2)
        if df.empty: return
        price = float(df["close"].iloc[-1])
        self._manage_common(sym, price, virtual=False)

# ===========================
# ========== MAIN ===========
# ===========================
def cooldown_ok(engine: str, sym: str, direction: str, bar_ts: int) -> bool:
    k = (engine, sym, direction)
    last = LAST_EXEC_BAR.get(k, None)
    if last is not None and last == bar_ts:
        log_skip_reason(sym, "cooldown_same_bar", f"{engine}/{direction}")
        return False
    return True

def register_exec(engine: str, sym: str, direction: str, bar_ts: int):
    LAST_EXEC_BAR[(engine, sym, direction)] = bar_ts

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
                        # 동일봉 쿨다운 체크(-2 확정봉 기준)
                        d4 = api.fetch_ohlcv(sym, "4h", 3)
                        if d4.empty or len(d4) < 2:
                            log_skip_reason(sym, "cooldown_no_barid_main")
                            sigM = None
                        else:
                            bar_id = int(pd.Timestamp(d4["timestamp"].iloc[-2]).timestamp())
                            direction = "long" if sigM["side"]=="long" else "short"
                            if not cooldown_ok("main", sym, direction, bar_id):
                                sigM = None
                    if sigM:
                        qtyM, fM = main_engine.sizing(sym,
                                                      conf_count=sigM.get("conf_count",0),
                                                      size_factor=sigM.get("size_factor",1.0))
                        if qtyM == 0.0:
                            log_skip_reason(sym, "main_qty_zero_after_budget")
                        else:
                            sideM = "Buy" if sigM["side"]=="long" else "Sell"
                            if VIRTUAL_PAPER:
                                main_engine.after_entry_virtual(sym, sigM["side"], sigM["entry"], sigM["tp1"], sigM.get("tp2"), sigM["sl"], qtyM, ee_flag=sigM.get("ee_flag",False))
                                # 가상모드도 쿨다운 등록
                                register_exec("main", sym, "long" if sigM["side"]=="long" else "short", bar_id)
                            else:
                                if RUNTIME_REVERSE:
                                    pos = api.get_position(sym)
                                    if pos:
                                        cur_sz = float(pos.get("size",0) or 0)
                                        cur_side = "Buy" if cur_sz > 0 else ("Sell" if cur_sz < 0 else None)
                                        if cur_side and ((cur_side=="Buy" and sideM=="Sell") or (cur_side=="Sell" and sideM=="Buy")):
                                            api.close_position(sym); time.sleep(0.3)
                                od = api.place_market(sym, sideM, qtyM, reduce_only=False)
                                if api.ok(od):
                                    main_engine.after_entry(sym, sigM["side"], sigM["entry"], sigM["tp1"], sigM.get("tp2"), sigM["sl"], qtyM, fM, ee_flag=sigM.get("ee_flag",False))
                                    register_exec("main", sym, "long" if sigM["side"]=="long" else "short", bar_id)
                                else:
                                    log_skip_reason(sym, "order_error_main", str(od))
                                    sigM = None
                        if sigM:
                            main_open_cnt += 1
                            log_trade_event("ENTER", symbol=sym, engine="main", side=sigM["side"], qty=qtyM,
                                            entry=sigM["entry"], sl=sigM["sl"], tp1=sigM["tp1"], tp2=sigM.get("tp2"),
                                            rr=round(sigM["rr"],2), mode="main", extra=f"conf={sigM.get('conf_count',0)}",
                                            bbw="", came_from_squeeze="", range_mode="", rebreak_ok="", ee_flag=sigM.get("ee_flag",False))

              
                # === SUB ===
                pos_main = api.get_position(sym)
                main_active_for_sym = bool(pos_main and float(pos_main.get("size",0) or 0) != 0)

                if sub_open_cnt < MAX_SUB_POS and not tick_dead("sub", sym):
                    sigS = sub_engine.build_signal(sym, leader_px, main_active_for_sym)
                    if sigS:
                        # 동일봉 쿨다운 체크(-2 확정봉 기준)
                        d15 = api.fetch_ohlcv(sym, "15m", 3)
                        if d15.empty or len(d15) < 2:
                            log_skip_reason(sym, "cooldown_no_barid_sub")
                            sigS = None
                        else:
                            bar_id = int(pd.Timestamp(d15["timestamp"].iloc[-2]).timestamp())
                            direction = "long" if sigS["side"]=="long" else "short"
                            if not cooldown_ok("sub", sym, direction, bar_id):
                                sigS = None
                    if sigS:
                        qtyS, fS = sub_engine.sizing(sym,
                                                     size_factor=sigS.get("size_factor",1.0),
                                                     conf_count=sigS.get("conf_count",0))
                        if qtyS == 0.0:
                            log_skip_reason(sym, "sub_qty_zero_after_budget")
                        else:
                            sideS = "Buy" if sigS["side"]=="long" else "Sell"
                            if VIRTUAL_PAPER:
                                sub_engine.after_entry_virtual(sym, sigS["side"], sigS["entry"], sigS["tp1"], sigS["tp2"],
                                                               sigS["sl"], qtyS, "sub", sigS.get("retest_px"), sigS)
                                register_exec("sub", sym, direction, bar_id)
                            else:
                                if RUNTIME_REVERSE:
                                    pos = api.get_position(sym)
                                    if pos:
                                        cur_sz = float(pos.get("size",0) or 0)
                                        cur_side = "Buy" if cur_sz > 0 else ("Sell" if cur_sz < 0 else None)
                                        if cur_side and ((cur_side=="Buy" and sideS=="Sell") or (cur_side=="Sell" and sideS=="Buy")):
                                            api.close_position(sym); time.sleep(0.3)
                                od = api.place_market(sym, sideS, qtyS, reduce_only=False)
                                if api.ok(od):
                                    sub_engine.after_entry(sym, sigS["side"], sigS["entry"], sigS["tp1"], sigS["tp2"],
                                                           sigS["sl"], qtyS, fS, "sub", sigS.get("retest_px"), sigS)
                                    register_exec("sub", sym, direction, bar_id)
                                else:
                                    log_skip_reason(sym, "order_error_sub", str(od))
                                    sigS = None
                        if sigS:
                            sub_open_cnt += 1
                            log_trade_event("ENTER", symbol=sym, engine="sub", side=sigS["side"], qty=qtyS,
                                            entry=sigS["entry"], sl=sigS["sl"], tp1=sigS["tp1"], tp2=sigS["tp2"],
                                            rr=round(sigS["rr"],2), mode="sub", 
                                            extra=f"conf={sigS.get('conf_count',0)}",
                                            bbw=sigS.get("bbw"), came_from_squeeze=sigS.get("came_from_squeeze"),
                                            range_mode=sigS.get("range_mode"), rebreak_ok=sigS.get("rebreak_ok"),
                                            ee_flag=sigS.get("ee_flag",False))

            # === 관리 루프 ===
            for sym in SYMBOLS:
                sub_engine.poll_manage_virtual(sym, api) if VIRTUAL_PAPER else sub_engine.poll_manage(sym, api)
                main_engine.poll_manage_virtual(sym, api) if VIRTUAL_PAPER else main_engine.poll_manage(sym, api)

            loop_i += 1
            time.sleep(POLL_SEC)

        except Exception as e:
            print("[ERROR main loop]", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
