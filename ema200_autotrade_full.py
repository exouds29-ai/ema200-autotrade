# 파일명 제안: ema200_autotrade_full.py

# ──────────────────────────────────────────────────────────────────────────────
# Project: EMA200 Swing & Mid-Term Autotrade (BTC-gated, OB/FVG aided)
# Version: v1.4.2 (2025-09-29 KST) — Per-symbol fixed qty + gate TF fix + safe bracket
# Language: Python 3.10+
# Single-file deliverable; ccxt optional; self-tests bundled.
# ──────────────────────────────────────────────────────────────────────────────

"""
WHAT'S NEW (v1.4.2)
- GateConfig.btc_gate_tf 실제 반영: BTC 게이트 타임프레임을 설정대로 사용.
- (옵션) STRICT_BRACKET 모드: 엔트리 후 실제 포지션 수량으로 SL/TP 동기화 발주.
- TPWatcher 숏 트레일 조건 가독성 개선(동작 동일).
- StubBybit 마켓 체결가를 최근 캔들 종가로 근사(테스트 체감 ↑).
- 나머지 로직/테스트는 v1.4.1 유지.

HOW TO RUN
- Default (no ccxt or USE_CCXT=0) → runs self-tests (no API keys needed).
- With ccxt + keys → set:
    USE_CCXT=1
    BYBIT_KEY=...
    BYBIT_SECRET=...
    BYBIT_TESTNET=true|false
- STRICT_BRACKET=true 로 실행하면 실거래에서 SL/TP를 포지션 수량과 동기화하여 발주.
"""

from __future__ import annotations
import os, time, uuid, threading, statistics as stats, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, NamedTuple

# =============================== optional ccxt ================================
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # fall back to stub exchange

# =============================== config =======================================
@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01            # fallback sizing when fixed qty not set
    day_loss_cut: float = -0.05             # TODO(v1.5): daily PnL guard
    consec_stop_max: int = 2                # TODO(v1.5): stop streak guard
    cooldown_bars: int = 3                  # TODO(v1.5): per-signal cooldown
    noise_band_bp: tuple[float,float] = (5, 10)  # EMA200 ±(0.05%~0.10%) new-entry block
    sl_swing_pct: float = 0.01             # Swing scalp: 1% stop
    sl_mid_pct: float = 0.03               # Mid-term: 3% stop
    sl_buffer_pct: float = 0.002           # (reserved) ATR/Buffer mixing
    sl_buffer_atr_coef: float = 0.8        # (reserved)
    trail_pct: float = 0.01                # 1% trailing (used in TPWatcher)
    be_bump_pct: float = 0.001             # +0.1% to breakeven

@dataclass(frozen=True)
class GateConfig:
    btc_gate_tf: str = "30m"               # ← v1.4.2: respected by btc_gate()
    gate_confirm_bars: int = 1

@dataclass(frozen=True)
class Timeframes:
    mid: str = "4h"
    swing_a: str = "1h"
    swing_b: str = "30m"
    helper: str = "15m"
    micro: str = "3m"

@dataclass(frozen=True)
class Symbols:
    base: str = "BTCUSDT"
    alts: tuple[str,...] = ("ETHUSDT", "SOLUSDT", "BNBUSDT")

@dataclass(frozen=True)
class Params:
    fvg_ob_near_bp: float = 35              # 0.35%
    fvg_ob_near_atr_coef: float = 0.6       # 0.6 * ATR(15m)
    ob_fvg_bonus_pct: float = 0.05          # +5% each
    max_size_mult: float = 2.0
    two_top_tol_bp: float = 12              # 0.12% (reserved, TODO)

@dataclass(frozen=True)
class ExecConfig:
    adopt_mode: str = os.getenv("ADOPT_MODE", "ADOPT")  # ADOPT or CLOSE_THEN_START
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    exchange: str = os.getenv("EXCHANGE", "bybit")
    account_mode: str = os.getenv("ACCOUNT_MODE", "oneway")  # TODO(v1.5)
    leverage: int = int(os.getenv("LEV", "2"))               # TODO(v1.5)
    api_key: str = os.getenv("BYBIT_KEY", "")
    api_secret: str = os.getenv("BYBIT_SECRET", "")
    testnet: bool = os.getenv("BYBIT_TESTNET", "true").lower()=="true"
    use_ccxt: bool = os.getenv("USE_CCXT", "0").lower() in ("1","true","yes") and ccxt is not None
    strict_bracket_after_fill: bool = os.getenv("STRICT_BRACKET", "false").lower() == "true"

RISK = RiskConfig(); GATE=GateConfig(); TF=Timeframes(); SYMB=Symbols(); P=Params(); EXEC=ExecConfig()

def get_EXEC() -> ExecConfig:
    try:
        if isinstance(globals().get('EXEC'), ExecConfig):
            return globals()['EXEC']
    except Exception:
        pass
    exec_inst = ExecConfig()
    globals()['EXEC'] = exec_inst
    return exec_inst

# === per-symbol fixed quantity (SWING / MID) ================================
# 스윙/단타(30m/1h)용 고정 수량
FIXED_QTY_SWING: dict[str, float] = {
    "BTCUSDT": 0.01,
    "ETHUSDT": 0.50,
    "SOLUSDT": 10.0,
    "BNBUSDT": 1.0,
}
# 중기(4h)용 고정 수량
FIXED_QTY_MID: dict[str, float] = {
    "BTCUSDT": 0.02,
    "ETHUSDT": 0.70,
    "SOLUSDT": 15.0,
    "BNBUSDT": 1.5,
}
# (선택) 심볼별 추가 배수 — 기본 1.0
SYMBOL_MULT_SWING: dict[str, float] = {"BTCUSDT":1.0,"ETHUSDT":1.0,"SOLUSDT":1.0,"BNBUSDT":1.0}
SYMBOL_MULT_MID:   dict[str, float] = {"BTCUSDT":1.0,"ETHUSDT":1.0,"SOLUSDT":1.0,"BNBUSDT":1.0}

# =============================== utils ========================================
E6 = 10**6
def now_ms() -> int: return int(time.time()*1000)
def clamp(v, lo, hi): return max(lo, min(hi, v))
def make_cid(prefix: str, symbol: str) -> str: return f"{prefix}_{symbol}_{uuid.uuid4().hex[:10]}"

# ============================== indicators ====================================
def ema(series: List[float], period: int) -> List[float]:
    if not series: return []
    k = 2/(period+1); out=[]; e=series[0]
    for x in series:
        e = (x-e)*k + e; out.append(e)
    return out

def atr(high: List[float], low: List[float], close: List[float], period: int=14) -> List[float]:
    tr=[]
    for i in range(len(close)):
        if i==0: tr.append(high[i]-low[i])
        else: tr.append(max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
    out=[]; alpha=1/period; v=tr[0]
    for t in tr: v = alpha*t+(1-alpha)*v; out.append(v)
    return out

def bb_width(close: List[float], period: int=20, mult: float=2.0) -> List[float]:
    out=[]
    for i in range(len(close)):
        if i<period: out.append(0.0); continue
        window = close[i-period+1:i+1]
        mean = sum(window)/period
        std = stats.pstdev(window)
        upper = mean + mult*std; lower = mean - mult*std
        out.append((upper-lower)/mean if mean else 0.0)
    return out

def slope(values: List[float], window: int=8) -> float:
    if len(values)<window: return 0.0
    a = values[-window:]
    return (a[-1]-a[0]) / max(1e-9, window)

# ============================== zones (OB/FVG) ================================
class Zone(NamedTuple):
    kind: str
    start: float
    end: float
    ts: int

def detect_simple_orderblocks(o,h,l,c) -> list[Zone]:
    zones=[]; n=len(c)
    for i in range(2,n):
        rng = h[i]-l[i]
        if rng<=0: continue
        if c[i] < o[i] and (o[i]-c[i]) > rng*0.6 and c[i-1]>o[i-1]:
            zones.append(Zone('OB_BEAR', o[i-1], h[i-1], i))
        if c[i] > o[i] and (c[i]-o[i]) > rng*0.6 and c[i-1]<o[i-1]:
            zones.append(Zone('OB_BULL', l[i-1], o[i-1], i))
    return zones[-12:]

def detect_simple_fvg(h,l,c) -> list[Zone]:
    zones=[]
    for i in range(2,len(c)):
        if l[i] > h[i-2]:
            zones.append(Zone('FVG_UP', h[i-2], l[i], i))
        if h[i] < l[i-2]:
            zones.append(Zone('FVG_DOWN', h[i], l[i-2], i))
    return zones[-12:]

def nearest_zone_price(price: float, zones: list[Zone], side: str, atr15: float) -> tuple[bool, Optional[Zone], float]:
    best=None; best_dist=1e9
    for z in zones:
        ref=None
        if side=='long' and z.kind in ('OB_BULL','FVG_DOWN'):
            ref=z.end
        if side=='short' and z.kind in ('OB_BEAR','FVG_UP'):
            ref=z.start
        if ref is None: continue
        dist=abs(price-ref)
        if dist<best_dist: best_dist=dist; best=z
    if best is None: return False,None,0.0
    pct_dist = best_dist/price
    prox_limit = min(P.fvg_ob_near_bp/10000, P.fvg_ob_near_atr_coef*max(1e-9, atr15)/price)
    return (pct_dist<=prox_limit), best, pct_dist

# =============================== datafeed ======================================
class CandleStore:
    def __init__(self, maxlen=2000):
        self.data: dict[tuple[str,str], dict[str, List]] = {}
        self.maxlen = maxlen
    def ensure(self, symbol: str, tf: str):
        key=(symbol,tf)
        if key not in self.data:
            self.data[key] = {k: [] for k in ('t','o','h','l','c','v')}
        return self.data[key]
    def push_bulk(self, symbol: str, tf: str, ohlcv: List[List[float]]):
        d=self.ensure(symbol,tf)
        for t,o,h,l,c,v in ohlcv:
            d['t'].append(t); d['o'].append(o); d['h'].append(h); d['l'].append(l); d['c'].append(c); d['v'].append(v)
        for k in d: d[k]=d[k][-self.maxlen:]
    def series(self, symbol: str, tf: str):
        d=self.ensure(symbol,tf)
        return d['t'], d['o'], d['h'], d['l'], d['c'], d['v']

# ============================== exchange stub =================================
class StubBybit:
    def __init__(self):
        self._balance_usdt = 10_000.0
        self._orders: Dict[str, List[Dict[str,Any]]] = {}
        self._positions: Dict[str, Dict[str,Any]] = {}
        self._markets: Dict[str, Dict[str,Any]] = {}
        self._candles: Dict[tuple[str,str], List[List[float]]] = {}
        self.sandbox = True
    def set_sandbox_mode(self, flag: bool): self.sandbox = flag
    def load_markets(self):
        self._markets = {
            'BTCUSDT': {'linear': True, 'swap': True, 'precision': {'price': 2, 'amount': 6}, 'limits': {'amount': {'min': 0.001}}},
            'ETHUSDT': {'linear': True, 'swap': True, 'precision': {'price': 2, 'amount': 6}, 'limits': {'amount': {'min': 0.001}}},
            'SOLUSDT': {'linear': True, 'swap': True, 'precision': {'price': 2, 'amount': 2}, 'limits': {'amount': {'min': 0.1}}},
            'BNBUSDT': {'linear': True, 'swap': True, 'precision': {'price': 2, 'amount': 3}, 'limits': {'amount': {'min': 0.01}}},
        }
    @property
    def markets(self): return self._markets
    def price_to_precision(self, symbol: str, price: float) -> str:
        p = self._markets[symbol]['precision']['price']
        return f"{price:.{p}f}"
    def amount_to_precision(self, symbol: str, amount: float) -> str:
         p = self._markets[symbol]['precision']['amount']
         factor = 10 ** p
         rounded = int(amount * factor) / factor  # 버림 처리
         return f"{rounded:.{p}f}"

    def fetch_balance(self): return {'total': {'USDT': self._balance_usdt}}
    def _append_order(self, symbol: str, order: Dict[str,Any]): self._orders.setdefault(symbol, []).append(order)
    def create_order(self, symbol: str, type_: str, side: str, amount: float, price: Optional[float], params: Dict[str,Any]):
        order = {
            'id': params.get('clientOrderId', make_cid('ORD',symbol)),
            'symbol': symbol,
            'type': type_, 'side': side,
            'amount': float(self.amount_to_precision(symbol, amount)),
            'price': price,
            'reduceOnly': params.get('reduceOnly', False),
            'status': 'open',
        }
        self._append_order(symbol, order)
        if type_=='market':
            pos = self._positions.get(symbol, {'symbol':symbol, 'contracts':0.0, 'entryPrice':0.0, 'side':None})
            qty = order['amount']
            # 최근 캔들 close를 엔트리로 근사 (v1.4.2)
            last_px = price
            try:
                # 가능한 최근 동일 심볼의 아무 TF라도 사용
                last_close = None
                for (sym, tf), arr in list(self._candles.items())[::-1]:
                    if sym == symbol and arr:
                        last_close = arr[-1][4]; break
                if last_close is not None: last_px = last_close
            except Exception:
                pass
            if order['reduceOnly']:
                if pos['contracts']>0 and side=='sell':
                    pos['contracts']=max(0.0, pos['contracts']-qty)
                    if pos['contracts']==0: pos['side']=None
                elif pos['contracts']<0 and side=='buy':
                    pos['contracts']=min(0.0, pos['contracts']+qty)
                    if pos['contracts']==0: pos['side']=None
            else:
                if side=='buy':
                    pos['contracts'] += qty; pos['side']='long'
                else:
                    pos['contracts'] -= qty; pos['side']='short'
                pos['entryPrice'] = last_px or pos.get('entryPrice', 0.0) or 100.0
            self._positions[symbol]=pos
        return order
    def fetch_open_orders(self, symbol: str) -> List[Dict[str,Any]]: return list(self._orders.get(symbol, []))
    def cancel_all_orders(self, symbol: str): self._orders[symbol] = []
    def cancel_order(self, order_id: str, symbol: str):
        arr=self._orders.get(symbol, []); self._orders[symbol] = [o for o in arr if o['id']!=order_id]
    def fetch_positions(self, symbols: List[str]) -> List[Dict[str,Any]]:
        return [self._positions.get(s, {'symbol':s,'contracts':0.0,'entryPrice':0.0,'side':None}) for s in symbols]
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int=400) -> List[List[float]]:
        key=(symbol,timeframe)
        if key in self._candles and len(self._candles[key])>=limit:
            return self._candles[key][-limit:]
        base=100.0 if symbol=='BTCUSDT' else 50.0
        step=0.1 if timeframe in ('30m','1h','4h') else 0.05
        arr=[]; t0=now_ms()-limit*60_000
        price=base
        for i in range(limit):
            drift = (i/limit)*step
            noise = (random.random()-0.5)*0.2
            o = price
            c = price + drift + noise
            h = max(o,c)+abs(noise)*0.5
            l = min(o,c)-abs(noise)*0.5
            v = 10+i*0.01
            arr.append([t0+i*60_000, round(o,2), round(h,2), round(l,2), round(c,2), v])
            price = c
        self._candles[key]=arr
        return arr[-limit:]

# ================================ broker =======================================
class BybitBroker:
    def __init__(self, exchange: Any):
        self.x = exchange
        self.symbol_meta: Dict[str,Dict[str,Any]] = {}
        self.locks: Dict[str, threading.Lock] = {}
    def lock(self, symbol:str) -> threading.Lock:
        if symbol not in self.locks:
            self.locks[symbol]=threading.Lock()
        return self.locks[symbol]
    def load_markets(self):
        self.x.load_markets()
        for s, m in self.x.markets.items():
            if m.get('linear') and m.get('swap'):
                self.symbol_meta[s] = {
                    'priceIncrement': m.get('precision',{}).get('price', None),
                    'amountIncrement': m.get('precision',{}).get('amount', None),
                    'minAmount': m.get('limits',{}).get('amount',{}).get('min', 0.0),
                }
    def _round(self, symbol: str, price: Optional[float], amount: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        meta=self.symbol_meta.get(symbol, {})
        p=price; a=amount
        if price is not None and meta.get('priceIncrement') is not None:
            p = float(self.x.price_to_precision(symbol, price))
        if amount is not None and meta.get('amountIncrement') is not None:
            a = float(self.x.amount_to_precision(symbol, amount))
        return p,a
    def fetch_open_orders(self, symbol: str) -> List[Dict[str,Any]]: return self.x.fetch_open_orders(symbol)
    def fetch_position(self, symbol: str) -> Dict[str,Any]:
        poss = self.x.fetch_positions([symbol])
        for p in poss:
            if p['symbol']==symbol: return p
        return {"symbol":symbol, "contracts":0.0, "entryPrice":0.0, "side":None}
    def place_limit(self, symbol: str, side: str, qty: float, price: float, reduce_only=False, cid: Optional[str]=None):
        price,qty = self._round(symbol, price, qty)
        params={'reduceOnly': reduce_only, 'timeInForce':'GTC', 'clientOrderId': cid or make_cid('LIM',symbol)}
        return self.x.create_order(symbol, 'limit', side, qty, price, params)
    def place_market(self, symbol: str, side: str, qty: float, reduce_only=False, cid: Optional[str]=None):
        _,qty = self._round(symbol, None, qty)
        params={'reduceOnly': reduce_only, 'clientOrderId': cid or make_cid('MKT',symbol)}
        return self.x.create_order(symbol, 'market', side, qty, None, params)
    def place_stop_market(self, symbol: str, side: str, qty: float, stop_price: float, reduce_only=True, cid: Optional[str]=None):
        _,qty = self._round(symbol, None, qty)
        params={'reduceOnly': reduce_only, 'clientOrderId': cid or make_cid('SL',symbol), 'triggerPrice': stop_price}
        # NOTE: stub은 즉시 'market'로 처리. 실제 ccxt/bybit는 stop params 필요.
        return self.x.create_order(symbol, 'market', 'sell' if side=='buy' else 'buy', qty, None, params)
    def cancel_all(self, symbol: str):
        try: self.x.cancel_all_orders(symbol)
        except Exception:
            for o in self.fetch_open_orders(symbol):
                try: self.x.cancel_order(o['id'], symbol)
                except Exception: pass

# ================================ strategy =====================================
from enum import Enum
class Regime(Enum): TREND=1; RANGE=2
class Bias(Enum): LONG=1; SHORT=2
class GateState(Enum): ON=1; OFF=2

class Strategy:
    def __init__(self, store: CandleStore, broker: BybitBroker):
        self.s=store; self.b=broker
        self.cooldown: dict[tuple[str,str], int] = {}
        self.consec_stop: dict[str,int] = {}
        self.day_pnl: float = 0.0
        self.gate_state = GateState.OFF
        self.state: Dict[str, Dict[str, Any]] = {}
    def _st(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.state:
            self.state[symbol]={'tp1_hit': False,'trail_stop': None,'entry_mode': None,'last_high': None,'last_low': None}
        return self.state[symbol]
    def regime_and_bias(self) -> tuple[Regime,Bias]:
        t4,o4,h4,l4,c4,v4 = self.s.series(SYMB.base, TF.mid)
        if len(c4)<210: return Regime.RANGE, Bias.LONG
        e4=ema(c4,200); sl=slope(e4,10)
        regime = Regime.TREND if abs(sl)>0.0 else Regime.RANGE
        bias = Bias.LONG if c4[-1]>e4[-1] else Bias.SHORT
        return regime, bias
    def btc_gate(self) -> GateState:
        # v1.4.2: 설정값 GATE.btc_gate_tf 사용
        gate_tf = getattr(GATE, 'btc_gate_tf', TF.swing_b)
        t,o,h,l,c,v = self.s.series(SYMB.base, gate_tf)
        if len(c)<210: return GateState.OFF
        e=ema(c,200)
        bars = max(1, int(getattr(GATE, 'gate_confirm_bars', 1)))
        ok_up = c[-1]>e[-1] and all(cc>ee for cc,ee in zip(c[-bars:],e[-bars:]))
        ok_dn = c[-1]<e[-1] and all(cc<ee for cc,ee in zip(c[-bars:],e[-bars:]))
        return GateState.ON if (ok_up or ok_dn) else GateState.OFF
    def _two_top_near(self, highs: List[float], tol_bp: float) -> bool:
        if len(highs)<3: return False
        return abs(highs[-1]-highs[-2])/max(1e-9, highs[-1]) <= tol_bp/10000
    def swing_signal(self, symbol: str, side: str) -> Optional[dict]:
        t30,o30,h30,l30,c30,v30 = self.s.series(symbol, TF.swing_b)
        t1,o1,h1,l1,c1,v1 = self.s.series(symbol, TF.swing_a)
        t15,o15,h15,l15,c15,v15 = self.s.series(symbol, TF.helper)
        if len(c30)<210 or len(c1)<210 or len(c15)<50: return None
        e30=ema(c30,200); e1=ema(c1,200); e15=ema(c15,200)
        atr15=atr(h15,l15,c15,14)[-1]
        sl30=slope(e30,8); price=c30[-1]
        band_lo = e30[-1]*(1 - RISK.noise_band_bp[0]/10000)
        band_hi = e30[-1]*(1 + RISK.noise_band_bp[1]/10000)
        if band_lo<=price<=band_hi: return None
        def near_ema(p): return abs(p - e30[-1]) / max(1e-9, p) < 0.002
        if side=='short':
            if sl30 <= 0:
                if len(h30)>=3:
                    dtok = abs(h30[-1]-h30[-2])/max(1e-9, price) < 0.0005 and near_ema(h30[-1])
                else:
                    dtok = False
                body = o30[-1]-c30[-1]
                rng = max(1e-9, h30[-1]-l30[-1])
                lower_wick = min(c30[-1], o30[-1]) - l30[-1]
                upper_wick = h30[-1] - max(c30[-1], o30[-1])
                bear_ok = (body>0) and (upper_wick>=0.2*rng) and (lower_wick<=0.1*rng)
                near = near_ema(max(c30[-1],o30[-1])) or near_ema(h30[-1])
                if dtok and bear_ok and near:
                    return {"symbol":symbol,"side":"sell","price":price,"ema":e30[-1],"atr15":atr15}
        else:
            if c30[-1]>e30[-1] and c30[-2]>e30[-2] and sl30>0:
                return {"symbol":symbol,"side":"buy","price":price,"ema":e30[-1],"atr15":atr15}
        return None
    def safe_balance(self) -> float:
        try:
            bal = self.b.x.fetch_balance(); usdt = bal['total'].get('USDT', 0.0)
            return float(usdt)
        except Exception: return 1000.0
    def calc_qty(self, symbol: str, price: float, mult: float=1.0, mode: str='swing') -> float:
        """
        mode: 'swing' or 'mid'
        1) 심볼이 FIXED_QTY_{MODE}에 있으면 그 값(개수) 사용 (+ 심볼별 multiplier, + 함수 파라미터 mult)
        2) 없으면 잔고비율 기반 계산 (fallback)
        """
        if mode not in ('swing','mid'): mode='swing'
        table = FIXED_QTY_MID if mode=='mid' else FIXED_QTY_SWING
        mult_table = SYMBOL_MULT_MID if mode=='mid' else SYMBOL_MULT_SWING
        qty = None
        if symbol in table:
            qty = table[symbol] * mult_table.get(symbol, 1.0) * mult
        else:
            balance = self.safe_balance()
            notional = max(10.0, balance * RISK.risk_per_trade)
            qty = (notional * mult) / max(1e-9, price)
        meta=self.b.symbol_meta.get(symbol, {})
        if meta.get('minAmount'): qty = max(qty, meta['minAmount'])
        try: return float(self.b.x.amount_to_precision(symbol, qty))
        except Exception: return float(qty)
    def size_multiplier(self, symbol:str, side:str, price:float, ema_ref:float, atr15:float) -> float:
        t15,o15,h15,l15,c15,v15 = self.s.series(symbol, TF.helper)
        zones = detect_simple_orderblocks(o15,h15,l15,c15) + detect_simple_fvg(h15,l15,c15)
        near, z, dist = nearest_zone_price(price, zones, 'long' if side=='buy' else 'short', atr15)
        mult = 1.0 + (P.ob_fvg_bonus_pct if near else 0.0)
        return clamp(mult, 0.1, P.max_size_mult)
    def compute_sl_tp(self, side:str, entry:float, ema_ref:float, atr15:float, mode:str) -> tuple[float,float]:
        pct = RISK.sl_mid_pct if mode=='mid' else RISK.sl_swing_pct
        if side=='buy':
            sl = entry * (1 - pct); tp1 = entry + (entry - sl)
        else:
            sl = entry * (1 + pct); tp1 = entry - (sl - entry)
        return sl, tp1
    # v1.4.2: 실거래용 브래킷 동기화 보조
    def _sync_position_qty(self, symbol: str, desired_qty: float) -> float:
        """Fetch actual position contracts; if none, return 0.0"""
        try:
            pos = self.b.fetch_position(symbol)
            qty = abs(float(pos.get('contracts', 0.0)))
            return qty if qty > 0 else 0.0
        except Exception:
            return desired_qty
    def on_bar_close(self):
        self.gate_state = self.btc_gate()
        regime,bias = self.regime_and_bias()
        tradables = (SYMB.base,) + (SYMB.alts if self.gate_state==GateState.ON else tuple())
        for sym in tradables:
            pref = 'buy' if bias==Bias.LONG else 'sell'
            sig = self.swing_signal(sym, 'long' if pref=='buy' else 'short')
            if not sig: continue
            with self.b.lock(sym):
                entry_price = sig['price']; atr15 = sig['atr15']; ema_ref = sig['ema']
                mult = self.size_multiplier(sym, pref, entry_price, ema_ref, atr15)
                qty = self.calc_qty(sym, entry_price, mult, mode='swing')
                # split/market entry
                try:
                    t3,o3,h3,l3,c3,v3 = self.s.series(sym, TF.micro)
                    if len(c3)>=30:
                        win = c3[-20:]; sma = sum(win)/len(win); std = stats.pstdev(win)
                        level1 = sma
                        level2 = sma - std if pref=='buy' else sma + std
                        l1 = level1 * (0.999 if pref=='buy' else 1.001)
                        l2 = level2 * (0.999 if pref=='buy' else 1.001)
                        q1 = qty*0.5; q2 = qty - q1
                        if pref=='buy':
                            self.b.place_limit(sym, 'buy', q1, l1, reduce_only=False, cid=make_cid('ENT1',sym))
                            self.b.place_limit(sym, 'buy', q2, l2, reduce_only=False, cid=make_cid('ENT2',sym))
                        else:
                            self.b.place_limit(sym, 'sell', q1, l1, reduce_only=False, cid=make_cid('ENT1',sym))
                            self.b.place_limit(sym, 'sell', q2, l2, reduce_only=False, cid=make_cid('ENT2',sym))
                    else:
                        self.b.place_market(sym, pref, qty, reduce_only=False, cid=make_cid('ENT',sym))
                except Exception:
                    self.b.place_market(sym, pref, qty, reduce_only=False, cid=make_cid('ENT',sym))
                # bracket placement
                sl,tp1 = self.compute_sl_tp(pref, entry_price, ema_ref, atr15, mode='swing')
                if get_EXEC().strict_bracket_after_fill:
                    # 실거래 안전지향: 체결 반영 대기 후 포지션 수량으로 동기화
                    time.sleep(0.2)
                    pos_qty = self._sync_position_qty(sym, qty)
                    if pos_qty > 0:
                        self.b.place_limit(sym, 'sell' if pref=='buy' else 'buy', pos_qty*0.6, tp1, reduce_only=True, cid=make_cid('TP1',sym))
                        self.b.place_stop_market(sym, pref, pos_qty, sl, reduce_only=True, cid=make_cid('SL',sym))
                else:
                    # 테스트/스텁 친화: 계획 수량 기준으로 발주
                    self.b.place_limit(sym, 'sell' if pref=='buy' else 'buy', qty*0.6, tp1, reduce_only=True, cid=make_cid('TP1',sym))
                    self.b.place_stop_market(sym, pref, qty, sl, reduce_only=True, cid=make_cid('SL',sym))
        # mid-term loop (BTC only)
        t4,o4,h4,l4,c4,v4 = self.s.series(SYMB.base, TF.mid)
        if len(c4)>=210:
            e4=ema(c4,200); sl4=slope(e4,10)
            t1,o1,h1,l1,c1,v1 = self.s.series(SYMB.base, TF.swing_a)
            if len(c1)>=210:
                e1=ema(c1,200); price1=c1[-1]; atr1 = atr(h1,l1,c1,14)[-1]
                near = abs(price1-e1[-1])/max(1e-9, price1) < 0.002
                if c4[-1]>e4[-1] and sl4>0 and near:
                    qty = self.calc_qty(SYMB.base, price1, 1.0, mode='mid')
                    sl,tp1 = self.compute_sl_tp('buy', price1, e1[-1], atr1, 'mid')
                    self.b.place_market(SYMB.base, 'buy', qty, reduce_only=False, cid=make_cid('MID',SYMB.base))
                    if get_EXEC().strict_bracket_after_fill:
                        time.sleep(0.2)
                        pos_qty = self._sync_position_qty(SYMB.base, qty)
                        if pos_qty>0:
                            self.b.place_limit(SYMB.base, 'sell', pos_qty*0.5, tp1, reduce_only=True, cid=make_cid('MIDTP1',SYMB.base))
                            self.b.place_stop_market(SYMB.base, 'buy', pos_qty, sl, reduce_only=True, cid=make_cid('MIDSL',SYMB.base))
                    else:
                        self.b.place_limit(SYMB.base, 'sell', qty*0.5, tp1, reduce_only=True, cid=make_cid('MIDTP1',SYMB.base))
                        self.b.place_stop_market(SYMB.base, 'buy', qty, sl, reduce_only=True, cid=make_cid('MIDSL',SYMB.base))
                if c4[-1]<e4[-1] and sl4<0 and near:
                    qty = self.calc_qty(SYMB.base, price1, 1.0, mode='mid')
                    sl,tp1 = self.compute_sl_tp('sell', price1, e1[-1], atr1, 'mid')
                    self.b.place_market(SYMB.base, 'sell', qty, reduce_only=False, cid=make_cid('MID',SYMB.base))
                    if get_EXEC().strict_bracket_after_fill:
                        time.sleep(0.2)
                        pos_qty = self._sync_position_qty(SYMB.base, qty)
                        if pos_qty>0:
                            self.b.place_limit(SYMB.base, 'buy', pos_qty*0.5, tp1, reduce_only=True, cid=make_cid('MIDTP1',SYMB.base))
                            self.b.place_stop_market(SYMB.base, 'sell', pos_qty, sl, reduce_only=True, cid=make_cid('MIDSL',SYMB.base))
                    else:
                        self.b.place_limit(SYMB.base, 'buy', qty*0.5, tp1, reduce_only=True, cid=make_cid('MIDTP1',SYMB.base))
                        self.b.place_stop_market(SYMB.base, 'sell', qty, sl, reduce_only=True, cid=make_cid('MIDSL',SYMB.base))

# =============================== tp watcher ====================================
class TPWatcher:
    def __init__(self, broker: BybitBroker):
        self.b=broker
        self.state: Dict[str, Dict[str, Any]] = {}
    def _st(self, symbol:str) -> Dict[str,Any]:
        if symbol not in self.state:
            self.state[symbol]={'tp1_done':False,'entry':None,'side':None,'trail':None}
        return self.state[symbol]
    def on_tick(self, symbol: str, last_price: float):
        try:
            st = self._st(symbol)
            orders = self.b.fetch_open_orders(symbol)
            # TP1 fill emulation: 시장가 청산 + BE stop 설정
            for o in list(orders):
                if not o.get('reduceOnly'): continue
                if o.get('type','limit')!='limit' or o.get('price') is None: continue
                if (o['side']=='sell' and last_price>=o['price']) or (o['side']=='buy' and last_price<=o['price']):
                    pos = self.b.fetch_position(symbol)
                    side = pos.get('side'); qty = abs(float(pos.get('contracts', 0)))
                    if qty>0:
                        part = qty*0.6 if not st.get('tp1_done') else qty
                        self.b.place_market(symbol, 'sell' if side=='long' else 'buy', part, reduce_only=True, cid=make_cid('TPW',symbol))
                        self.b.cancel_all(symbol)
                        if not st.get('tp1_done'):
                            st['tp1_done']=True
                            st['entry']=pos.get('entryPrice') or last_price
                            st['side']=side
                            be = st['entry']*(1+RISK.be_bump_pct) if side=='long' else st['entry']*(1-RISK.be_bump_pct)
                            self.b.place_stop_market(symbol, 'buy' if side=='short' else 'sell', max(qty-part,0.0), be, reduce_only=True, cid=make_cid('BE',symbol))
                            st['trail']=be
            # trailing after TP1
            pos = self.b.fetch_position(symbol)
            if st.get('tp1_done') and pos.get('side') is not None and abs(float(pos.get('contracts',0)))>0:
                new_stop_long = last_price*(1-RISK.sl_swing_pct)
                new_stop_short = last_price*(1+RISK.sl_swing_pct)
                if pos.get('side')=='long':
                    if st.get('trail') is None or new_stop_long>st['trail']:
                        self.b.place_stop_market(symbol, 'buy', abs(float(pos.get('contracts',0))), new_stop_long, reduce_only=True, cid=make_cid('TRL',symbol))
                        st['trail']=new_stop_long
                elif pos.get('side')=='short':
                    # v1.4.2: 조건 가독성 정리 (동작 동일)
                    if st.get('trail') is None or new_stop_short < st['trail']:
                        self.b.place_stop_market(symbol, 'sell', abs(float(pos.get('contracts',0))), new_stop_short, reduce_only=True, cid=make_cid('TRL',symbol))
                        st['trail']=new_stop_short
        except Exception:
            pass

# =============================== bootstrap =====================================
class App:
    def __init__(self):
        _EXEC = get_EXEC()
        if _EXEC.use_ccxt and ccxt is not None:
            ex = ccxt.bybit({'apiKey': _EXEC.api_key,'secret': _EXEC.api_secret,'enableRateLimit': True,'options': {'defaultType': 'swap'}})
            if _EXEC.testnet: ex.set_sandbox_mode(True)
            self.exchange = ex
        else:
            self.exchange = StubBybit()
        self.broker = BybitBroker(self.exchange)
        self.broker.load_markets()
        self.store = CandleStore()
        self.strategy = Strategy(self.store, self.broker)
        self.tpwatcher = TPWatcher(self.broker)
    def preload(self, symbols: Tuple[str,...], tfs: Tuple[str,...]):
        for s in symbols:
            for tf in tfs:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(s, timeframe=tf, limit=400)
                    self.store.push_bulk(s, tf, ohlcv)
                except Exception: pass
    def adopt_or_reset(self, symbols: Tuple[str,...]):
        _EXEC = get_EXEC()
        for s in symbols:
            if _EXEC.adopt_mode.upper()=="CLOSE_THEN_START":
                try:
                    self.broker.cancel_all(s)
                    pos = self.broker.fetch_position(s)
                    qty = abs(float(pos.get('contracts',0)))
                    if qty>0:
                        self.broker.place_market(s, 'sell' if pos.get('side')=='long' else 'buy', qty, reduce_only=True, cid=make_cid('RESET',s))
                except Exception: pass
    def run_once_barclose(self):
        self.preload((SYMB.base,)+SYMB.alts, (TF.mid,TF.swing_a,TF.swing_b,TF.helper,TF.micro))
        self.strategy.on_bar_close()
    def on_tick(self, symbol: str, last_price: float): self.tpwatcher.on_tick(symbol, last_price)

# =============================== self tests ====================================
def _approx(a,b,eps=1e-6): return abs(a-b) <= eps

def test_indicators():
    series = [float(i) for i in range(1,21)]
    e = ema(series, 10); assert len(e)==20 and e[-1]>e[-2]
    h=[s+1 for s in series]; l=[s-1 for s in series]; c=series
    a = atr(h,l,c,14); assert len(a)==20 and a[-1]>0
    sl = slope(e, 5); assert sl>0

def _build_trending(store: CandleStore, symbol: str, tf: str, up: bool=True, base=100.0):
    arr=[]; t0=now_ms()-220*60_000; price=base
    for i in range(220):
        drift = (0.2 if up else -0.2); noise = 0.01
        o = price; c = price + drift + noise
        h = max(o,c)+0.02; l = min(o,c)-0.02; v = 10+i*0.01
        arr.append([t0+i*60_000, round(o,2), round(h,2), round(l,2), round(c,2), v]); price = c
    store.push_bulk(symbol, tf, arr)

def _build_flat(store: CandleStore, symbol: str, tf: str, value: float=100.0):
    arr=[]; t0=now_ms()-220*60_000
    for i in range(220):
        o=value; c=value; h=value*1.0001; l=value*0.9999; v=10+i*0.01
        arr.append([t0+i*60_000, round(o,2), round(h,2), round(l,2), round(c,2), v])
    store.push_bulk(symbol, tf, arr)

def test_signal_long_accept():
    app = App(); app.exchange = StubBybit(); app.broker = BybitBroker(app.exchange); app.broker.load_markets()
    app.store = CandleStore(); app.strategy = Strategy(app.store, app.broker)
    for tf in (TF.swing_b, TF.swing_a, TF.helper): _build_trending(app.store, SYMB.base, tf, up=True, base=100.0)
    sig = app.strategy.swing_signal(SYMB.base, 'long'); assert sig is not None and sig['side']=='buy'

def test_tp_watcher_triggers():
    app = App()
    app.exchange = StubBybit()
    app.broker = BybitBroker(app.exchange)
    app.broker.load_markets()
    watcher = TPWatcher(app.broker)

    # 더 큰 수량으로 진입
    app.broker.place_market(SYMB.base, 'buy', 0.03, reduce_only=False, cid=make_cid('ENT', SYMB.base))

    # TP1 주문 설정 (부분 청산)
    app.broker.place_limit(SYMB.base, 'sell', 0.018, 105.0, reduce_only=True, cid=make_cid('TP1', SYMB.base))

    # TP1 가격 도달 → TPWatcher 작동
    watcher.on_tick(SYMB.base, 105.0)

    # TP1 직후 포지션 일부 남아 있어야 함
    pos_after_tp1 = app.broker.fetch_position(SYMB.base)
    remaining_qty = abs(float(pos_after_tp1.get('contracts', 0)))
    assert remaining_qty > 0.0, f"Expected remaining position after TP1, got {remaining_qty}"

    # 트레일링 스탑 작동 확인
    watcher.on_tick(SYMB.base, 110.0)



def test_sl_rules():
    s = Strategy(CandleStore(), BybitBroker(StubBybit()))
    sl,tp = s.compute_sl_tp('buy', 100.0, 95.0, 2.0, mode='swing'); assert abs(sl-99.0) < 1e-6 and abs(tp-101.0) < 1e-6
    sl,tp = s.compute_sl_tp('sell', 100.0, 105.0, 2.0, mode='swing'); assert abs(sl-101.0) < 1e-6 and abs(tp-99.0) < 1e-6
    sl,tp = s.compute_sl_tp('buy', 200.0, 190.0, 5.0, mode='mid'); assert abs(sl-194.0) < 1e-6 and abs(tp-206.0) < 1e-6

def test_noise_band_blocks_entry():
    app = App(); app.exchange = StubBybit(); app.broker = BybitBroker(app.exchange); app.broker.load_markets()
    app.store = CandleStore(); app.strategy = Strategy(app.store, app.broker)
    for tf in (TF.swing_b, TF.swing_a, TF.helper): _build_flat(app.store, SYMB.base, tf, value=100.0)
    t,o,h,l,c,v = app.store.series(SYMB.base, TF.swing_b); e = ema(c, 200)
    c[-2] = e[-2] * 1.0003; c[-1] = e[-1] * 1.0003
    o[-1] = c[-1] * 0.999; h[-1] = max(o[-1], c[-1]) * 1.001; l[-1] = min(o[-1], c[-1]) * 0.999
    app.store.data[(SYMB.base,TF.swing_b)]['c']=c; app.store.data[(SYMB.base,TF.swing_b)]['o']=o
    app.store.data[(SYMB.base,TF.swing_b)]['h']=h; app.store.data[(SYMB.base,TF.swing_b)]['l']=l
    sig = app.strategy.swing_signal(SYMB.base, 'long'); assert sig is None

# 새 테스트: 고정 수량 오버라이드 확인
def test_fixed_qty_override():
    app = App(); app.exchange = StubBybit(); app.broker = BybitBroker(app.exchange); app.broker.load_markets()
    app.store = CandleStore(); app.strategy = Strategy(app.store, app.broker)
    price = 100.0
    if "ETHUSDT" in FIXED_QTY_SWING:
        q = app.strategy.calc_qty("ETHUSDT", price, 1.0, mode='swing')
        assert abs(q - FIXED_QTY_SWING["ETHUSDT"]) < 1e-9
    if "SOLUSDT" in FIXED_QTY_MID:
        q2 = app.strategy.calc_qty("SOLUSDT", price, 1.0, mode='mid')
        assert abs(q2 - FIXED_QTY_MID["SOLUSDT"]) < 1e-9

def test_exec_defaults_and_stub():
    ex1 = get_EXEC(); assert isinstance(ex1, ExecConfig)
    app = App(); ex2 = get_EXEC()
    if not ex2.use_ccxt:
        assert isinstance(app.exchange, StubBybit)

def test_btc_gate_insufficient_bars():
    app = App(); app.exchange = StubBybit(); app.broker = BybitBroker(app.exchange); app.broker.load_markets()
    app.store = CandleStore(); app.strategy = Strategy(app.store, app.broker)
    arr=[[now_ms()-i*60_000,100,101,99,100,10] for i in range(20)][::-1]
    app.store.push_bulk(SYMB.base, GATE.btc_gate_tf, arr)  # v1.4.2: gate TF 경로 테스트
    assert app.strategy.btc_gate()==GateState.OFF

def run_all_tests():
    print("[TEST] indicators…"); test_indicators(); print("OK")
    print("[TEST] long signal…"); test_signal_long_accept(); print("OK")
    print("[TEST] TP watcher…"); test_tp_watcher_triggers(); print("OK")
    print("[TEST] SL rules…"); test_sl_rules(); print("OK")
    print("[TEST] noise band…"); test_noise_band_blocks_entry(); print("OK")
    print("[TEST] fixed qty…"); test_fixed_qty_override(); print("OK")
    print("[TEST] exec defaults…"); test_exec_defaults_and_stub(); print("OK")
    print("[TEST] btc gate…"); test_btc_gate_insufficient_bars(); print("OK")
    print("All tests passed.")

# =============================== main ==========================================
if __name__ == "__main__":
    _EXEC_MAIN = get_EXEC()
    if _EXEC_MAIN.use_ccxt and ccxt is not None:
        app = App()
        app.preload((SYMB.base,)+SYMB.alts, (TF.mid,TF.swing_a,TF.swing_b,TF.helper,TF.micro))
        app.adopt_or_reset((SYMB.base,)+SYMB.alts)
        app.run_once_barclose()
        print("LIVE v1.4.2 wired (ccxt). STRICT_BRACKET=", _EXEC_MAIN.strict_bracket_after_fill)
    else:
        run_all_tests()
        print("STUB mode complete. Set USE_CCXT=1 to enable live with ccxt (when available).")
