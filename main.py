1  # -*- coding: utf-8 -*-
2  # MP Bot v55 – Hybrid Momentum + Mean Reversion
3  # MOCK + LIVE (KuCoin spot, endast LONG i LIVE)
4  # Live-PnL baserad på riktiga fills från KuCoin (funds + fee)
5  # TESTBUY fixad – fungerar via knappar + kommando
6  # Close_all SELL fixad – korrekt KuCoin rounding
9  import os
10 import io
11 import csv
12 import json
13 import time
14 import hmac
15 import uuid
16 import base64
17 import hashlib
18 import asyncio
19 from dataclasses import dataclass, field
20 from typing import Dict, List, Optional
21 from datetime import datetime, timezone, timedelta
22  
23 import httpx
24 from fastapi import FastAPI, Request
25 from fastapi.responses import JSONResponse, PlainTextResponse
26 from pydantic import BaseModel
27 from telegram import (
28     Update,
29     KeyboardButton,
30     ReplyKeyboardMarkup,
31     InlineKeyboardButton,
32     InlineKeyboardMarkup,
33 )
34 from telegram.ext import Application, CommandHandler, CallbackQueryHandler
35 import logging
36 from decimal import Decimal, ROUND_DOWN   # <—— NY IMPORT (sell-fix)
37  
38 # --------------------------------------------------
39 # LOGGING
40 # --------------------------------------------------
41 logging.basicConfig(
42     level=logging.INFO,
43     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
44 )
45 logger = logging.getLogger("mp_bot_v55")
46  
47 # --------------------------------------------------
48 # ENV
49 # --------------------------------------------------
50 BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
51 if not BOT_TOKEN:
52     raise RuntimeError("Saknar TELEGRAM_BOT_TOKEN/BOT_TOKEN i env.")
53  
54 WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "").rstrip("/")
55  
56 DEFAULT_SYMBOLS = (
57     os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,ADA-USDT,LINK-USDT,XRP-USDT")
58     .replace(" ", "")
59 ).split(",")
60 DEFAULT_TFS = (os.getenv("TIMEFRAMES", "3m").replace(" ", "")).split(",")
61  
62 MOCK_SIZE_USDT = float(os.getenv("MOCK_SIZE_USDT", "10"))
63 FEE_PER_SIDE = float(os.getenv("FEE_PER_SIDE", "0.001"))  # 0.1% per sida
64 MAX_OPEN_POS = int(os.getenv("MAX_POS", "4"))
65  
66 TP_PCT = float(os.getenv("TP_PCT", "0.30"))       # take-profit i %
67 SL_PCT = float(os.getenv("SL_PCT", "0.50"))       # stop-loss i %
68 TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "2.0"))
69 TRAIL_PCT = float(os.getenv("TRAIL_PCT", "2.5"))
70  
71 ENTRY_THRESHOLD = float(os.getenv("ENTRY_THRESHOLD", "0.55"))
72 ALLOW_SHORTS_DEFAULT = (
73     os.getenv("ALLOW_SHORTS", "false").lower() in ("1", "true", "on", "yes")
74 )
75  
76 MR_DEV_PCT = float(os.getenv("MR_DEV_PCT", "1.20"))
77 TREND_SLOPE_MIN = float(os.getenv("TREND_SLOPE_MIN", "0.20"))
78 RANGE_ATR_MAX = float(os.getenv("RANGE_ATR_MAX", "0.80"))
79  
80 LOSS_GUARD_ON_DEFAULT = True
81 LOSS_GUARD_N_DEFAULT = int(os.getenv("LOSS_GUARD_N", "2"))
82 LOSS_GUARD_PAUSE_MIN_DEFAULT = int(os.getenv("LOSS_GUARD_PAUSE_MIN", "15"))
83  
84 # KuCoin private API
85 KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "")
86 KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "")
87 KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "")
88 KUCOIN_API_KEY_VERSION = os.getenv("KUCOIN_API_KEY_VERSION", "3")
89 KUCOIN_BASE_URL = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")
90  
91 KUCOIN_KLINES_URL = "https://api.kucoin.com/api/v1/market/candles"
92 TF_MAP = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min"}
93  
94 logger.info(
95     "[KUCOIN CREDS] key_len=%d, secret_len=%d, passphrase_len=%d, version='%s'",
96     len(KUCOIN_API_KEY or ""),
97     len(KUCOIN_API_SECRET or ""),
98     len(KUCOIN_API_PASSPHRASE or ""),
99     KUCOIN_API_KEY_VERSION,
100 )
101  
102 # --------------------------------------------------
103 # STATE
104 # --------------------------------------------------
105 @dataclass
106 class Position:
107     side: str
108     entry_price: float
109     qty: float
110     opened_at: datetime
111     high_water: float
112     low_water: float
113     trailing: bool = False
114     regime: str = "trend"
115     reason: str = "MOMO"
116     usd_in: float = 0.0
117     fee_in: float = 0.0
118     entry_order_id: Optional[str] = None
119  
120  
121 @dataclass
122 class SymState:
123     pos: Optional[Position] = None
124     realized_pnl: float = 0.0
125     trades_log: List[Dict] = field(default_factory=list)
126  
127  
128 @dataclass
129 class EngineState:
130     engine_on: bool = False
131     symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
132     tfs: List[str] = field(default_factory=lambda: DEFAULT_TFS.copy())
133     per_sym: Dict[str, SymState] = field(default_factory=dict)
134  
135     mock_size: float = MOCK_SIZE_USDT
136     fee_side: float = FEE_PER_SIDE
137     threshold: float = ENTRY_THRESHOLD
138     allow_shorts: bool = ALLOW_SHORTS_DEFAULT
139  
140     tp_pct: float = TP_PCT
141     sl_pct: float = SL_PCT
142     trail_start_pct: float = TRAIL_START_PCT
143     trail_pct: float = TRAIL_PCT
144     max_pos: int = MAX_OPEN_POS
145  
146     mr_on: bool = True
147     regime_auto: bool = True
148     mr_dev_pct: float = MR_DEV_PCT
149     trend_slope_min: float = TREND_SLOPE_MIN
150     range_atr_max: float = RANGE_ATR_MAX
151  
152     trade_mode: str = "mock"
153  
154     loss_guard_on: bool = LOSS_GUARD_ON_DEFAULT
155     loss_guard_n: int = LOSS_GUARD_N_DEFAULT
156     loss_guard_pause_min: int = LOSS_GUARD_PAUSE_MIN_DEFAULT
157     loss_streak: int = 0
158     paused_until: Optional[datetime] = None
159  
160     chat_id: Optional[int] = None
161  
162  
163 STATE = EngineState()
164 for s in STATE.symbols:
165     STATE.per_sym[s] = SymState()
166  
167 LAST_LIVE_ENTRY_INFO: Dict[str, dict] = {}
168 LAST_LIVE_EXIT_INFO: Dict[str, dict] = {}
169  
170 # --------------------------------------------------
171 # Helpers
172 # --------------------------------------------------
173 def ensure_dir(path: str) -> None:
174     d = os.path.dirname(path)
175     if d and not os.path.exists(d):
176         os.makedirs(d, exist_ok=True)
177  
178  
179 def log_mock(row: Dict) -> None:
180     fname = "mock_trade_log.csv"
181     new = not os.path.exists(fname)
182     ensure_dir(fname)
183     with open(fname, "a", newline="", encoding="utf-8") as f:
184         w = csv.DictWriter(
185             f,
186             fieldnames=[
187                 "time", "symbol", "action", "side", "price", "qty",
188                 "gross", "fee_in", "fee_out", "net", "info",
189             ],
190         )
191         if new:
192             w.writeheader()
193         w.writerow(row)
194  
195  
196 def log_real(row: Dict) -> None:
197     fname = "real_trade_log.csv"
198     new = not os.path.exists(fname)
199     ensure_dir(fname)
200     with open(fname, "a", newline="", encoding="utf-8") as f:
201         w = csv.DictWriter(
202             f,
203             fieldnames=[
204                 "time", "symbol", "action", "side", "price", "qty",
205                 "gross", "fee_in", "fee_out", "net", "info",
206             ],
207         )
208         if new:
209             w.writeheader()
210         w.writerow(row)
211 
212 
213 def kucoin_creds_ok() -> bool:
214     return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)
215 
216 
217 async def kucoin_private_request(
218     method: str,
219     path: str,
220     body: Optional[dict] = None,
221 ) -> Optional[dict]:
222     """
223     Enkel KuCoin private-request (V3) för Spot.
224     """
225     if not kucoin_creds_ok():
226         return None
227 
228     body = body or {}
229     body_str = json.dumps(body, separators=(",", ":")) if method.upper() != "GET" else ""
230     ts = str(int(time.time() * 1000))
231 
232     prehash = ts + method.upper() + path + body_str
233     sign = base64.b64encode(
234         hmac.new(
235             KUCOIN_API_SECRET.encode("utf-8"),
236             prehash.encode("utf-8"),
237             hashlib.sha256,
238         ).digest()
239     ).decode("utf-8")
240 
241     passphrase_hashed = base64.b64encode(
242         hmac.new(
243             KUCOIN_API_SECRET.encode("utf-8"),
244             KUCOIN_API_PASSPHRASE.encode("utf-8"),
245             hashlib.sha256,
246         ).digest()
247     ).decode("utf-8")
248 
249     headers = {
250         "KC-API-KEY": KUCOIN_API_KEY,
251         "KC-API-SIGN": sign,
252         "KC-API-TIMESTAMP": ts,
253         "KC-API-PASSPHRASE": passphrase_hashed,
254         "KC-API-KEY-VERSION": KUCOIN_API_KEY_VERSION,
255         "Content-Type": "application/json",
256     }
257 
258     try:
259         async with httpx.AsyncClient(base_url=KUCOIN_BASE_URL, timeout=10) as client:
260             resp = await client.request(
261                 method.upper(), path, headers=headers, content=body_str or None
262             )
263     except Exception as e:
264         return {
265             "_http_status": None,
266             "code": None,
267             "msg": f"Request error: {e}",
268         }
269 
270     try:
271         data = resp.json()
272     except Exception:
273         data = {"raw": resp.text or ""}
274 
275     data["_http_status"] = resp.status_code
276     return data
277 
278 
279 async def kucoin_get_fills_for_order(order_id: str) -> Optional[dict]:
280     """
281     Hämtar fills för en spot-order (TRADES).
282     """
283     if not kucoin_creds_ok():
284         return None
285 
286     path = f"/api/v1/fills?orderId={order_id}&tradeType=TRADE"
287     fills: List[dict] = []
288 
289     for _ in range(6):
290         data = await kucoin_private_request("GET", path)
291         if not data:
292             await asyncio.sleep(0.2)
293             continue
294 
295         status = data.get("_http_status")
296         code = data.get("code")
297         if status != 200 or code != "200000":
298             await asyncio.sleep(0.2)
299             continue
300 
301         raw = data.get("data")
302         if isinstance(raw, list):
303             fills = raw
304         elif isinstance(raw, dict) and "items" in raw:
305             fills = raw["items"]
306         else:
307             fills = []
308 
309         if fills:
310             break
311 
312         await asyncio.sleep(0.2)
313 
314     if not fills:
315         return None
316 
317     total_size = 0.0
318     total_funds = 0.0
319     total_fee = 0.0
320 
321     for f in fills:
322         try:
323             total_size += float(f.get("size", 0) or 0)
324             total_funds += float(f.get("funds", 0) or 0)
325             total_fee += float(f.get("fee", 0) or 0)
326         except Exception:
327             continue
328 
329     if total_size <= 0 or total_funds <= 0:
330         return None
331 
332     avg_price = total_funds / total_size
333     return {
334         "orderId": order_id,
335         "size": total_size,
336         "funds": total_funds,
337         "fee": total_fee,
338         "avg_price": avg_price,
339     }
340 
341 
342 async def kucoin_place_market_order(symbol: str, side: str, amount: float):
343     """
344     MARKET-buy/sell med korrekt KuCoin-rundning.
345     side=buy  -> amount = USDT (funds)
346     side=sell -> amount = qty  (size)
347     """
348     if not kucoin_creds_ok():
349         return False, "Inga KuCoin-creds satta."
350 
351     side_l = side.lower()
352     body = {
353         "clientOid": str(uuid.uuid4()),
354         "side": side_l,
355         "symbol": symbol,
356         "type": "market",
357     }
358 
359     # 🔥 FIX: korrekt KuCoin-rounding
360     def _round_qty(q):
361         return str(Decimal(str(q)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
362 
363     if side_l == "buy":
364         body["funds"] = f"{amount:.2f}"
365     else:
366         body["size"] = _round_qty(amount)
367 
368     data = await kucoin_private_request("POST", "/api/v1/orders", body)
369     if not data:
370         return False, "Inget svar från KuCoin (data=None)."
371 
372     status = data.get("_http_status")
373     code = data.get("code")
374     msg = data.get("msg") or data.get("message") or data.get("raw") or ""
375 
376     if status != 200 or code != "200000":
377         return False, f"HTTP={status}, code={code}, msg={msg}"
378 
379     od = data.get("data") or {}
380     order_id = od.get("orderId") or od.get("id")
381     if not order_id:
382         return False, "OrderId saknas i svar."
383 
384     fills_info = await kucoin_get_fills_for_order(order_id)
385     if fills_info:
386         if side_l == "buy":
387             LAST_LIVE_ENTRY_INFO[symbol] = fills_info
388         else:
389             LAST_LIVE_EXIT_INFO[symbol] = fills_info
390 
391     return True, ""
392 
393 
394 # --------------------------------------------------
395 # Data & Indicators
396 # --------------------------------------------------
397 async def get_klines(symbol: str, tf: str, limit: int = 100):
398     tf_api = TF_MAP.get(tf, tf)
399     params = {"symbol": symbol, "type": tf_api}
400     async with httpx.AsyncClient(timeout=10) as client:
401         r = await client.get(KUCOIN_KLINES_URL, params=params)
402         r.raise_for_status()
403         data = r.json()["data"]
404     return data[::-1][:limit]
405 
406 
407 def ema(series: List[float], period: int) -> List[float]:
408     if not series or period <= 1:
409         return series[:]
410     k = 2.0 / (period + 1.0)
411     out = []
412     val = series[0]
413     for x in series:
414         val = (x - val) * k + val
415         out.append(val)
416     return out
417 
418 
419 def rsi(closes: List[float], period: int = 14) -> List[float]:
420     if len(closes) < period + 1:
421         return [50.0] * len(closes)
422     gains = []
423     losses = []
424     for i in range(1, len(closes)):
425         ch = closes[i] - closes[i - 1]
426         gains.append(max(ch, 0.0))
427         losses.append(-min(ch, 0.0))
428     avg_gain = sum(gains[:period]) / period
429     avg_loss = sum(losses[:period]) / period
430     out = [50.0] * period
431     for i in range(period, len(gains)):
432         avg_gain = (avg_gain * (period - 1) + gains[i]) / period
433         avg_loss = (avg_loss * (period - 1) + losses[i]) / period
434         if avg_loss == 0:
435             rs = 999.0
436         else:
437             rs = avg_gain / avg_loss
438         val = 100.0 - (100.0 / (1.0 + rs))
439         out.append(val)
440     return [50.0] + out
441 
442 
443 def compute_features(candles):
444     closes = [float(c[2]) for c in candles]
445     highs = [float(c[3]) for c in candles]
446     lows = [float(c[4]) for c in candles]
447     if len(closes) < 40:
448         last = closes[-1]
449         return {
450             "close": last,
451             "ema20": last,
452             "ema50": last,
453             "rsi": 50.0,
454             "mom": 0.0,
455             "atrp": 0.2,
456             "trend_slope": 0.0,
457         }
458 
459     ema20_series = ema(closes, 20)
460     ema50_series = ema(closes, 50)
461     ema20 = ema20_series[-1]
462     ema50 = ema50_series[-1]
463 
464     mom = (closes[-1] - closes[-6]) / (closes[-6] or 1.0) * 100.0
465     rsi_last = rsi(closes, 14)[-1]
466 
467     trs = []
468     for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:]):
469         tr = (h - l) / (c or 1.0) * 100.0
470         trs.append(tr)
471     atrp = sum(trs) / len(trs) if trs else 0.2
472 
473     if len(ema20_series) > 6 and ema20_series[-6] != 0:
474         trend_slope = (ema20_series[-1] - ema20_series[-6]) / ema20_series[-6] * 100.0
475     else:
476         trend_slope = 0.0
477 
478     return {
479         "close": closes[-1],
480         "ema20": ema20,
481         "ema50": ema50,
482         "rsi": rsi_last,
483         "mom": mom,
484         "atrp": atrp,
485         "trend_slope": trend_slope,
486     }
487 
488 
489 def momentum_score(feats) -> float:
490     ema20 = feats["ema20"]
491     ema50 = feats["ema50"]
492     mom = feats["mom"]
493     rsi_val = feats["rsi"]
494 
495     if ema20 > ema50:
496         trend = 1.0
497     elif ema20 < ema50:
498         trend = -1.0
499     else:
500         trend = 0.0
501 
502     rsi_dev = (rsi_val - 50.0) / 10.0
503     score = trend * (abs(mom) / 0.1) + rsi_dev
504     if mom < 0:
505         score = -abs(score)
506     else:
507         score = abs(score)
508     if trend < 0:
509         score = -score
510     return score
511 
512 
513 def decide_regime(feats) -> str:
514     if not STATE.regime_auto:
515         return "trend"
516 
517     atrp = feats.get("atrp", 0.0)
518     slope = abs(feats.get("trend_slope", 0.0))
519 
520     if atrp >= STATE.range_atr_max and slope >= STATE.trend_slope_min:
521         return "trend"
522     if atrp <= STATE.range_atr_max and slope < STATE.trend_slope_min:
523         return "range"
524     return "trend"
525 
526 
527 # --------------------------------------------------
528 # Trading helpers
529 # --------------------------------------------------
530 def _fee(amount_usdt: float) -> float:
531     return amount_usdt * STATE.fee_side
532 
533 
534 def _log_trade(row: Dict) -> None:
535     if STATE.trade_mode == "live":
536         log_real(row)
537     else:
538         log_mock(row)
539 
540 
541 def open_position(
542     sym: str,
543     side: str,
544     price: float,
545     st: SymState,
546     regime: str,
547     reason: str,
548 ):
549     """
550     Öppna position — MOCK eller LIVE (med riktiga fills).
551     """
552     if STATE.trade_mode == "live":
553         info = LAST_LIVE_ENTRY_INFO.pop(sym, None)
554         if info:
555             entry_price = float(info.get("avg_price") or price)
556             qty = float(info.get("size", 0.0) or 0.0)
557             usd_in = float(info.get("funds", 0.0) or 0.0)
558             fee_in = float(info.get("fee", 0.0) or 0.0)
559             entry_order_id = info.get("orderId")
560         else:
561             entry_price = price
562             qty = STATE.mock_size / price
563             usd_in = qty * entry_price
564             fee_in = _fee(usd_in)
565             entry_order_id = None
566     else:
567         entry_price = price
568         qty = STATE.mock_size / price
569         usd_in = qty * entry_price
570         fee_in = _fee(usd_in)
571         entry_order_id = None
572 
573     st.pos = Position(
574         side=side,
575         entry_price=entry_price,
576         qty=qty,
577         opened_at=datetime.now(timezone.utc),
578         high_water=entry_price,
579         low_water=entry_price,
580         trailing=False,
581         regime=regime,
582         reason=reason,
583         usd_in=usd_in,
584         fee_in=fee_in,
585         entry_order_id=entry_order_id,
586     )
587 
588     _log_trade(
589         {
590             "time": datetime.now(timezone.utc).isoformat(),
591             "symbol": sym,
592             "action": "ENTRY",
593             "side": side,
594             "price": round(entry_price, 6),
595             "qty": round(qty, 8),
596             "gross": "",
597             "fee_in": round(fee_in, 6),
598             "fee_out": "",
599             "net": "",
600             "info": f"size_usdt={STATE.mock_size};regime={regime};reason={reason};mode={STATE.trade c_mode}",
601         }
602     )
603 
604 
605 def close_position(
606     sym: str,
607     st: SymState,
608     reason: str,
609     approx_price: Optional[float] = None,
610 ) -> float:
611     """
612     Stänger position — MOCK eller LIVE (hämtar riktiga fills).
613     Returnerar net PnL.
614     """
615     if not st.pos:
616         return 0.0
617 
618     pos = st.pos
619 
620     if STATE.trade_mode == "live":
621         info = LAST_LIVE_EXIT_INFO.pop(sym, None)
622         if info:
623             usd_in = pos.usd_in
624             fee_in = pos.fee_in
625             usd_out = float(info.get("funds", 0.0) or 0.0)
626             fee_out = float(info.get("fee", 0.0) or 0.0)
627             exit_price = float(info.get("avg_price") or approx_price or pos.entry_price)
628             gross = usd_out - usd_in
629             net = gross - fee_in - fee_out
630         else:
631             price = approx_price or pos.entry_price
632             usd_in = pos.qty * pos.entry_price
633             usd_out = pos.qty * price
634             fee_in = _fee(usd_in)
635             fee_out = _fee(usd_out)
636             if pos.side == "LONG":
637                 gross = pos.qty * (price - pos.entry_price)
638             else:
639                 gross = pos.qty * (pos.entry_price - price)
640             net = gross - fee_in - fee_out
641             exit_price = price
642 
643     else:
644         price = approx_price or pos.entry_price
645         usd_in = pos.qty * pos.entry_price
646         usd_out = pos.qty * price
647         fee_in = _fee(usd_in)
648         fee_out = _fee(usd_out)
649         if pos.side == "LONG":
650             gross = pos.qty * (price - pos.entry_price)
651         else:
652             gross = pos.qty * (pos.entry_price - price)
653         net = gross - fee_in - fee_out
654         exit_price = price
655 
656     st.realized_pnl += net
657     st.trades_log.append(
658         {
659             "time": datetime.now(timezone.utc).isoformat(),
660             "symbol": sym,
661             "side": pos.side,
662             "entry": pos.entry_price,
663             "exit": exit_price,
664             "gross": round(gross, 6),
665             "fee_in": round(fee_in, 6),
666             "fee_out": round(fee_out, 6),
667             "net": round(net, 6),
668             "reason": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
669         }
670     )
671 
672     _log_trade(
673         {
674             "time": datetime.now(timezone.utc).isoformat(),
675             "symbol": sym,
676             "action": "EXIT",
677             "side": pos.side,
678             "price": round(exit_price, 6),
679             "qty": round(pos.qty, 8),
680             "gross": round(gross, 6),
681             "fee_in": round(fee_in, 6),
682             "fee_out": round(fee_out, 6),
683             "net": round(net, 6),
684             "info": f"{reason};regime={pos.regime};src={pos.reason};mode={STATE.trade_mode}",
685         }
686     )
687 
688     st.pos = None
689     return net
690 
691 
692 # --------------------------------------------------
693 # ENGINE LOOP
694 # --------------------------------------------------
695 async def engine_loop(app: Application):
696     await asyncio.sleep(2)
697     while True:
698         try:
699             if STATE.engine_on:
700                 now = datetime.now(timezone.utc)
701 
702                 if (
703                     STATE.loss_guard_on
704                     and STATE.paused_until is not None
705                     and now < STATE.paused_until
706                 ):
707                     await asyncio.sleep(3)
708                     continue
709 
710                 # -----------------------------
711                 # ENTRY-LOGIK
712                 # -----------------------------
713                 open_syms = [s for s, st in STATE.per_sym.items() if st.pos]
714                 if len(open_syms) < STATE.max_pos:
715                     for sym in STATE.symbols:
716                         st = STATE.per_sym[sym]
717                         if st.pos:
718                             continue
719 
720                         tf = STATE.tfs[0] if STATE.tfs else "3m"
721                         try:
722                             kl = await get_klines(sym, tf, limit=80)
723                         except Exception:
724                             continue
725 
726                         feats = compute_features(kl)
727                         price = feats["close"]
728                         regime = decide_regime(feats)
729 
730                         allow_shorts_effective = (
731                             STATE.allow_shorts if STATE.trade_mode == "mock" else False
732                         )
733 
734                         # -------- TREND (MOMO) --------
735                         if regime == "trend":
736                             score = momentum_score(feats)
737 
738                             if score > STATE.threshold:
739                                 # LIVE BUY
740                                 if STATE.trade_mode == "live":
741                                     ok, err = await kucoin_place_market_order(
742                                         sym, "buy", STATE.mock_size
743                                     )
744                                     if not ok:
745                                         if STATE.chat_id:
746                                             await app.bot.send_message(
747                                                 STATE.chat_id,
748                                                 f"⚠️ LIVE BUY {sym} misslyckades (MOMO).\n{err}",
749                                             )
750                                         continue
751 
752                                 open_position(
753                                     sym, "LONG", price, st,
754                                     regime="trend", reason="MOMO"
755                                 )
756                                 open_syms.append(sym)
757 
758                                 if STATE.chat_id:
759                                     await app.bot.send_message(
760                                         STATE.chat_id,
761                                         f"🟢 MOMO ENTRY {sym} LONG @ {price:.4f} "
762                                         f"| score={score:.2f} | thr={STATE.threshold:.2f} | "
763                                         f"regime=trend | mode={STATE.trade_mode}"
764                                     )
765                                 if len(open_syms) >= STATE.max_pos:
766                                     break
767 
768                             # -------- TREND SHORT (endast mock) --------
769                             elif allow_shorts_effective and score < -STATE.threshold:
770                                 open_position(
771                                     sym, "SHORT", price, st,
772                                     regime="trend", reason="MOMO"
773                                 )
774                                 open_syms.append(sym)
775                                 if STATE.chat_id:
776                                     await app.bot.send_message(
777                                         STATE.chat_id,
778                                         f"🔻 MOMO ENTRY {sym} SHORT @ {price:.4f} "
779                                         f"| score={score:.2f} | thr={STATE.threshold:.2f} | "
780                                         f"regime=trend | mode={STATE.trade_mode}"
781                                     )
782                                 if len(open_syms) >= STATE.max_pos:
783                                     break
784 
785                         # -------- MEAN REVERSION --------
786                         elif regime == "range" and STATE.mr_on:
787                             ema20 = feats["ema20"]
788                             if ema20 == 0:
789                                 continue
790 
791                             dev_pct = (price - ema20) / ema20 * 100.0
792 
793                             # MR LONG
794                             if dev_pct <= -STATE.mr_dev_pct:
795                                 if STATE.trade_mode == "live":
796                                     ok, err = await kucoin_place_market_order(
797                                         sym, "buy", STATE.mock_size
798                                     )
799                                     if not ok:
800                                         if STATE.chat_id:
801                                             await app.bot.send_message(
802                                                 STATE.chat_id,
803                                                 f"⚠️ LIVE BUY {sym} misslyckades (MR).\n{err}",
804                                             )
805                                         continue
806 
807                                 open_position(
808                                     sym, "LONG", price, st,
809                                     regime="range", reason="MR"
810                                 )
811                                 open_syms.append(sym)
812 
813                                 if STATE.chat_id:
814                                     await app.bot.send_message(
815                                         STATE.chat_id,
816                                         f"🟢 MR ENTRY {sym} LONG @ {price:.4f} "
817                                         f"| dev={dev_pct:.2f}% | regime=range | mode={STATE.trade_mode}"
818                                     )
819                                 if len(open_syms) >= STATE.max_pos:
820                                     break
821 
822                             # MR SHORT (endast mock)
823                             elif dev_pct >= STATE.mr_dev_pct and allow_shorts_effective:
824                                 open_position(
825                                     sym, "SHORT", price, st,
826                                     regime="range", reason="MR"
827                                 )
828                                 open_syms.append(sym)
829                                 if STATE.chat_id:
830                                     await app.bot.send_message(
831                                         STATE.chat_id,
832                                         f"🔻 MR ENTRY {sym} SHORT @ {price:.4f} "
833                                         f"| dev={dev_pct:.2f}% | regime=range | mode={STATE.trade_mode}"
834                                     )
835                                 if len(open_syms) >= STATE.max_pos:
836                                     break
837 
838                 # -----------------------------
839                 # POSITION-MANAGEMENT
840                 # -----------------------------
841                 for sym in STATE.symbols:
842                     st = STATE.per_sym[sym]
843                     if not st.pos:
844                         continue
845 
846                     tf = STATE.tfs[0] if STATE.tfs else "3m"
847                     try:
848                         kl = await get_klines(sym, tf, limit=5)
849                     except Exception:
850                         continue
851 
852                     feats = compute_features(kl)
853                     price = feats["close"]
854                     pos = st.pos
855 
856                     # uppdatera high/low
857                     pos.high_water = max(pos.high_water, price)
858                     pos.low_water = min(pos.low_water, price)
859 
860                     move_pct = (price - pos.entry_price) / (pos.entry_price or 1.0) * 100.0
861                     if pos.side == "SHORT":
862                         move_pct = -move_pct
863 
864                     # Starta trailing
865                     if (not pos.trailing) and move_pct >= STATE.trail_start_pct:
866                         pos.trailing = True
867                         if STATE.chat_id:
868                             await app.bot.send_message(
869                                 STATE.chat_id,
870                                 f"🔒 TRAIL ON {sym} | move≈{move_pct:.2f}% "
871                                 f"(regime={pos.regime},src={pos.reason},mode={STATE.trade_mode})",
872                             )
873 
874                     # -------- TP (ej trailing) --------
875                     if move_pct >= STATE.tp_pct and not pos.trailing:
876                         if STATE.trade_mode == "live" and pos.side == "LONG":
877                             ok, err = await kucoin_place_market_order(
878                                 sym, "sell", pos.qty
879                             )
880                             if not ok:
881                                 if STATE.chat_id:
882                                     await app.bot.send_message(
883                                         STATE.chat_id,
884                                         f"⚠️ LIVE TP-sälj {sym} misslyckades.\n{err}",
885                                     )
886                                 continue
887 
888                         net = close_position(sym, st, reason="TP", approx_price=price)
889                         if net < 0:
890                             STATE.loss_streak += 1
891                         else:
892                             STATE.loss_streak = 0
893 
894                         if STATE.chat_id:
895                             mark = "✅" if net >= 0 else "❌"
896                             await app.bot.send_message(
897                                 STATE.chat_id,
898                                 f"🎯 TP EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
899                             )
900 
901                         if (
902                             STATE.loss_guard_on
903                             and STATE.loss_streak >= STATE.loss_guard_n
904                         ):
905                             STATE.paused_until = datetime.now(timezone.utc) + timedelta(
906                                 minutes=STATE.loss_guard_pause_min
907                             )
908                             STATE.loss_streak = 0
909                             if STATE.chat_id:
910                                 pause_t = STATE.paused_until.astimezone().strftime(
911                                     "%H:%M"
912                                 )
913                                 await app.bot.send_message(
914                                     STATE.chat_id,
915                                     f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
916                                 )
917                         continue
918 
919                     # -------- TRAILING STOP --------
920                     if pos.trailing:
921                         if pos.side == "LONG":
922                             trail_stop = pos.high_water * (
923                                 1.0 - STATE.trail_pct / 100.0
924                             )
925                             if price <= trail_stop:
926                                 if STATE.trade_mode == "live":
927                                     ok, err = await kucoin_place_market_order(
928                                         sym, "sell", pos.qty
929                                     )
930                                     if not ok:
931                                         if STATE.chat_id:
932                                             await app.bot.send_message(
933                                                 STATE.chat_id,
934                                                 f"⚠️ LIVE TRAIL-sälj {sym} misslyckades.\n{err}",
935                                             )
936                                         continue
937 
938                                 net = close_position(
939                                     sym, st, reason="TRAIL", approx_price=price
940                                 )
941                                 if net < 0:
942                                     STATE.loss_streak += 1
943                                 else:
944                                     STATE.loss_streak = 0
945 
946                                 if STATE.chat_id:
947                                     mark = "✅" if net >= 0 else "❌"
948                                     await app.bot.send_message(
949                                         STATE.chat_id,
950                                         f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
951                                     )
952 
953                                 if (
954                                     STATE.loss_guard_on
955                                     and STATE.loss_streak >= STATE.loss_guard_n
956                                 ):
957                                     STATE.paused_until = datetime.now(
958                                         timezone.utc
959                                     ) + timedelta(
960                                         minutes=STATE.loss_guard_pause_min
961                                     )
962                                     STATE.loss_streak = 0
963                                     if STATE.chat_id:
964                                         pause_t = STATE.paused_until.astimezone().strftime(
965                                             "%H:%M"
966                                         )
967                                         await app.bot.send_message(
968                                             STATE.chat_id,
969                                             f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
970                                         )
971                                 continue
972 
973                         else:  # SHORT trailing (mock only)
974                             trail_stop = pos.low_water * (
975                                 1.0 + STATE.trail_pct / 100.0
976                             )
977                             if price >= trail_stop:
978                                 net = close_position(
979                                     sym, st, reason="TRAIL", approx_price=price
980                                 )
981                                 if net < 0:
982                                     STATE.loss_streak += 1
983                                 else:
984                                     STATE.loss_streak = 0
985 
986                                 if STATE.chat_id:
987                                     mark = "✅" if net >= 0 else "❌"
988                                     await app.bot.send_message(
989                                         STATE.chat_id,
990                                         f"🏁 TRAIL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
991                                     )
992 
993                                 if (
994                                     STATE.loss_guard_on
995                                     and STATE.loss_streak >= STATE.loss_guard_n
996                                 ):
997                                     STATE.paused_until = datetime.now(
998                                         timezone.utc
999                                     ) + timedelta(
1000                                         minutes=STATE.loss_guard_pause_min
1001                                     )
1002                                 continue
1003 
1004                     # -------- STOP LOSS --------
1005                     if move_pct <= -STATE.sl_pct:
1006                         if STATE.trade_mode == "live" and pos.side == "LONG":
1007                             ok, err = await kucoin_place_market_order(
1008                                 sym, "sell", pos.qty
1009                             )
1010                             if not ok:
1011                                 if STATE.chat_id:
1012                                     await app.bot.send_message(
1013                                         STATE.chat_id,
1014                                         f"⚠️ LIVE SL-sälj {sym} misslyckades.\n{err}",
1015                                     )
1016                                 continue
1017 
1018                         net = close_position(sym, st, reason="SL", approx_price=price)
1019                         if net < 0:
1020                             STATE.loss_streak += 1
1021                         else:
1022                             STATE.loss_streak = 0
1023 
1024                         if STATE.chat_id:
1025                             mark = "✅" if net >= 0 else "❌"
1026                             await app.bot.send_message(
1027                                 STATE.chat_id,
1028                                 f"⛔ SL EXIT {sym} @ {price:.4f} | Net {net:+.4f} USDT {mark}",
1029                             )
1030 
1031                         if (
1032                             STATE.loss_guard_on
1033                             and STATE.loss_streak >= STATE.loss_guard_n
1034                         ):
1035                             STATE.paused_until = datetime.now(
1036                                 timezone.utc
1037                             ) + timedelta(minutes=STATE.loss_guard_pause_min)
1038                             STATE.loss_streak = 0
1039                             if STATE.chat_id:
1040                                 pause_t = STATE.paused_until.astimezone().strftime(
1041                                     "%H:%M"
1042                                 )
1043                                 await app.bot.send_message(
1044                                     STATE.chat_id,
1045                                     f"🛑 Loss-guard: pausar nya entries till ca {pause_t}.",
1046                                 )
1047                         continue
1048 
1049             await asyncio.sleep(3)
1050 
1051         except Exception as e:
1052             logger.exception("Engine-fel: %s", e)
1053             if STATE.chat_id:
1054                 try:
1055                     await app.bot.send_message(STATE.chat_id, f"⚠️ Engine-fel: {e}")
1056                 except Exception:
1057                     pass
1058             await asyncio.sleep(5)
1059 
1060 
1061 # --------------------------------------------------
1062 # TELEGRAM UI
1063 # --------------------------------------------------
1064 def reply_kb() -> ReplyKeyboardMarkup:
1065     rows = [
1066         [KeyboardButton("/status"), KeyboardButton("/pnl")],
1067         [KeyboardButton("/engine_on"), KeyboardButton("/engine_off")],
1068         [KeyboardButton("/timeframe"), KeyboardButton("/threshold")],
1069         [KeyboardButton("/risk"), KeyboardButton("/export_csv")],
1070         [KeyboardButton("/close_all"), KeyboardButton("/reset_pnl")],
1071         [KeyboardButton("/mode"), KeyboardButton("/test_buy")],
1072     ]
1073     return ReplyKeyboardMarkup(rows, resize_keyboard=True)
1074 
1075 
1076 def status_text() -> str:
1077     total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
1078     pos_lines = []
1079 
1080     for s in STATE.symbols:
1081         st = STATE.per_sym[s]
1082         if st.pos:
1083             p = st.pos
1084             move_pct = (p.high_water - p.entry_price) / (p.entry_price or 1.0) * 100.0
1085             pos_lines.append(
1086                 f"{s}: {p.side} @ {p.entry_price:.4f} | qty {p.qty:.6f} | "
1087                 f"hi {p.high_water:.4f} | lo {p.low_water:.4f} | "
1088                 f"max_move≈{move_pct:.2f}% | regime={p.regime},src={p.reason}",
1089             )
1090 
1091     regime_line = (
1092         "Regime: AUTO (trend + mean reversion)"
1093         if STATE.regime_auto
1094         else "Regime: TREND only"
1095     )
1096 
1097     mr_line = (
1098         f"MR: {'ON' if STATE.mr_on else 'OFF'} | dev={STATE.mr_dev_pct:.2f}% | "
1099         f"range_atr_max={STATE.range_atr_max:.2f}%"
1100     )
1101 
1102     trend_line = f"Trend-slope min: {STATE.trend_slope_min:.2f}%"
1103 
1104     mode_line = (
1105         "Läge: LIVE SPOT (endast LONG)"
1106         if STATE.trade_mode == "live"
1107         else "Läge: MOCK (simulerad handel)"
1108     )
1109 
1110     if STATE.loss_guard_on:
1111         if STATE.paused_until and datetime.now(timezone.utc) < STATE.paused_until:
1112             rest = STATE.paused_until.astimezone().strftime("%H:%M")
1113             lg_line = (
1114                 f"Loss-guard: ON | N={STATE.loss_guard_n} | pause={STATE.loss_guard_pause_min}m "
1115                 f"(aktiv paus till ca {rest})"
1116             )
1117         else:
1118             lg_line = (
1119                 f"Loss-guard: ON | N={STATE.loss_guard_n} | "
1120                 f"pause={STATE.loss_guard_pause_min}m"
1121             )
1122     else:
1123         lg_line = "Loss-guard: OFF"
1124 
1125     lines = [
1126         f"Engine: {'ON ✅' if STATE.engine_on else 'OFF ⛔️'}",
1127         "Strategi: Hybrid Momentum + Mean Reversion (MOCK/LIVE)",
1128         regime_line,
1129         mr_line,
1130         trend_line,
1131         lg_line,
1132         mode_line,
1133         f"Threshold (MOMO): {STATE.threshold:.2f}",
1134         f"Timeframes: {', '.join(STATE.tfs)}",
1135         f"Symbols: {', '.join(STATE.symbols)}",
1136         f"Mock-size: {STATE.mock_size:.1f} USDT | Fee per sida (modell): {STATE.fee_side:.4%}",
1137         f"Risk: tp={STATE.tp_pct:.2f}% | sl={STATE.sl_pct:.2f}% | "
1138         f"trail_start={STATE.trail_start_pct:.2f}% | trail={STATE.trail_pct:.2f}% | "
1139         f"shorts={'ON' if STATE.allow_shorts and STATE.trade_mode=='mock' else 'OFF'} | "
1140         f"max_pos={STATE.max_pos}",
1141         f"PnL total (NET): {total:+.4f} USDT",
1142         "Positioner: " + (", ".join(pos_lines) if pos_lines else "inga"),
1143     ]
1144     return "\n".join(lines)
1145 
1146 
1147 # --------------------------------------------------
1148 # TELEGRAM Application + Commands
1149 # --------------------------------------------------
1150 tg_app = Application.builder().token(BOT_TOKEN).build()
1151 
1152 
1153 async def cmd_start(update: Update, _):
1154     STATE.chat_id = update.effective_chat.id
1155     await tg_app.bot.send_message(
1156         STATE.chat_id,
1157         "🤖 MP Bot v55 – Hybrid Momentum + Mean Reversion\n"
1158         "Standardläge: MOCK (simulerad handel)\n"
1159         "Starta engine med /engine_on\n"
1160         "Byt mellan MOCK/LIVE med /mode.\n"
1161         "Justera momentum-känslighet med /threshold 0.55 (lägre = fler trades)",
1162         reply_markup=reply_kb(),
1163     )
1164     await tg_app.bot.send_message(STATE.chat_id, status_text())
1165 
1166 
1167 async def cmd_status(update: Update, _):
1168     STATE.chat_id = update.effective_chat.id
1169     await tg_app.bot.send_message(
1170         STATE.chat_id, status_text(), reply_markup=reply_kb()
1171     )
1172 
1173 
1174 async def cmd_engine_on(update: Update, _):
1175     STATE.chat_id = update.effective_chat.id
1176     STATE.engine_on = True
1177     await tg_app.bot.send_message(
1178         STATE.chat_id,
1179         f"Engine: ON ✅ ({STATE.trade_mode.upper()})",
1180         reply_markup=reply_kb(),
1181     )
1182 
1183 
1184 async def cmd_engine_off(update: Update, _):
1185     STATE.chat_id = update.effective_chat.id
1186     STATE.engine_on = False
1187     await tg_app.bot.send_message(
1188         STATE.chat_id, "Engine: OFF ⛔️", reply_markup=reply_kb()
1189     )
1190 
1191 
1192 async def cmd_timeframe(update: Update, _):
1193     STATE.chat_id = update.effective_chat.id
1194     msg = update.message.text.strip()
1195     parts = msg.split(" ", 1)
1196     if len(parts) == 2:
1197         tfs = [x.strip() for x in parts[1].split(",") if x.strip()]
1198         if tfs:
1199             STATE.tfs = tfs
1200             await tg_app.bot.send_message(
1201                 STATE.chat_id,
1202                 f"Timeframes satta: {', '.join(STATE.tfs)}",
1203                 reply_markup=reply_kb(),
1204             )
1205             return
1206     await tg_app.bot.send_message(
1207         STATE.chat_id, "Använd: /timeframe 1m,3m,5m", reply_markup=reply_kb()
1208     )
1209 
1210 
1211 async def cmd_threshold(update: Update, _):
1212     STATE.chat_id = update.effective_chat.id
1213     toks = update.message.text.strip().split()
1214     if len(toks) == 1:
1215         await tg_app.bot.send_message(
1216             STATE.chat_id,
1217             f"Aktuellt momentum-threshold: {STATE.threshold:.2f}",
1218             reply_markup=reply_kb(),
1219         )
1220         return
1221     try:
1222         val = float(toks[1])
1223         if val <= 0:
1224             raise ValueError()
1225         STATE.threshold = val
1226         await tg_app.bot.send_message(
1227             STATE.chat_id,
1228             f"Momentum-threshold uppdaterad: {val:.2f}",
1229             reply_markup=reply_kb(),
1230         )
1231     except Exception:
1232         await tg_app.bot.send_message(
1233             STATE.chat_id,
1234             "Fel värde. Ex: /threshold 0.55",
1235             reply_markup=reply_kb(),
1236         )
1237 
1238 
1239 def _yesno(s: str) -> bool:
1240     return s.lower() in ("1", "true", "on", "yes", "ja")
1241 
1242 
1243 async def cmd_risk(update: Update, _):
1244     STATE.chat_id = update.effective_chat.id
1245     toks = update.message.text.strip().split()
1246 
1247     # /risk set key value
1248     if len(toks) == 4 and toks[0] == "/risk" and toks[1] == "set":
1249         key, val = toks[2], toks[3]
1250         try:
1251             if key == "size":
1252                 STATE.mock_size = float(val)
1253             elif key == "tp":
1254                 STATE.tp_pct = float(val)
1255             elif key == "sl":
1256                 STATE.sl_pct = float(val)
1257             elif key == "trail_start":
1258                 STATE.trail_start_pct = float(val)
1259             elif key == "trail":
1260                 STATE.trail_pct = float(val)
1261             elif key == "max_pos":
1262                 STATE.max_pos = int(val)
1263             elif key == "allow_shorts":
1264                 if STATE.trade_mode == "live":
1265                     await tg_app.bot.send_message(
1266                         STATE.chat_id,
1267                         "Shorts är inte tillåtna i LIVE-spotläge.",
1268                         reply_markup=reply_kb(),
1269                     )
1270                     return
1271                 STATE.allow_shorts = _yesno(val)
1272             elif key == "mr_on":
1273                 STATE.mr_on = _yesno(val)
1274             elif key == "mr_dev":
1275                 STATE.mr_dev_pct = float(val)
1276             elif key == "regime_auto":
1277                 STATE.regime_auto = _yesno(val)
1278             elif key == "trend_slope_min":
1279                 STATE.trend_slope_min = float(val)
1280             elif key == "range_atr_max":
1281                 STATE.range_atr_max = float(val)
1282             elif key == "loss_guard_on":
1283                 STATE.loss_guard_on = _yesno(val)
1284             elif key == "loss_guard_n":
1285                 STATE.loss_guard_n = int(val)
1286             elif key == "loss_guard_pause":
1287                 STATE.loss_guard_pause_min = int(val)
1288             else:
1289                 await tg_app.bot.send_message(
1290                     STATE.chat_id,
1291                     "Stödjer: size, tp, sl, trail_start, trail, max_pos, allow_shorts, "
1292                     "mr_on, mr_dev, regime_auto, trend_slope_min, range_atr_max, "
1293                     "loss_guard_on, loss_guard_n, loss_guard_pause",
1294                     reply_markup=reply_kb(),
1295                 )
1296                 return
1297 
1298             await tg_app.bot.send_message(
1299                 STATE.chat_id, "Risk/Regime uppdaterad.", reply_markup=reply_kb()
1300             )
1301 
1302         except Exception:
1303             await tg_app.bot.send_message(
1304                 STATE.chat_id, "Felaktigt värde.", reply_markup=reply_kb()
1305             )
1306         return
1307 
1308     # Snabb-knappar
1309     kb = InlineKeyboardMarkup(
1310         [
1311             [
1312                 InlineKeyboardButton("Size 10", callback_data="risk_size_10"),
1313                 InlineKeyboardButton("Size 30", callback_data="risk_size_30"),
1314                 InlineKeyboardButton("Size 50", callback_data="risk_size_50"),
1315             ],
1316             [
1317                 InlineKeyboardButton("TP 0.3%", callback_data="risk_tp_0.3"),
1318                 InlineKeyboardButton("TP 0.5%", callback_data="risk_tp_0.5"),
1319                 InlineKeyboardButton("TP 1.0%", callback_data="risk_tp_1.0"),
1320             ],
1321             [
1322                 InlineKeyboardButton("SL 0.5%", callback_data="risk_sl_0.5"),
1323                 InlineKeyboardButton("SL 1.0%", callback_data="risk_sl_1.0"),
1324                 InlineKeyboardButton("SL 2.0%", callback_data="risk_sl_2.0"),
1325             ],
1326             [
1327                 InlineKeyboardButton("max_pos 1", callback_data="risk_maxpos_1"),
1328                 InlineKeyboardButton("max_pos 2", callback_data="risk_maxpos_2"),
1329                 InlineKeyboardButton("max_pos 4", callback_data="risk_maxpos_4"),
1330             ],
1331         ]
1332     )
1333 
1334     text = (
1335         f"Aktuellt size per trade: {STATE.mock_size:.1f} USDT\n"
1336         f"Aktuellt TP: {STATE.tp_pct:.2f}% | SL: {STATE.sl_pct:.2f}%\n"
1337         f"Aktuellt max_pos: {STATE.max_pos}\n\n"
1338         "Välj snabb-knapp under eller använd:\n"
1339         "`/risk set <key> <value>`\n"
1340         "Ex: `/risk set size 20`"
1341     )
1342 
1343     await tg_app.bot.send_message(
1344         STATE.chat_id, text, reply_markup=kb
1345     )
1346 
1347 
1348 async def cmd_pnl(update: Update, _):
1349     STATE.chat_id = update.effective_chat.id
1350     total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
1351     lines = [f"📈 PnL total (NET): {total:+.4f} USDT"]
1352     for s in STATE.symbols:
1353         lines.append(f"• {s}: {STATE.per_sym[s].realized_pnl:+.4f} USDT")
1354     await tg_app.bot.send_message(
1355         STATE.chat_id, "\n".join(lines), reply_markup=reply_kb()
1356     )
1357 
1358 
1359 async def cmd_export_csv(update: Update, _):
1360     STATE.chat_id = update.effective_chat.id
1361 
1362     rows = [
1363         [
1364             "time",
1365             "symbol",
1366             "side",
1367             "entry",
1368             "exit",
1369             "gross",
1370             "fee_in",
1371             "fee_out",
1372             "net",
1373             "reason",
1374         ]
1375     ]
1376 
1377     for s in STATE.symbols:
1378         for r in STATE.per_sym[s].trades_log:
1379             rows.append(
1380                 [
1381                     r["time"],
1382                     r["symbol"],
1383                     r["side"],
1384                     r["entry"],
1385                     r["exit"],
1386                     r["gross"],
1387                     r["fee_in"],
1388                     r["fee_out"],
1389                     r["net"],
1390                     r["reason"],
1391                 ]
1392             )
1393 
1394     if len(rows) == 1:
1395         await tg_app.bot.send_message(
1396             STATE.chat_id, "Inga trades loggade ännu.", reply_markup=reply_kb()
1397         )
1398         return
1399 
1400     buf = io.StringIO()
1401     csv.writer(buf).writerows(rows)
1402     buf.seek(0)
1403 
1404     await tg_app.bot.send_document(
1405         STATE.chat_id,
1406         document=io.BytesIO(buf.getvalue().encode("utf-8")),
1407         filename="trades_hybrid_momo.csv",
1408         caption="Export CSV",
1409     )
1410 
1411 
1412 async def cmd_mode(update: Update, _):
1413     """
1414     /mode
1415     /mode mock
1416     /mode live
1417     /mode live JA
1418     """
1419     STATE.chat_id = update.effective_chat.id
1420     toks = update.message.text.strip().split()
1421 
1422     if len(toks) == 1:
1423         kb = InlineKeyboardMarkup(
1424             [
1425                 [
1426                     InlineKeyboardButton("MOCK", callback_data="mode_choose_mock"),
1427                     InlineKeyboardButton("LIVE", callback_data="mode_choose_live"),
1428                 ]
1429             ]
1430         )
1431         await tg_app.bot.send_message(
1432             STATE.chat_id,
1433             f"Aktuellt läge: {STATE.trade_mode.upper()}\nVälj nytt läge:",
1434             reply_markup=kb,
1435         )
1436         return
1437 
1438     target = toks[1].lower()
1439 
1440     if target == "mock":
1441         STATE.trade_mode = "mock"
1442         await tg_app.bot.send_message(
1443             STATE.chat_id,
1444             "Läge satt till MOCK (endast simulering). "
1445             "Shorts kan åter aktiveras via /risk set allow_shorts on.",
1446             reply_markup=reply_kb(),
1447         )
1448         return
1449 
1450     if target == "live":
1451         if len(toks) < 3 or toks[2].upper() != "JA":
1452             kb = InlineKeyboardMarkup(
1453                 [
1454                     [
1455                         InlineKeyboardButton(
1456                             "✅ JA, slå på LIVE",
1457                             callback_data="mode_live_yes",
1458                         ),
1459                         InlineKeyboardButton("❌ NEJ", callback_data="mode_live_no"),
1460                     ]
1461                 ]
1462             )
1463             await tg_app.bot.send_message(
1464                 STATE.chat_id,
1465                 "⚠️ Är du säker på att du vill slå på LIVE?\n"
1466                 "Detta skickar riktiga SPOT-marknadsordrar på KuCoin.",
1467                 reply_markup=kb,
1468             )
1469             return
1470 
1471         if not kucoin_creds_ok():
1472             await tg_app.bot.send_message(
1473                 STATE.chat_id,
1474                 "❌ Kan inte aktivera LIVE: saknar KUCOIN_API_KEY/SECRET/PASSPHRASE.",
1475                 reply_markup=reply_kb(),
1476             )
1477             return
1478 
1479         STATE.trade_mode = "live"
1480         STATE.allow_shorts = False
1481         await tg_app.bot.send_message(
1482             STATE.chat_id,
1483             "✅ LIVE-läge AKTIVERAT (spot, endast LONG).\n"
1484             "Engine körs med riktiga market-ordrar.",
1485             reply_markup=reply_kb(),
1486         )
1487         return
1488 
1489     await tg_app.bot.send_message(
1490         STATE.chat_id,
1491         "Använd: /mode, /mode mock, /mode live, /mode live JA",
1492         reply_markup=reply_kb(),
1493     )
1494 
1495 
1496 async def cmd_close_all(update: Update, _):
1497     STATE.chat_id = update.effective_chat.id
1498     closed = 0
1499     total_net = 0.0
1500 
1501     for sym in STATE.symbols:
1502         st = STATE.per_sym[sym]
1503         if not st.pos:
1504             continue
1505 
1506         pos = st.pos
1507         tf = STATE.tfs[0] if STATE.tfs else "3m"
1508         try:
1509             kl = await get_klines(sym, tf, limit=2)
1510             price = compute_features(kl)["close"]
1511         except Exception:
1512             price = pos.entry_price
1513 
1514         if STATE.trade_mode == "live" and pos.side == "LONG":
1515             ok, err = await kucoin_place_market_order(sym, "sell", pos.qty)
1516             if not ok:
1517                 await tg_app.bot.send_message(
1518                     STATE.chat_id,
1519                     f"⚠️ LIVE close-all SELL {sym} misslyckades.\n{err}",
1520                 )
1521                 continue
1522 
1523         net = close_position(
1524             sym,
1525             st,
1526             reason="MANUAL_CLOSE",
1527             approx_price=price,
1528         )
1529 
1530         closed += 1
1531         total_net += net
1532 
1533     await tg_app.bot.send_message(
1534         STATE.chat_id,
1535         f"Close all klart.\nStängde: {closed} positioner.\nNetto PnL: {total_net:+.4f} USDT",
1536         reply_markup=reply_kb(),
1537     )
1538 
1539 
1540 async def cmd_reset_pnl(update: Update, _):
1541     STATE.chat_id = update.effective_chat.id
1542 
1543     for s in STATE.symbols:
1544         st = STATE.per_sym[s]
1545         st.realized_pnl = 0.0
1546         st.trades_log.clear()
1547 
1548     await tg_app.bot.send_message(
1549         STATE.chat_id,
1550         "PnL återställt i RAM (loggfiler påverkas ej).",
1551         reply_markup=reply_kb(),
1552     )
1553 
1554 
1555 async def cmd_test_buy(update: Update, _):
1556     """
1557     /test_buy SYMBOL [USDT]
1558 
1559     Exempel:
1560        /test_buy BTC-USDT
1561        /test_buy ETH-USDT 15
1562 
1563     - Testköp öppnar position i valt coin
1564     - Använder live/mocksystemet
1565     - Testköp *inkluderas* i PnL och engine hanterar SL/TP/Trail
1566     """
1567 
1568     STATE.chat_id = update.effective_chat.id
1569     toks = update.message.text.strip().split()
1570 
1571     if len(toks) == 1:
1572         await tg_app.bot.send_message(
1573             STATE.chat_id,
1574             "Använd: /test_buy SYMBOL [USDT]\n"
1575             "Ex: /test_buy BTC-USDT 10",
1576             reply_markup=reply_kb(),
1577         )
1578         return
1579 
1580     symbol = toks[1].upper()
1581     if symbol not in STATE.symbols:
1582         await tg_app.bot.send_message(
1583             STATE.chat_id,
1584             f"Symbol {symbol} finns inte i listan.\n"
1585             f"Aktuella: {', '.join(STATE.symbols)}",
1586             reply_markup=reply_kb(),
1587         )
1588         return
1589 
1590     size = STATE.mock_size
1591     if len(toks) >= 3:
1592         try:
1593             size = float(toks[2])
1594         except Exception:
1595             pass
1596 
1597     mode_txt = "LIVE" if STATE.trade_mode == "live" else "MOCK"
1598 
1599     kb = InlineKeyboardMarkup(
1600         [
1601             [
1602                 InlineKeyboardButton(
1603                     f"✅ JA, köp {symbol} {size:.2f} USDT ({mode_txt})",
1604                     callback_data=f"testbuy_yes|{symbol}|{size:.4f}",
1605                 ),
1606                 InlineKeyboardButton("❌ NEJ", callback_data="testbuy_no"),
1607             ]
1608         ]
1609     )
1610 
1611     await tg_app.bot.send_message(
1612         STATE.chat_id,
1613         f"Vill du göra ett {mode_txt} test-köp i {symbol} för {size:.2f} USDT?",
1614         reply_markup=kb,
1615     )
1616 
1617 
1618 # --------------------------------------------------
1619 # CALLBACK KNAPPAR
1620 # --------------------------------------------------
1621 async def on_button(update: Update, _):
1622     query = update.callback_query
1623     data = (query.data or "").strip()
1624     chat_id = query.message.chat_id
1625     STATE.chat_id = chat_id
1626 
1627     # ----------------------------
1628     # RISK: SIZE KNAPPAR
1629     # ----------------------------
1630     if data.startswith("risk_size_"):
1631         await query.answer()
1632         try:
1633             size = float(data.split("_")[-1])
1634         except Exception:
1635             size = STATE.mock_size
1636         STATE.mock_size = size
1637         await query.edit_message_text(f"Size per trade uppdaterad till {size:.1f} USDT.")
1638         return
1639 
1640     # ----------------------------
1641     # RISK: TP KNAPPAR
1642     # ----------------------------
1643     if data.startswith("risk_tp_"):
1644         await query.answer()
1645         try:
1646             tp = float(data.split("_")[-1])
1647             STATE.tp_pct = tp
1648             await query.edit_message_text(f"TP uppdaterat till {tp:.2f}%.")
1649         except Exception:
1650             await query.edit_message_text("Fel TP.")
1651         return
1652 
1653     # ----------------------------
1654     # RISK: SL KNAPPAR
1655     # ----------------------------
1656     if data.startswith("risk_sl_"):
1657         await query.answer()
1658         try:
1659             sl = float(data.split("_")[-1])
1660             STATE.sl_pct = sl
1661             await query.edit_message_text(f"SL uppdaterat till {sl:.2f}%.")
1662         except Exception:
1663             await query.edit_message_text("Fel SL.")
1664         return
1665 
1666     # ----------------------------
1667     # RISK: MAX_POS KNAPPAR
1668     # ----------------------------
1669     if data.startswith("risk_maxpos_"):
1670         await query.answer()
1671         try:
1672             val = int(data.split("_")[-1])
1673             STATE.max_pos = val
1674             await query.edit_message_text(f"max_pos uppdaterat till {val}.")
1675         except Exception:
1676             await query.edit_message_text("Fel max_pos.")
1677         return
1678 
1679     # ----------------------------
1680     # MODE KNAPPAR
1681     # ----------------------------
1682     if data == "mode_choose_mock":
1683         await query.answer()
1684         STATE.trade_mode = "mock"
1685         await query.edit_message_text(
1686             "Läge satt till MOCK.\nShorts möjliga via: /risk set allow_shorts on"
1687         )
1688         return
1689 
1690     if data == "mode_choose_live":
1691         await query.answer()
1692         kb = InlineKeyboardMarkup(
1693             [
1694                 [
1695                     InlineKeyboardButton(
1696                         "✅ JA, slå på LIVE", callback_data="mode_live_yes"
1697                     ),
1698                     InlineKeyboardButton("❌ NEJ", callback_data="mode_live_no"),
1699                 ]
1700             ]
1701         )
1702         await query.edit_message_text(
1703             "⚠️ Är du säker på att du vill slå på LIVE?\nDetta skickar riktiga SPOT-ordrar.",
1704             reply_markup=kb,
1705         )
1706         return
1707 
1708     if data == "mode_live_no":
1709         await query.answer("Avbrutet.")
1710         await query.edit_message_text("LIVE-läge avbrutet.")
1711         return
1712 
1713     if data == "mode_live_yes":
1714         await query.answer()
1715         if not kucoin_creds_ok():
1716             await query.edit_message_text(
1717                 "❌ Kan inte aktivera LIVE — saknar KUCOIN creds."
1718             )
1719             return
1720 
1721         STATE.trade_mode = "live"
1722         STATE.allow_shorts = False
1723         await query.edit_message_text(
1724             "✅ LIVE-läge AKTIVERAT.\nSpot endast LONG.\nEngine skickar riktiga market-ordrar."
1725         )
1726         return
1727 
1728     # ----------------------------
1729     # TESTBUY KNAPPAR
1730     # ----------------------------
1731     if data == "testbuy_no":
1732         await query.answer("Avbrutet.")
1733         await query.edit_message_text("Testköp avbrutet.")
1734         return
1735 
1736     if data.startswith("testbuy_yes|"):
1737         await query.answer()
1738 
1739         parts = data.split("|")
1740         if len(parts) != 3:
1741             await query.edit_message_text("Fel format i test_buy.")
1742             return
1743 
1744         symbol = parts[1].upper()
1745         try:
1746             size = float(parts[2])
1747         except Exception:
1748             size = STATE.mock_size
1749 
1750         if symbol not in STATE.symbols:
1751             await query.edit_message_text(
1752                 f"{symbol} finns ej i listan.\nAktuella: {', '.join(STATE.symbols)}"
1753             )
1754             return
1755 
1756         tf = STATE.tfs[0] if STATE.tfs else "3m"
1757         try:
1758             kl = await get_klines(symbol, tf, limit=5)
1759             feats = compute_features(kl)
1760             price = feats["close"]
1761         except Exception:
1762             price = 0.0
1763 
1764         if STATE.trade_mode == "live":
1765             ok, err = await kucoin_place_market_order(symbol, "buy", size)
1766             if not ok:
1767                 await query.edit_message_text(
1768                     f"❌ LIVE testköp misslyckades.\n{err}"
1769                 )
1770                 return
1771 
1772         st = STATE.per_sym[symbol]
1773         open_position(
1774             symbol,
1775             "LONG",
1776             price,
1777             st,
1778             regime="trend",
1779             reason="TESTBUY",
1780         )
1781 
1782         mode_txt = "LIVE" if STATE.trade_mode == "live" else "MOCK"
1783         await query.edit_message_text(
1784             f"✅ Testköp öppnat i {symbol} @ {price:.4f} ({mode_txt}).\n"
1785             f"Positionen hanteras nu av engine (TP/SL/trail)."
1786         )
1787         return
1788 
1789     # ----------------------------
1790     # FALLBACK
1791     # ----------------------------
1792     await query.answer()
1793     try:
1794         await query.edit_message_text("Ok.")
1795     except Exception:
1796         pass
1797 
1798 
1799 # Registrera handlers
1800 tg_app.add_handler(CallbackQueryHandler(on_button))
1801 # --------------------------------------------------
1802 # REGISTRERA COMMAND HANDLERS
1803 # --------------------------------------------------
1804 tg_app.add_handler(CommandHandler("start", cmd_start))
1805 tg_app.add_handler(CommandHandler("status", cmd_status))
1806 tg_app.add_handler(CommandHandler("engine_on", cmd_engine_on))
1807 tg_app.add_handler(CommandHandler("engine_off", cmd_engine_off))
1808 tg_app.add_handler(CommandHandler("timeframe", cmd_timeframe))
1809 tg_app.add_handler(CommandHandler("threshold", cmd_threshold))
1810 tg_app.add_handler(CommandHandler("risk", cmd_risk))
1811 tg_app.add_handler(CommandHandler("pnl", cmd_pnl))
1812 tg_app.add_handler(CommandHandler("export_csv", cmd_export_csv))
1813 tg_app.add_handler(CommandHandler("mode", cmd_mode))
1814 tg_app.add_handler(CommandHandler("close_all", cmd_close_all))
1815 tg_app.add_handler(CommandHandler("reset_pnl", cmd_reset_pnl))
1816 tg_app.add_handler(CommandHandler("test_buy", cmd_test_buy))
1817 # Callback handler registrerades tidigare i del 9
1818 
1819 
1820 # --------------------------------------------------
1821 # FASTAPI – DIGITALOCEAN WEBHOOK + HEALTHCHECK
1822 # --------------------------------------------------
1823 app = FastAPI()
1824 
1825 
1826 class TgUpdate(BaseModel):
1827     update_id: int | None = None
1828 
1829 
1830 @app.on_event("startup")
1831 async def on_startup():
1832     logger.info("⬆️ Startup: init Telegram + webhook + engine")
1833 
1834     await tg_app.initialize()
1835 
1836     if WEBHOOK_BASE:
1837         url = f"{WEBHOOK_BASE}/webhook/{BOT_TOKEN}"
1838         await tg_app.bot.set_webhook(url)
1839         logger.info(f"Webhook satt till: {url}")
1840 
1841     await tg_app.start()
1842 
1843     # Starta engine-loop i bakgrunden
1844     asyncio.create_task(engine_loop(tg_app))
1845 
1846 
1847 @app.on_event("shutdown")
1848 async def on_shutdown():
1849     logger.info("⬇️ Shutdown: stoppar Telegram app")
1850     await tg_app.stop()
1851     await tg_app.shutdown()
1852 
1853 
1854 @app.get("/", response_class=PlainTextResponse)
1855 async def root():
1856     total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
1857     return (
1858         f"MP Bot v55 Hybrid-Momo RUNNING | "
1859         f"engine_on={STATE.engine_on} | mode={STATE.trade_mode} | "
1860         f"thr={STATE.threshold:.2f} | pnl_total={total:+.4f}"
1861     )
1862 
1863 
1864 @app.get("/health", response_class=JSONResponse)
1865 async def health():
1866     total = sum(STATE.per_sym[s].realized_pnl for s in STATE.symbols)
1867     return {
1868         "ok": True,
1869         "engine_on": STATE.engine_on,
1870         "threshold": STATE.threshold,
1871         "tfs": STATE.tfs,
1872         "symbols": STATE.symbols,
1873         "mode": STATE.trade_mode,
1874         "mr_on": STATE.mr_on,
1875         "regime_auto": STATE.regime_auto,
1876         "pnl_total": round(total, 6),
1877     }
1878 
1879 
1880 @app.post(f"/webhook/{BOT_TOKEN}")
1881 async def telegram_webhook(request: Request):
1882     data = await request.json()
1883     update = Update.de_json(data, tg_app.bot)
1884     await tg_app.process_update(update)
1885     return {"ok": True}
1886 
1887 # ---- SLUT ----
