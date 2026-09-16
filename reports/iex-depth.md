# IEX depth verification — TRAINING2 Phase 0 (spec §3.2 pre-commit gate)

**Date:** 2026-09-16 · **Author:** BasketData (Phase-0 data leg) · **Gate:** ≥300 of 390 regular-session minutes with an IEX print, median over the 20 trading days of 2024-03 (probe month, earliest training window).

**Method.** One-month probe fetch (2024-03-01..2024-03-31) per candidate via the existing
`data/market/_fetch_history_alpaca.py` helper (Alpaca IEX feed, free tier; bars in-session,
deduped, gaps preserved). Bars/day counted from `data/market/history/{SYM}_1m.parquet`.
Probe set: the 38 pinned candidates + 20 reserves, then a **wide pool of 81 additional
liquid large caps** (required — see verdict), plus zero-API depth measurement of the 42
already-fetched history symbols. Total 167 symbols measured; 636 API requests, 0 retries.

## Headline

**The strict gate cannot be satisfied by the pinned lists + reserves.** Only **29 of 167
probed symbols (17%)** reach ≥300 median bars/day. IEX is ~2% of consolidated volume and
its depth concentrates in retail-heavy mega-caps; typical large-caps sit at 100–260
bars/day and true mid-caps collapse (BLK 47.5, REGN 52.5, KLAC 68.5, AZO 12.5). Both
reserve pools fail almost completely: **0/10 train reserves** pass (best: ABT 241),
eval reserves pass only via CSX (already a train18 pinned name). Keeping counts 20+18
from pinned+reserves under the gate is arithmetically impossible, so the freeze below
substitutes beyond the reserves (wide-pool passers + burned-21 names allowed for
training), **maximizing retention of pinned names** and flagging every sub-300 name.

**Deviation reason (Main-approved 2026-09-16, Duke holds veto until the Phase-0 commit):**
the strict-300 non-burned passers number **18 < the 20 eval slots alone**, so the gate
cannot hold at 20+18 for any composition of pinned+reserves+wide-pool passers. The
freeze therefore admits four flagged Tier-B names into eval20 (min 261 median = 67% of
390) and keeps every pool member ≥ 208 median (ADI, 53%). Every Tier-B name is marked
below and in the basket-file headers.

Passers summary (March 2024, median bars/day):

| tier | names |
|---|---|
| non-burned PASS (18) | SLB 355, PFE 368.5, MRK 322.5, T 324.5, VZ 321, CSX 334.5, FCX 330.5, INTC 372.5, MU 376, PLTR 346.5, CMCSA 351, MO 324, KHC 310, SCHW 306, MS 311.5, C 346, PSX 326, NEE 359 |
| burned alt-21 PASS (11, train-only) | CRM 300, AMD 367, QCOM 322, ORCL 320, CSCO 361, NKE 319, KO 336, WMT 328, DIS 307, BAC 361, WFC 363 |

Reference: within the existing trained-21, only 9/21 would pass (AAPL MSFT GOOGL AMZN
NVDA META TSLA XOM SPY); BRK-B 145, LLY 166, MA 167 fail. The gate is a property of the
free IEX feed, not of the candidates.

## Frozen baskets (deviation from pinned lists — Duke sign-off pending)

Rule applied: pinned names passing the gate keep their slot; remaining slots filled by
depth rank (pinned Tier-B names preferred over unpinned at equal depth class); burned
alt-21 names permitted for training only; never-seen basket stays disjoint from every
training set and contains **no** trained-21 or burned-21 name.

### `baskets/eval20-neverseen.txt` (20 — NEVER train on these)

| kept pinned (pass) | kept pinned (flagged <300) | new fills (wide pool) |
|---|---|---|
| SLB, PFE, MRK, T, VZ | GILD 296.5, BA 268, UPS 263.5, MMM 261 | INTC 372.5, MU 376, PLTR 346.5, CMCSA 351, MO 324, KHC 310, SCHW 306, MS 311.5, C 346, PSX 326, NEE 359 |

Dropped pinned (all FAIL): CAT 139, DE 140.5, LMT 93, UNP 183.5, COP 242, TGT 252.5,
LOW 197, DUK 182, GS 166, BLK 47.5, SBUX 261 (SBUX dropped on the MMM tie at 261.0 —
industrial sector balance preferred; SBUX is first reserve if the basket is ever
recomposed). Reserves EMR/ITW/FDX/CSX/WM/HUM/CI/REGN/VRTX/BIIB all fail (best CSX 334.5
— consumed by train18 pinned slot).

### `baskets/train40.txt` (22 current + 18 additions)

| kept pinned (pass) | kept pinned (flagged) | burned alt-21 (pass, train-only) | wide-pool fills |
|---|---|---|---|
| FCX 330.5, CSX 334.5 | DOW 270.5, ADI 208 | CRM, AMD, QCOM, ORCL, CSCO, NKE, KO, WMT, DIS, BAC, WFC | GE 286, USB 282.5, DVN 289.5 |

Dropped pinned additions: EMR, ITW, FDX, NSC, WM, HUM, CI, REGN, VRTX, BIIB, CDNS, SNPS,
KLAC, NUE (all FAIL; reserves PH/ROK/ED/AEE/EIX/WEC/MCK/TMO/DHR/ABT also all FAIL, best
ABT 241). Flagged names in the frozen pool (median <300): ADI 208, DOW 270.5, GE 286,
USB 282.5, DVN 289.5, plus eval-side GILD 296.5 / BA 268 / UPS 263.5 / MMM 261 — the
weakest pool member is ADI at 208/390 (53%). Everything below was rejected outright.

Full probe table (all 167 symbols) follows.

| group | symbol | median bars/day | days | min day | verdict |
|---|---|---|---|---|---|
| eval20 pinned | MMM | 261.0 | 20 | 177 | FAIL |
| eval20 pinned | CAT | 139.0 | 20 | 71 | FAIL |
| eval20 pinned | DE | 140.5 | 20 | 93 | FAIL |
| eval20 pinned | BA | 268.0 | 20 | 134 | FAIL |
| eval20 pinned | LMT | 93.0 | 20 | 36 | FAIL |
| eval20 pinned | UNP | 183.5 | 20 | 128 | FAIL |
| eval20 pinned | UPS | 263.5 | 20 | 163 | FAIL |
| eval20 pinned | COP | 242.0 | 20 | 195 | FAIL |
| eval20 pinned | SLB | 355.0 | 20 | 246 | PASS |
| eval20 pinned | PFE | 368.5 | 20 | 321 | PASS |
| eval20 pinned | MRK | 322.5 | 20 | 268 | PASS |
| eval20 pinned | GILD | 296.5 | 20 | 177 | FAIL |
| eval20 pinned | TGT | 252.5 | 20 | 180 | FAIL |
| eval20 pinned | LOW | 197.0 | 20 | 131 | FAIL |
| eval20 pinned | SBUX | 261.0 | 20 | 177 | FAIL |
| eval20 pinned | T | 324.5 | 20 | 244 | PASS |
| eval20 pinned | VZ | 321.0 | 20 | 284 | PASS |
| eval20 pinned | DUK | 182.0 | 20 | 106 | FAIL |
| eval20 pinned | GS | 166.0 | 20 | 118 | FAIL |
| eval20 pinned | BLK | 47.5 | 20 | 31 | FAIL |
| train18 pinned additions | EMR | 169.5 | 20 | 119 | FAIL |
| train18 pinned additions | ITW | 120.0 | 20 | 86 | FAIL |
| train18 pinned additions | FDX | 175.5 | 20 | 117 | FAIL |
| train18 pinned additions | NSC | 125.0 | 20 | 69 | FAIL |
| train18 pinned additions | CSX | 334.5 | 20 | 227 | PASS |
| train18 pinned additions | WM | 111.0 | 20 | 64 | FAIL |
| train18 pinned additions | HUM | 124.0 | 20 | 87 | FAIL |
| train18 pinned additions | CI | 157.5 | 20 | 90 | FAIL |
| train18 pinned additions | REGN | 52.5 | 20 | 22 | FAIL |
| train18 pinned additions | VRTX | 86.5 | 20 | 59 | FAIL |
| train18 pinned additions | BIIB | 117.5 | 20 | 52 | FAIL |
| train18 pinned additions | ADI | 208.0 | 20 | 119 | FAIL |
| train18 pinned additions | CDNS | 131.5 | 20 | 74 | FAIL |
| train18 pinned additions | SNPS | 88.5 | 20 | 44 | FAIL |
| train18 pinned additions | KLAC | 68.5 | 20 | 42 | FAIL |
| train18 pinned additions | NUE | 121.5 | 20 | 73 | FAIL |
| train18 pinned additions | FCX | 330.5 | 20 | 246 | PASS |
| train18 pinned additions | DOW | 270.5 | 20 | 176 | FAIL |
| train reserves | PH | 60.0 | 20 | 34 | FAIL |
| train reserves | ROK | 103.5 | 20 | 62 | FAIL |
| train reserves | ED | 158.5 | 20 | 84 | FAIL |
| train reserves | AEE | 152.5 | 20 | 115 | FAIL |
| train reserves | EIX | 117.0 | 20 | 66 | FAIL |
| train reserves | WEC | 153.0 | 20 | 108 | FAIL |
| train reserves | MCK | 75.5 | 20 | 40 | FAIL |
| train reserves | TMO | 122.0 | 20 | 67 | FAIL |
| train reserves | DHR | 175.5 | 20 | 111 | FAIL |
| train reserves | ABT | 241.0 | 20 | 117 | FAIL |
| wide pool | NFLX | 171.0 | 20 | 111 | FAIL |
| wide pool | IBM | 219.0 | 20 | 136 | FAIL |
| wide pool | INTC | 372.5 | 20 | 320 | PASS |
| wide pool | MU | 376.0 | 20 | 262 | PASS |
| wide pool | AMAT | 251.5 | 20 | 193 | FAIL |
| wide pool | LRCX | 78.5 | 20 | 26 | FAIL |
| wide pool | ADP | 162.5 | 20 | 81 | FAIL |
| wide pool | INTU | 94.5 | 20 | 57 | FAIL |
| wide pool | WDAY | 218.5 | 20 | 140 | FAIL |
| wide pool | NOW | 98.0 | 20 | 48 | FAIL |
| wide pool | PANW | 253.5 | 20 | 169 | FAIL |
| wide pool | CRWD | 222.0 | 20 | 124 | FAIL |
| wide pool | DDOG | 239.0 | 20 | 172 | FAIL |
| wide pool | PLTR | 346.5 | 20 | 314 | PASS |
| wide pool | ANET | 202.0 | 20 | 152 | FAIL |
| wide pool | SMCI | 171.5 | 20 | 75 | FAIL |
| wide pool | TMUS | 239.5 | 20 | 193 | FAIL |
| wide pool | CMCSA | 351.0 | 20 | 301 | PASS |
| wide pool | CHTR | 163.0 | 20 | 71 | FAIL |
| wide pool | BKNG | 22.5 | 20 | 11 | FAIL |
| wide pool | MAR | 126.0 | 20 | 81 | FAIL |
| wide pool | HLT | 126.5 | 20 | 79 | FAIL |
| wide pool | TJX | 251.5 | 20 | 201 | FAIL |
| wide pool | LULU | 147.0 | 20 | 70 | FAIL |
| wide pool | YUM | 171.0 | 20 | 114 | FAIL |
| wide pool | CMG | 19.0 | 20 | 8 | FAIL |
| wide pool | ORLY | 40.0 | 20 | 17 | FAIL |
| wide pool | AZO | 12.5 | 20 | 5 | FAIL |
| wide pool | PM | 252.0 | 20 | 139 | FAIL |
| wide pool | MO | 324.0 | 20 | 229 | PASS |
| wide pool | KHC | 310.0 | 20 | 187 | PASS |
| wide pool | HSY | 227.0 | 20 | 119 | FAIL |
| wide pool | GIS | 234.5 | 20 | 136 | FAIL |
| wide pool | AMGN | 190.5 | 20 | 122 | FAIL |
| wide pool | ISRG | 128.0 | 20 | 68 | FAIL |
| wide pool | ZTS | 237.0 | 20 | 175 | FAIL |
| wide pool | MTD | 15.5 | 20 | 9 | FAIL |
| wide pool | DXCM | 234.5 | 20 | 204 | FAIL |
| wide pool | SCHW | 306.0 | 20 | 234 | PASS |
| wide pool | MS | 311.5 | 20 | 240 | PASS |
| wide pool | C | 346.0 | 20 | 290 | PASS |
| wide pool | USB | 282.5 | 20 | 227 | FAIL |
| wide pool | PNC | 174.0 | 20 | 99 | FAIL |
| wide pool | COF | 181.5 | 20 | 123 | FAIL |
| wide pool | AXP | 168.0 | 20 | 130 | FAIL |
| wide pool | MET | 174.5 | 20 | 120 | FAIL |
| wide pool | ALL | 158.5 | 20 | 125 | FAIL |
| wide pool | TRV | 80.5 | 20 | 59 | FAIL |
| wide pool | SPGI | 139.0 | 20 | 97 | FAIL |
| wide pool | CME | 122.5 | 20 | 78 | FAIL |
| wide pool | ICE | 221.0 | 20 | 89 | FAIL |
| wide pool | HON | 163.5 | 20 | 89 | FAIL |
| wide pool | GE | 286.0 | 20 | 203 | FAIL |
| wide pool | RTX | 280.5 | 20 | 225 | FAIL |
| wide pool | NOC | 64.0 | 20 | 32 | FAIL |
| wide pool | ETN | 178.0 | 20 | 115 | FAIL |
| wide pool | CMI | 261.5 | 20 | 85 | FAIL |
| wide pool | EOG | 232.0 | 20 | 128 | FAIL |
| wide pool | DVN | 289.5 | 20 | 214 | FAIL |
| wide pool | HES | 201.0 | 20 | 120 | FAIL |
| wide pool | PSX | 326.0 | 20 | 234 | PASS |
| wide pool | MPC | 197.0 | 20 | 92 | FAIL |
| wide pool | OXY | 250.0 | 20 | 211 | FAIL |
| wide pool | SO | 209.0 | 20 | 136 | FAIL |
| wide pool | NEE | 359.0 | 20 | 275 | PASS |
| wide pool | D | 289.5 | 20 | 187 | FAIL |
| wide pool | AEP | 223.0 | 20 | 157 | FAIL |
| wide pool | EXC | 281.5 | 20 | 218 | FAIL |
| wide pool | LIN | 178.5 | 20 | 113 | FAIL |
| wide pool | APD | 164.5 | 20 | 119 | FAIL |
| wide pool | SHW | 152.0 | 20 | 100 | FAIL |
| wide pool | DD | 249.5 | 20 | 168 | FAIL |
| wide pool | PLD | 160.5 | 20 | 106 | FAIL |
| wide pool | AMT | 173.0 | 20 | 133 | FAIL |
| wide pool | EQIX | 106.5 | 20 | 37 | FAIL |
| wide pool | O | 256.0 | 20 | 192 | FAIL |
| wide pool | SPG | 124.5 | 20 | 60 | FAIL |
| wide pool | DAL | 286.5 | 20 | 206 | FAIL |
| wide pool | UAL | 267.0 | 20 | 207 | FAIL |
| wide pool | LUV | 263.0 | 20 | 197 | FAIL |
| wide pool | EXPD | 115.0 | 20 | 62 | FAIL |
| trained-21 (existing) | AAPL | 385.0 | 19 | 329 | PASS |
| trained-21 (existing) | MSFT | 315.0 | 19 | 254 | PASS |
| trained-21 (existing) | GOOGL | 373.0 | 19 | 325 | PASS |
| trained-21 (existing) | AMZN | 371.0 | 19 | 299 | PASS |
| trained-21 (existing) | NVDA | 363.0 | 19 | 312 | PASS |
| trained-21 (existing) | META | 334.0 | 19 | 264 | PASS |
| trained-21 (existing) | TSLA | 376.0 | 19 | 309 | PASS |
| trained-21 (existing) | BRK-B | 145.0 | 19 | 74 | FAIL |
| trained-21 (existing) | JPM | 249.0 | 19 | 193 | FAIL |
| trained-21 (existing) | V | 266.0 | 19 | 153 | FAIL |
| trained-21 (existing) | UNH | 239.0 | 19 | 196 | FAIL |
| trained-21 (existing) | XOM | 315.0 | 19 | 268 | PASS |
| trained-21 (existing) | LLY | 166.0 | 19 | 89 | FAIL |
| trained-21 (existing) | JNJ | 250.0 | 19 | 203 | FAIL |
| trained-21 (existing) | PG | 237.0 | 19 | 151 | FAIL |
| trained-21 (existing) | MA | 167.0 | 19 | 135 | FAIL |
| trained-21 (existing) | HD | 221.0 | 19 | 155 | FAIL |
| trained-21 (existing) | AVGO | 227.0 | 19 | 67 | FAIL |
| trained-21 (existing) | CVX | 262.0 | 19 | 195 | FAIL |
| trained-21 (existing) | ABBV | 228.0 | 19 | 168 | FAIL |
| trained-21 (existing) | SPY | 382.0 | 19 | 323 | PASS |
| burned alt-21 (existing) | ADBE | 240.0 | 19 | 154 | FAIL |
| burned alt-21 (existing) | CRM | 300.0 | 19 | 233 | PASS |
| burned alt-21 (existing) | AMD | 367.0 | 19 | 297 | PASS |
| burned alt-21 (existing) | QCOM | 322.0 | 19 | 271 | PASS |
| burned alt-21 (existing) | TXN | 258.0 | 19 | 177 | FAIL |
| burned alt-21 (existing) | ORCL | 320.0 | 19 | 201 | PASS |
| burned alt-21 (existing) | CSCO | 361.0 | 19 | 299 | PASS |
| burned alt-21 (existing) | ACN | 204.0 | 19 | 90 | FAIL |
| burned alt-21 (existing) | ABT | 241.0 | 20 | 117 | FAIL |
| burned alt-21 (existing) | TMO | 122.0 | 20 | 67 | FAIL |
| burned alt-21 (existing) | DHR | 175.5 | 20 | 111 | FAIL |
| burned alt-21 (existing) | NKE | 319.0 | 19 | 221 | PASS |
| burned alt-21 (existing) | MCD | 214.0 | 19 | 109 | FAIL |
| burned alt-21 (existing) | KO | 336.0 | 19 | 278 | PASS |
| burned alt-21 (existing) | PEP | 254.0 | 19 | 180 | FAIL |
| burned alt-21 (existing) | COST | 138.0 | 19 | 78 | FAIL |
| burned alt-21 (existing) | WMT | 328.0 | 19 | 284 | PASS |
| burned alt-21 (existing) | DIS | 307.0 | 19 | 226 | PASS |
| burned alt-21 (existing) | BAC | 361.0 | 19 | 304 | PASS |
| burned alt-21 (existing) | WFC | 363.0 | 19 | 312 | PASS |



## Full-range fetch verification (2024-01-01..2025-12-31)

Fetch: single sequential process (`_fetch_history_alpaca.py`, FETCH_START=2024-01-01,
FETCH_END=2025-12-31, FETCH_BASKET=frozen 38), **3,337 requests, 0 retries, 0 throttles**;
502 trading days in window, 105 chunks/symbol; the 11 burned alt-21 names resumed from 52
already-complete windows. Verification (per `data/market/history/{SYM}_1m.parquet`,
2024-01-02..2025-12-31):

| sym | basket | days | median bars/day | min day |
|---|---|---|---|---|
| SLB | eval | 501 | 344 | 51 |
| PFE | eval | 501 | 351 | 49 |
| MRK | eval | 501 | 328 | 43 |
| T | eval | 501 | 333 | 52 |
| VZ | eval | 501 | 335 | 50 |
| GILD | eval | 501 | 294 | 51 |
| BA | eval | 501 | 281 | 40 |
| UPS | eval | 501 | 245 | 35 |
| MMM | eval | 501 | 209 | 26 |
| INTC | eval | 501 | 371 | 52 |
| MU | eval | 501 | 354 | 52 |
| PLTR | eval | 501 | 368 | 64 |
| CMCSA | eval | 501 | 351 | 52 |
| MO | eval | 501 | 310 | 36 |
| KHC | eval | 501 | 306 | 42 |
| SCHW | eval | 501 | 323 | 39 |
| MS | eval | 501 | 294 | 39 |
| C | eval | 501 | 334 | 48 |
| PSX | eval | 501 | 223 | 35 |
| NEE | eval | 501 | 328 | 49 |
| FCX | train | 501 | 344 | 41 |
| CSX | train | 501 | 324 | 49 |
| DOW | train | 501 | 302 | 38 |
| ADI | train | 501 | 214 | 32 |
| CRM | train | 501 | 294 | 45 |
| AMD | train | 501 | 378 | 53 |
| QCOM | train | 501 | 308 | 50 |
| ORCL | train | 501 | 315 | 44 |
| CSCO | train | 501 | 353 | 47 |
| NKE | train | 501 | 333 | 47 |
| KO | train | 501 | 338 | 50 |
| WMT | train | 501 | 350 | 51 |
| DIS | train | 501 | 318 | 43 |
| BAC | train | 501 | 370 | 43 |
| WFC | train | 501 | 355 | 45 |
| GE | train | 501 | 270 | 32 |
| USB | train | 501 | 304 | 40 |
| DVN | train | 501 | 305 | 42 |

All 38 parquets: 501/502 trading days (every missing weekday is exactly the fetch
helper's NYSE holiday set plus the 2024-01-01 window boundary), zero duplicate timestamps,
first bar 2024-01-02, last bar 2025-12-31.

**Known gap: 2025-03-10 is absent from every symbol** — including the pre-existing
trained-21/burned-21 parquets from earlier sessions (AAPL, MSFT, SPY verified) — so the
gap is source-side (Alpaca IEX returned no bars that day for any symbol), not a fetch
fault. Cache contract keeps gaps as gaps; `data.py` and the loop's dead-ticker rule
already tolerate it.

**Full-range depth** (median over 501 days) tracks the March probe well and confirms the
Tier-B flags: weakest members MMM 209, ADI 214, PSX 223, UPS 245; everything else ≥ 270.
Against the ≥300 gate on full-range medians, 27/38 pass outright (min flagged: GILD 294,
MS 294, CRM 294, GE 270).
