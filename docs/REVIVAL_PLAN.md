# 15min Revival Plan — Quant Assessment & Roadmap

*Written 2026-08-16, after a full read of the codebase and its 50-commit history
(2026-05-18 → 2026-06-06). This is the reference document for where the project
stands, where edge realistically lives in this market, and what to do next.*

---

## 1. Where the project actually is

You remember it as spaghetti. It mostly isn't. The commit history shows a
disciplined arc that many professional desks would recognize:

1. **v1 (DIR era)** — a 5-model directional ensemble (orderbook imbalance,
   technical momentum, trend regression, Black-Scholes digital, LightGBM)
   voting on P(YES), with an AutoTrader running time-phase logic
   (GTC maker early → IOC prime → no entries late), Kelly sizing, and
   phase-tiered exits.
2. **Instrumentation era** — when DIR underperformed, you didn't just tweak
   knobs; you built telemetry: raw Kalshi WS frame capture, per-venue
   top-of-book recording, decision logs, `replay diagnose` (which gate killed
   which entry), `REJECT[gate]` logging, Friday audit tools.
3. **v2 (FV era)** — BRTI reconstruction from Coinbase/Kraken/Bitstamp
   (median + MAD outlier rejection), a fair-value brain
   `P(YES) = N(ln(S/K)/(σ√τ))` fed by a 60-second realized-vol nowcast, run in
   **shadow against DIR for 30 hours**, and promoted on evidence:
   FV Brier 0.1204 vs DIR 0.1539; simulated hold-to-settle P&L +$2.75 vs −$5.88.

That's a real research loop: hypothesis → shadow test → measured promotion.
The genuinely good assets worth preserving:

- **The recording/replay stack** (`btc15/recording/`) — this is the most
  valuable thing in the repo. Alpha research is impossible without data;
  you built the data layer.
- **The CLI dashboard** (`btc15/cli/terminal.py`) — you like it; it stays.
  It's cleanly separated (pure `build_*_panel(state)` functions off a state
  dict), so backend surgery doesn't touch it.
- **The Kalshi client** — RSA auth, WS orderbook maintenance, V2 field-name
  fixes, fill dedup, fee capture. Battle-tested plumbing.
- **The shadow-brain pattern** — `production_brain` config flip with the
  loser still logging. Keep this forever; it's how every future strategy
  should be evaluated.

### What's actually wrong

- **Your P&L history is corrupted.** The last two commits (June 6) fixed a
  settlement race where any position held into a drained orderbook was
  recorded as a −100% loss even when it settled as a win (confirmed example:
  TCD91D2D8, a +$1.17 win booked as −$7.83). 8 of the last 14 "emergency
  stops" were this bug. **Every conclusion drawn from session P&L before
  June 6 is suspect — including your own sense that the bot "doesn't work."**
  The project stopped the same day the accounting was fixed. You never saw a
  clean session.
- **Gate archaeology.** `config.yaml` is a fossil record: half the gates were
  calibrated to DIR's confidence semantics, then neutered one by one when they
  silently blocked FV (`entry_suppression_enabled: false`,
  `reversal_require_orderbook_confirm: false`, price bands widened 10-60 →
  3-97, confidence floors 0.55 → 0.05). What remains is ~15 interacting gates
  where maybe 4 are load-bearing. This is the spaghetti you sensed — not the
  module structure, the *decision policy*.
- **One latent crash** (now fixed on this branch): the settlement-lock entry
  formatted `prob_binary_options` (always `None` under the FV brain) with
  `:.0%` — every settlement-lock fire under FV would have thrown `TypeError`.
  Your highest-conviction late-window strategy could literally never fire.
- **No tests.** 14K lines of trading code, zero test files (until this
  branch). The settlement race and the lock crash are exactly the class of
  bug tests catch.

---

## 2. The market, honestly

Facts that govern everything (verified against current public sources):

- **Settlement**: KXBTC15M settles on the **arithmetic average of 60
  one-second CF Benchmarks BRTI readings over the final minute** before
  expiration — not the closing print. Your code's comments had this right.
- **Fees**: taker ≈ `0.07 × P × (1−P)` per contract → **1.75¢ at 50¢**, but
  only ~0.33¢ at 95¢. Maker is 0 on these markets. A round trip near the
  strike costs ~7% of position value; at the extremes it costs almost nothing.
- **Competition**: automated MMs quote these books. You will not beat them on
  latency from a residential connection, and you will not out-predict them
  ATM with RSI/MACD — 15-minute BTC direction is nearly a martingale, and the
  DIR ensemble's audit Brier of **0.283 (worse than always saying 50%)**
  proved it empirically. Burying ATM directional prediction is correct.

### Where edge can realistically live for a small account

The fee curve + settlement mechanics + retail flow create three structural
pockets, in order of confidence:

**(a) The settlement TWAP window.** In the final minute, `(60−u)/60` of the
settlement value is *already known* — it's printed BRTI history. A pricer that
conditions on the accrued average knows the true probability with rapidly
collapsing variance, while part of the market is still pricing "endpoint risk"
that no longer exists. Even before the final minute, the averaging clips
variance: the correct horizon is `τ − 40s`, not `τ`. Your `settlement_lock`
heuristic ("BSM ≥ 88% with <60s left = free money") was groping at exactly
this — but with the wrong formula and (due to the crash) it never fired. This
is the sharpest, most defensible edge candidate in the project, and it's now
implemented correctly (see §3).

**(b) Favorite–longshot bias at extreme prices.** Retail buyers overpay for
lottery tickets (the 3-10¢ side), which means the 90-97¢ favorite is
persistently a touch cheap — and the fee at 95¢ is ~0.33¢, ~5× cheaper than
ATM. Buying well-cleared favorites in the back half of the window is a
positive-expectation grind *if* your σ is honest. Notably, your own config
comments observed that FV's would-fire opportunities "cluster at extreme
strikes" — the model was already pointing here.

**(c) Maker-side spread capture.** 0% maker fee vs 1.75¢ taker is a real
subsidy. Binary inventory self-liquidates every 15 minutes, which caps the
usual MM nightmare (inventory risk). But adverse selection against faster
traders is brutal in the last minutes, so this is a *later* experiment, run
small, early-window only. The `mm_aggressive` scaffolding already exists.

**What is *not* edge**: ATM directional calls from technicals (DIR proved it),
latency racing, and trading the 40-60¢ zone at all — that's where the fee is
maximal and your information advantage is minimal. The cleanest expression:
**stop trading the middle of the price range entirely.**

### Expectation setting

With a $100 bankroll and $10 caps, even a genuine 3% average edge at
realistic fill rates is a few dollars a day. That's the correct way to value
this project: **a live laboratory that teaches you market microstructure and
produces a verified track record** — which is worth far more than the P&L. If
the edge verifies, scaling capital is the easy part.

---

## 3. What's already done on this branch

### 3a. The TWAP-settlement pricer

`btc15/models/settlement_twap.py` replaces the endpoint formula. KXBTC15M
settles on the **mean of the final 60s of BRTI**, not the closing print:

  - tau > 60s: `z = ln(S/K) / (sigma*sqrt(tau_eff/yr))`, `tau_eff = tau - 40s`
    (variance of the time-average of a Brownian path over the final minute).
  - tau <= 60s: conditions on the accrued average `A` of observed in-window
    ticks. `settle = ((W-u)/W)*A + (u/W)*M`, `M ~ N(S, S^2 sigma^2 u/3)`, so
    `P(YES) = N((S - K_eff)/(S*sigma*sqrt(u/3)))`, `K_eff = (W*K - (W-u)*A)/u`.
    As u -> 0 this becomes the exact settlement-lock step function.

### 3b. The brain was rebuilt from scratch, not patched

The whole legacy decision stack was **deleted** — ~6,000 lines: the
5-model ensemble (audit Brier 0.283, worse than always saying 50%), the
personas AutoTrader and its ~15 interacting gates, the DIR-era sizer and
risk manager, the shadow-comparison analyzers, and the Friday audit
tools. `config.yaml` went from 340 lines of archaeology to ~110 lines
where every knob is read by live code.

The replacement is `btc15/core/`, a clean vertical:

| Module | Responsibility |
|---|---|
| `fees.py` | Kalshi fee math + a runtime calibrator that measures the true rate from real fills |
| `sigma.py` | Blended fast/slow realized-vol nowcast (one quiet minute can't halve it) |
| `pricer.py` | `MarketQuote` — model probability *beside* market probability, always |
| `policy.py` | The entire decision policy: window, slice, EV-after-fees, Kelly. One exit rule. |
| `paper.py` | Honest paper broker: depth-walking fills, real fees, cash constraints |
| `engine.py` | Async loop; the CLI dashboard is untouched and reads the same state dict |
| `score.py` | Model vs. **market mid**, sliced, with paired bootstrap CIs |

Three decisions worth calling out:

1. **The EV gate replaces the flat edge threshold.** `p*(100-price) -
   (1-p)*price - fee(price) >= margin`. Because the fee peaks at 1.75c
   mid-range and falls to 0.33c at the extremes, the gate is
   automatically strict in the coin-flip zone and permissive where edge
   structurally lives. No special rule needed.

2. **Exits are one rule: flip only.** No loss cuts, no profit takes. A
   binary settles 0/100; selling into a thin book at 3c is dominated by
   holding whenever real P(win) > 3%. The June 6 post-mortem showed the
   old tiers realized losses on positions that settled as wins.

3. **Paper fidelity is the point.** Fills walk displayed depth, pay the
   real fee curve, and respect cash. The invariant `cash == start +
   realized_pnl` is asserted over a full synthetic session. A test
   confirms that when the market quotes our exact probability, the bot
   fires **zero** trades — the old "fill at ask, no fees" path would have
   shown a fantasy profit there.

### 3c. Tests: 63, from zero

`tests/test_core.py` (41), `test_settlement_twap.py` (9),
`test_paper_session.py` (6 end-to-end synthetic sessions),
`test_engine_wiring.py` (7, drives the real engine against a fake Kalshi).

They already caught three real bugs: a float-rounding error billing an
extra cent on every mid-price fill, a mixed fee convention that broke
cash conservation, and a latent `TypeError` that made the old
settlement-lock entry impossible to fire.

### 3d. Kalshi API currency (validated August 2026)

- Legacy integer `count`/cents fields **removed 2026-03-12** — the client
  already prefers `_fp`/`_dollars` everywhere. No action needed.
- `tick_size` **removed 2026-05-07** — never used.
- `/portfolio/orders*` deprecated (>= 2026-05-21) — we already use
  `/portfolio/events/orders` and the batched DELETE.
- **Open risk — the fee rate.** 0.07 is the documented standard-category
  rate, but reports of the July 2026 revision describe per-category
  multipliers and suggest crypto may price higher. Kalshi's docs and API
  are both blocked by this environment's egress policy, so it is
  unverified. `fee_rate`/`fee_multiplier` are now config knobs, and
  `FeeCalibrator` measures the real rate from live fills and logs
  `[FEE] MODEL MISMATCH` with the value to set. **Verify with one small
  live trade before trusting any EV number.**

---

## 4. Roadmap

Phased, each with a falsifiable exit criterion. Don't skip gates.

### Phase R1 — Measure against the market (paper, ~1 week of sessions)

**The tooling for this phase is built and tested. R1 is now just running it.**

- Run paper sessions: `./run.sh run --trade`. Every market-scan writes a
  decision row carrying both `p_model` and `p_market`.
- After each session: `./run.sh replay enrich <session>` then
  `./run.sh score --suggest`.
- The bar is the market, not 0.25. `score` reports
  `Δ = Brier(market) − Brier(model)` per (phase × price band) slice with a
  paired bootstrap CI; a slice is real only when its CI clears zero.
- Leave `core.enabled_slices: []` for the whole phase — observe everything,
  restrict later. Leave `auto_trade` on so fills and fees are exercised, but
  remember paper P&L is secondary here; the Brier delta is the finding.
- Exit criterion: ≥300 settled observations. If no slice beats the market,
  the directional book is dead and only §2(c) maker capture survives.

### Phase R2 — Trade only where the model beats the market
- Expect the win slices to be: final-minute lock-ins, and extreme bands
  (≤15¢ / ≥85¢) in the back half. Restrict entries to the winning slices —
  i.e., re-introduce price-band gates, but this time *derived from measured
  Brier advantage*, not from session anecdotes.
- *(Done in the rebuild: the DIR-era gates no longer exist, and the policy
  already reads exactly four things — window, slice, fee-aware EV, Kelly.)*
- *(Done: `core.ev_margin_cents` gates on EV after the exact fee curve,
  which is structurally strict mid-range and permissive at the extremes.)*
- σ upgrade (the FV brain's real weak spot): blend the 60s nowcast with a
  5-minute EWMA of squared returns and de-noise the 1s reconstruction
  (microstructure noise biases σ up, which drags every FV prob toward 0.5).
  Validate σ by regressing realized 15-min outcome variance on forecast.

### Phase R3 — Small live deployment
- $100, live, only the verified slices, daily loss limit $10, two weeks.
- Success = positive P&L *and* live Brier consistent with paper. Either
  failing → back to R2 with the recordings.

### Phase R4 — Optional expansions (only after R3 verifies)
- Passive early-window MM experiment (small, per-side caps, cancel at T−2min).
- The `KXBTC` hourly and `KXETH15M` series — same code, more surface area.

### Phase A — Agentic layer (parallel track, start anytime)

The right role for an LLM here is **not** per-tick trade decisions (a 1-second
scan loop needs milliseconds and determinism; a model call is 100–1000× too
slow and non-reproducible). The right roles mirror what *you* were doing
manually every Friday:

1. **Session post-mortem analyst** (highest value, lowest risk). A scheduled
   Claude agent that, after each session: runs `replay diagnose` +
   `shadow_analysis` + the counterfactual P&L sim over the recorded Parquet,
   writes a structured markdown report (what fired, what was rejected and by
   which gate, Brier vs market by slice, anomalies), and files it into
   `docs/sessions/`. Your Friday-audit workflow, automated nightly.
2. **Hypothesis runner.** You (or the analyst agent) phrase a hypothesis
   ("the flow gate is rejecting winners"); an agent writes the backtest
   against recordings, runs it, reports effect size. The
   `tools/friday_*.py` scripts show this pattern works — an agent can
   generate them on demand instead of you hand-writing each one.
3. **Config-change proposer with guardrails.** The analyst agent may *propose*
   a config diff with evidence attached; a human (you) merges it. Never let
   an agent hot-edit live trading parameters.
4. **Regime awareness feed.** A slow loop (minutes, not seconds) that flags
   scheduled macro events (CPI, FOMC) and unusual realized-vol regimes into
   the state dict; the engine reads a simple `regime` flag to widen σ or
   stand down. LLM builds/maintains the calendar; the engine stays
   deterministic.

This division — deterministic engine, agentic research loop around it — is how
real desks use LLMs today, and your recording infrastructure is precisely what
makes it possible.

---

## 5. The pivot question (options / equities)

Recommendation: **don't pivot yet, and don't pivot to listed options first.**

The premise "real edge can be found in the real stock market" is mostly
backwards for a solo retail trader: listed options are quoted by Citadel,
Susquehanna, Jane Street et al. — you'd face *more* institutional competition
than on Kalshi 15-minute crypto, plus brokerage integration, pattern-day-trader
rules, and slower feedback loops (weeks per hypothesis vs 96 markets/day
here). Kalshi's 15-min market is one of the few venues where a meaningful
share of counterparty flow is recreational.

The Kalshi thesis gets a fair, fast trial: R1–R3 is roughly a month of mostly
unattended runtime, and the TWAP/extreme-band edges are testable with the
tooling you already built. If R1 falsifies everything — every slice loses to
the market mid, MM bleeds to adverse selection — *then* pivot, and the assets
carry: the CLI dashboard, the recording/replay pattern, the shadow-brain
evaluation discipline, and Kelly/risk framework are all venue-agnostic. The
natural adjacent step would be other Kalshi series (hourly BTC/ETH, index
ranges) before leaving the platform entirely.

---

## 6. Immediate next steps (this week)

1. Apply this branch and run the suite: `for t in tests/*.py; do python3 $t; done`
   (63 tests; all should pass).
2. `./run.sh` for one window and confirm the dashboard populates — Signals,
   BRTI, sigma, fair_value.
3. Start nightly paper sessions: `./run.sh run --trade`.
4. After ~5 sessions: `./run.sh replay enrich <session>` then
   `./run.sh score --suggest`. That output decides R2.
5. **Before any live trade**: verify the fee rate (§3d). One small order,
   compare Kalshi's reported fee to `taker_fee_usd`, set `core.fee_rate`.
