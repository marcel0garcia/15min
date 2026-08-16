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

1. **`btc15/models/settlement_twap.py`** — TWAP-aware fair value, replacing
   the endpoint formula as the default (`strategy.fair_value_pricing: "twap"`,
   flip back to `"endpoint"` for A/B):
   - τ > 60s: `z = ln(S/K) / (σ√(τ_eff/yr))` with `τ_eff = τ − 40s`
     (variance of the time-average of a Brownian path over the final minute:
     `(τ−W) + W/3`).
   - τ ≤ 60s: conditions on the accrued average `A` of observed in-window
     ticks: settlement = `((W−u)/W)·A + (u/W)·M` with
     `M ~ N(S, S²σ²·u/3)`, so
     `P(YES) = N((S − K_eff)/(S·σ·√(u/3/yr)))`,
     `K_eff = (W·K − (W−u)·A)/u`. As u→0 this becomes the exact
     settlement-lock step function.
   - 9 unit tests (`tests/test_settlement_twap.py`): regime continuity at the
     60s boundary, lock-in monotonicity, ATM = coin flip, degenerate inputs,
     accrued-window bookkeeping.
2. **Engine wiring** — the scan loop computes the accrued in-window average
   from the BRTI tick buffer and feeds it to the pricer; config-gated.
3. **Crash fix** — settlement-lock entry no longer formats `None` under the
   FV brain; it reports whichever probability actually cleared the gate.

Effect on behavior: mid-window ITM probabilities firm up (less fake variance
→ FV stops leaking edge to the market's correct prices), and the final-minute
signal becomes *the correct math* instead of a heuristic that crashed.

---

## 4. Roadmap

Phased, each with a falsifiable exit criterion. Don't skip gates.

### Phase R1 — Clean re-baseline (paper, ~1 week of sessions)
- Run paper sessions on this branch (settlement race fixed + TWAP pricer).
- **Benchmark against the market, not against 0.25.** A Brier of 0.12 means
  nothing if the Kalshi mid scores 0.10. Extend `shadow_analysis.py` to score
  the *market mid* as a third brain. Edge exists only where
  `Brier(FV) < Brier(market)` — measure it overall and sliced by
  (phase × price band).
- Exit criterion: ≥300 settled market-observations with clean accounting.
  If FV never beats the mid in *any* (phase × band) slice, the directional
  book is dead and only §2(c) MM survives.

### Phase R2 — Trade only where the model beats the market
- Expect the win slices to be: final-minute lock-ins, and extreme bands
  (≤15¢ / ≥85¢) in the back half. Restrict entries to the winning slices —
  i.e., re-introduce price-band gates, but this time *derived from measured
  Brier advantage*, not from session anecdotes.
- Kill remaining DIR-era gates outright (suppression tiers, flow gate,
  orderbook-confirm) rather than leaving them toggled off. The policy should
  read: fee-aware edge threshold + Kelly + risk caps + the measured-slice
  gate. Four things.
- **Make the edge threshold fee-aware**: require
  `edge ≥ fee(price)/100 + spread_half + margin` instead of a flat 5%. At
  95¢ that's ~1¢; at 50¢ it's ~3¢+. This single change encodes "don't trade
  the middle."
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

1. Merge this branch; run `python3 tests/test_settlement_twap.py`.
2. Start nightly paper sessions with recording on (R1).
3. Extend `shadow_analysis.py` with the market-mid brain (small change,
   biggest informational payoff in the project).
4. After ~5 sessions: run the slice analysis and let the data pick the
   tradeable slices for R2.
