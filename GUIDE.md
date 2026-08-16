# 15-MIN BTC BOT — Operator Guide (v3 core)

Trades Kalshi's `KXBTC15M` binary markets: *will BTC settle above this
strike at the end of the 15-minute window?* YES pays $1 if it does, NO
pays $1 if it doesn't.

This is the **v3 rebuild**. The previous brain (a 5-model directional
ensemble plus a time-phase persona trader with ~15 interacting entry
gates) was deleted, not refactored — see "What changed and why" below.

---

## The one-paragraph version

Every second, for each open market, the bot prices the contract with a
closed-form model of the **actual settlement instrument** — the mean of
the final 60 seconds of BRTI — and compares that probability to what
Kalshi is charging. It enters only when the expected value per contract,
*after the exact taker fee*, clears a margin. It sizes with quarter-Kelly
on the fee-adjusted payoff, holds to settlement unless the model
genuinely flips, and writes every observation to disk so you can later
ask the only question that matters: **did we beat the market's own
price?**

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Watch it think. No orders, no risk. This is where you start.
./run.sh

# 2. Simulated trading with honest fills, fees, and settlement.
./run.sh run --trade

# 3. After a few sessions — the R1 measurement that decides everything.
./run.sh replay enrich <session_id>    # fetch settled outcomes
./run.sh score --suggest               # did we beat the market?
```

Real money requires `--live` **and** a typed confirmation. Don't go there
until `score` says a slice beats the market and you've verified the fee
rate (see the warning below).

---

## How a decision gets made

**1. Price the real instrument.** `core/pricer.py` → `models/settlement_twap.py`

KXBTC15M does *not* settle on the closing print. It settles on the
average of 60 one-second BRTI readings in the final minute. That changes
the math in two places:

- **Before the final minute**, the averaging clips variance. The correct
  horizon is `τ − 40s`, not `τ`. Pricing off `τ` overstates uncertainty
  by up to 3× near the close and pulls probabilities toward 0.50 exactly
  where the market is most active.
- **Inside the final minute**, part of the settlement value is *already
  known* — it's printed BRTI history. The pricer conditions on the
  accrued average `A`:

  ```
  settle = ((60−u)/60)·A + (u/60)·M      u = seconds left
  M ~ Normal(S, S²σ²·u/3)
  P(YES) = N( (S − K_eff) / (S·σ·√(u/3)) ),  K_eff = (60K − (60−u)A) / u
  ```

  As `u → 0` this becomes an exact step function of the locked-in
  average. The old "settlement lock" heuristic was groping at this; now
  it's just the formula.

σ comes from `core/sigma.py`: a variance-space blend of a 60s and a 300s
realized-vol window, so one quiet minute can't halve the estimate.

**2. Decide.** `core/policy.py` — four ideas, and only four:

| # | Gate | What it does |
|---|------|--------------|
| 1 | **Window** | Only trade inside `[min_seconds, max_seconds]`. |
| 2 | **Slice** | Only trade (phase × price-band) buckets you've enabled. Empty = all, which is correct until `score` tells you otherwise. |
| 3 | **EV** | `p·(100−price) − (1−p)·price − fee(price) ≥ margin`. |
| 4 | **Size** | Quarter-Kelly on the fee-adjusted payoff, then hard caps. |

The EV gate is the important one. It replaces the old flat "edge ≥ 5%"
and it is *self-adjusting*: the Kalshi fee peaks at 1.75¢ per contract at
50¢ and falls to ~0.33¢ at 95¢, so the gate is automatically strict in
the middle of the price range and permissive at the extremes. That single
change encodes "don't trade the coin-flip zone" without a special rule.

**3. Exit.** One rule: leave only if the model now wants the *other* side
by `exit_flip_min_ev_cents` **and** at least `exit_min_seconds` remain.

There are deliberately **no loss cuts, no profit takes, no trailing
stops**. A binary settles at 0 or 100; selling into a thin late book at
3¢ is dominated by holding whenever your real P(win) exceeds 3%. The
June 6 post-mortem found the old loss-cut tiers were realizing losses on
positions that went on to settle as *wins* — 8 of the last 14 "emergency
stops" were that bug.

**4. Fill.** `core/paper.py` in paper mode:

- **Depth** — an IOC buy walks the displayed ladder level by level. Want
  12 contracts with 3 showing at the top? You get 3 there and pay up for
  the rest, or fill partially. Empty book = **no fill**, never an
  invented one.
- **Fees** — every fill pays the real curve, from the same function the
  EV gate used to authorize the trade.
- **Cash** — you cannot spend money you don't have.

The invariant, asserted over a full synthetic session in
`tests/test_paper_session.py`:

```
cash == starting_cash + realized_pnl        (realized_pnl is net of every fee)
```

**5. Settle.** Kalshi's official result is authoritative. In paper, the
fallback is our own final-minute BRTI TWAP — the real instrument. The
engine **never** marks a held position off a drained orderbook, which is
what produced the phantom −100% losses in the old logs.

---

## Commands

| Command | What it does |
|---|---|
| `./run.sh` | Dashboard, no trading |
| `./run.sh run --trade` | Paper trading |
| `./run.sh run --trade --live` | **Real money** (confirmation required) |
| `./run.sh run --web` | Also serve the browser dashboard |
| `./run.sh score` | **Model vs. market**, sliced. The R1 gate. |
| `./run.sh score --suggest` | Emit the `enabled_slices` the data supports |
| `./run.sh score --calibration` | Predicted vs. realized decile table |
| `./run.sh replay list` / `enrich` / `convert` / `grid` / `analyze` | Recorded-session tooling |
| `./run.sh markets` / `positions` / `balance` / `history` / `report` | Account & activity |
| `./run.sh trade yes TICKER 10` | One manual trade (paper unless `--live`) |
| `./run.sh config` | Print the loaded config |

---

## Reading the dashboard

- **Header** — BRTI price, feed age, PAPER/LIVE, AUTO/SIGNAL, win rate, uptime
- **Signals** — per market: model P(YES), confidence, edge vs. the market
- **Open Markets** — live book: bid, ask, volume, time left
- **Risk / Balance** — cash, session P&L, open positions, fees paid, HALTED
- **Positions** — entry, cost, live mark-to-market
- **Trade Log** — `[P]` paper, ENTER / EXIT / SETTLED with realized P&L
- **BTC / Kalshi tapes** and **BRTI** — venue health and the consolidated mid
- **PnL** — cumulative realized curve

---

## The measurement that decides the project

A Brier score of 0.12 sounds good and means **nothing** on its own. If
the Kalshi mid scores 0.10, we'd be paying fees to underperform a price
we could have simply accepted. So `./run.sh score` scores three things on
every recorded observation:

```
brier_model    (p_model  − outcome)²
brier_market   (p_market − outcome)²
Δ = brier_market − brier_model         ← positive means WE are better
```

...aggregated per `phase:band` slice, with a paired bootstrap CI so a
12-observation slice can't get promoted on noise. A slice is only real
when its CI is **entirely above zero** — that's what `--suggest` emits.

Rows are thinned to one per ticker per 15 seconds: 900 rows from one
market is one independent event, not 900.

**Workflow:** run paper sessions → `replay enrich` → `score --suggest` →
paste the winning slices into `core.enabled_slices` → re-measure.

---

## ⚠ Before you trade real money

**Verify the fee rate.** `core.fee_rate: 0.07` is the documented rate for
standard categories. Reports of the July 2026 fee revision describe
per-category multipliers and suggest **crypto may price above standard**,
and Kalshi's fee docs were unreachable from the environment where this
was built — so the number is *unverified*.

This is not a rounding concern. If the true crypto rate is double, the EV
gate under-charges by up to 1.75¢ per contract mid-range, which turns
marginal winners into certain losers, in our favor, silently.

Two safeguards:

1. Place **one small live trade**, compare the fee Kalshi reports against
   `taker_fee_usd`, and set `core.fee_rate` to the measured value.
2. The engine's `FeeCalibrator` watches every live fill and logs
   `[FEE] MODEL MISMATCH` with the exact value to put in the config.

---

## What changed and why

| Deleted | Why |
|---|---|
| `models/ensemble.py`, `technical.py`, `ml_model.py`, `bootstrap.py` | The 5-model directional brain scored Brier **0.283** in its own audit — *worse than always saying 50%*. 15-minute BTC direction is very close to a martingale; RSI/MACD don't predict it. |
| `strategy/personas.py`, `strategy/engine.py` | ~15 interacting entry gates, most calibrated to the ensemble's confidence semantics and progressively neutered when they silently blocked the fair-value brain. |
| `strategy/sizer.py`, `risk/manager.py` | Folded into `core/policy.py` so sizing, risk caps, and the EV gate share one fee model instead of three. |
| `recording/shadow_*.py`, `gate_trace.py`, `tools/friday_*.py` | DIR-vs-FV era analysis, superseded by `core/score.py` scoring against the market. |
| `models/fair_value.py` | Priced the closing print — the wrong instrument. Replaced by `settlement_twap.py`. |

| Kept | Why |
|---|---|
| `cli/terminal.py` | The dashboard is the product. Untouched. |
| `kalshi/` | Battle-tested V2 client: RSA auth, WS book maintenance, fill dedup. |
| `recording/` | The data layer. Alpha research is impossible without it. |
| `feeds/brti_feed.py` | The settlement instrument's price source. |

**API currency** (validated against the public changelog, August 2026):
the March 12 2026 removal of legacy integer `count`/cents fields is
handled — the client prefers `_fp`/`_dollars` throughout; `tick_size`
(removed May 7 2026) was never used; and order placement already targets
`/portfolio/events/orders`, the endpoint that supersedes the deprecated
`/portfolio/orders*`.

---

## Extending it

The rule: **a gate earns its place only by measurably beating the market
on recorded data.** Not by sounding sensible, and not because of one bad
session. Add it in shadow, measure with `score`, promote if the CI clears
zero, and delete it if it doesn't.

Deliberately absent, each a separate future experiment with its own
measurement: market making, arbitrage, pyramiding, GTC laddering,
IOC escalation, cool-offs.

---

## Files

```
main.py / run.sh          entry points
config.yaml               every live knob (nothing dead)
btc15/
  core/                   ← the v3 brain
    fees.py               Kalshi fee math + runtime rate calibrator
    sigma.py              blended realized-vol nowcast
    pricer.py             MarketQuote: model prob beside market prob
    policy.py             the whole decision policy (4 gates + 1 exit)
    paper.py              honest paper broker (depth, fees, cash)
    engine.py             async loop wiring it together
    score.py              model vs. market, sliced, bootstrapped
  models/settlement_twap.py   the TWAP-settlement pricer
  cli/                    dashboard + commands (preserved)
  kalshi/                 REST + WS client (preserved)
  feeds/                  BRTI price feed (preserved)
  recording/              telemetry capture + replay (preserved)
tests/                    63 tests — run them before you trade
docs/REVIVAL_PLAN.md      strategy assessment and roadmap
```

---

## Troubleshooting

**"No trades ever fire."** Usually correct behavior — the EV gate rejects
efficiently-priced markets, which is most of them. Confirm with
`grep REJECT logs/bot.log` to see which gate. To loosen, lower
`core.ev_margin_cents` (but expect to pay for it in fees).

**"Feed stale / BRTI unhealthy."** Needs ≥2 healthy venues. Check the
BRTI panel; venue WS reconnects on its own.

**"Awaiting Kalshi result."** Normal for 5–30s after close. Live mode
waits for the official result rather than guessing.

**`INCORRECT_API_KEY_SIGNATURE`** — wrong `KALSHI_RSA_KEY_PATH`, or the
key isn't PEM (PKCS#8/PKCS#1).
