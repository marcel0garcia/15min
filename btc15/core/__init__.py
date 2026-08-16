"""The rebuilt decision core (v3).

A clean vertical replacing the legacy ensemble/personas stack:

  fees.py    Kalshi fee arithmetic (the house edge we must clear)
  sigma.py   blended short-horizon realized-vol nowcast
  pricer.py  TWAP-settlement fair value -> MarketQuote per market per scan
  policy.py  the entire decision policy: EV-after-fees gate, slice gate,
             Kelly sizing, one principled exit rule
  paper.py   honest paper broker: depth-walking IOC fills, real fee math,
             authoritative settlement, append-only ledger
  engine.py  async loop wiring feeds + Kalshi market data -> pricer ->
             policy -> broker -> CLI state dict + decision JSONL
  score.py   offline Brier scoring of the model AGAINST THE MARKET MID,
             sliced by phase x price band — the R1 measurement

Design rules (why this exists — see docs/REVIVAL_PLAN.md):
  - Every gate is a measurable hypothesis, not an anecdote. The policy has
    exactly four ideas: window, slice, EV-after-fees, Kelly+risk caps.
  - Paper fills must be pessimistic-realistic: walk displayed depth, pay
    taker fees, optional adverse tick. No "assume filled at ask."
  - Settlement is authoritative: Kalshi's official result first, our own
    final-minute TWAP as the paper fallback. Never mark a held position
    off a drained book.
"""
from btc15.core.engine import CoreEngine

__all__ = ["CoreEngine"]
