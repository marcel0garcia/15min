"""Constants shared by the live engine and the offline replay harness.

These live in their own module, with no imports, for one reason: replay
must feed the model exactly what the engine feeds it, and importing
`btc15.core.engine` to find that out would drag in the Kalshi client, the
websockets, and the venue feeds — none of which offline research needs.

If a number here changes, it changes for both, which is the point. A
replay that hands the same estimator a different span of history is
silently measuring a different bot.
"""
from __future__ import annotations

# How much BRTI history the vol nowcast and the accrued-average calculation
# see on every scan.
TICK_HISTORY_SEC = 360.0

# Price levels of each side stamped into every decision row. This is what
# makes recordings self-sufficient for offline fill simulation, and why raw
# Kalshi frame capture can be left off for long unattended runs.
DECISION_BOOK_LEVELS = 5
