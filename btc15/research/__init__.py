"""Offline research: re-run the real decision core over recorded sessions.

KXBTC15M lists exactly ONE market at a time — 96 per day, each a single
near-coin-flip settled 15 minutes later. You cannot tell a 3% edge from
zero by watching a few live sessions; separating them takes hundreds of
settled markets *per configuration*. Tuning knobs against wall-clock is
therefore hopeless: one config test would cost days.

This package replays recorded sessions through the SAME pricer, policy,
and paper broker the live engine runs, under any configuration, so a
config test costs seconds. Live sessions become what they are actually
good for — manufacturing honest data.
"""
