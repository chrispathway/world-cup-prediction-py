#!/usr/bin/env python3
"""Backtest the Dixon-Coles model on the 2026 World Cup and predict the R16.

Two jobs, both using a model that has never seen a single 2026 World Cup game:

  1. Predict every 2026 finals match already played and score those predictions
     against the real results (accuracy, log-loss, Brier, calibration).
  2. Predict three upcoming, neutral-venue Round-of-16 fixtures.

For every fixture we print expected goals for both sides, the win/draw/loss
probabilities and the single most likely scoreline.

Usage:
    python evaluate_2026.py
"""

from __future__ import annotations

import math

from dataset import load_training_matches, load_world_cup_2026_played
from dixon_coles import DixonColesModel, fit_dixon_coles
from worldcup2026 import WC2026_TEAMS

# Display-name lookup so output reads nicely (the model is keyed on CSV names).
_DISPLAY = {t.csv_name: t.name for t in WC2026_TEAMS}


def display(name: str) -> str:
    return _DISPLAY.get(name, name)


def outcome(home_goals: int, away_goals: int) -> str:
    """'H', 'D' or 'A' — the result from the home side's perspective."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


# --------------------------------------------------------------------------
# Part 1 — backtest against the played 2026 matches
# --------------------------------------------------------------------------

def backtest(model: DixonColesModel) -> None:
    played = load_world_cup_2026_played()

    hits = 0            # times the model's most-probable outcome was correct
    log_loss_sum = 0.0  # -log(prob assigned to the actual outcome)
    brier_sum = 0.0     # squared error over the 3-way probability vector
    goal_abs_err = 0.0  # |predicted total goals - actual total goals|

    print("=" * 78)
    print("  BACKTEST — 2026 WORLD CUP MATCHES ALREADY PLAYED")
    print("  (predicted by a Dixon-Coles model trained only on pre-tournament data)")
    print("=" * 78)
    print(f"  {'Fixture':<34}{'Pred W/D/L %':>16}{'xG':>9}  {'Actual':>8}  {'✓':>1}")
    print("  " + "-" * 74)

    for m in played:
        r = model.match_report(m.home, m.away, neutral=m.neutral)
        probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        actual = outcome(m.home_goals, m.away_goals)

        pred = max(probs, key=probs.get)
        correct = pred == actual
        hits += correct

        # Probabilistic scores. Clamp before the log so a 0% never blows up.
        p_actual = max(probs[actual], 1e-12)
        log_loss_sum += -math.log(p_actual)
        brier_sum += sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2 for k in probs)
        goal_abs_err += abs((r["xg_home"] + r["xg_away"]) - (m.home_goals + m.away_goals))

        fixture = f"{display(m.home)} v {display(m.away)}"
        wdl = f"{r['p_home']*100:4.0f}/{r['p_draw']*100:2.0f}/{r['p_away']*100:2.0f}"
        xg = f"{r['xg_home']:.1f}-{r['xg_away']:.1f}"
        res = f"{m.home_goals}-{m.away_goals} {actual}"
        print(f"  {fixture:<34}{wdl:>16}{xg:>9}  {res:>8}  {'✓' if correct else '·':>1}")

    n = len(played)
    print("  " + "-" * 74)
    print(f"  Matches scored              : {n}")
    print(f"  Outcome accuracy (top pick) : {hits}/{n} = {hits/n*100:.1f}%")
    print(f"  Mean log-loss (lower better): {log_loss_sum/n:.3f}   "
          f"[coin-flip≈1.099, always-33% baseline]")
    print(f"  Mean Brier   (lower better) : {brier_sum/n:.3f}   [uniform guess≈0.667]")
    print(f"  Mean |total-goals| error    : {goal_abs_err/n:.2f} goals")
    print()


# --------------------------------------------------------------------------
# Part 2 — predict the upcoming Round-of-16 fixtures (all neutral ground)
# --------------------------------------------------------------------------

R16_FIXTURES = [
    ("Portugal", "Spain"),
    ("United States", "Belgium"),
    ("Argentina", "Egypt"),
    ("Switzerland", "Colombia"),
]


def predict_fixture(model: DixonColesModel, home: str, away: str) -> None:
    r = model.match_report(home, away, neutral=True)
    a, b = display(home), display(away)
    sx, sy = r["score"]

    print(f"  {a}  vs  {b}   (neutral ground)")
    print("  " + "-" * 50)
    print(f"    Expected goals : {a} {r['xg_home']:.2f}  –  {r['xg_away']:.2f} {b}")
    print(f"    {a + ' win':<16}: {r['p_home']*100:5.1f}%")
    print(f"    {'Draw':<16}: {r['p_draw']*100:5.1f}%")
    print(f"    {b + ' win':<16}: {r['p_away']*100:5.1f}%")
    print(f"    Most likely score : {sx}-{sy}  ({a} {sx} – {sy} {b}), "
          f"{r['p_score']*100:.1f}% of the time")
    print()


def predict_round_of_16(model: DixonColesModel) -> None:
    print("=" * 78)
    print("  ROUND OF 16 — UPCOMING FIXTURES")
    print("=" * 78)
    print()
    for home, away in R16_FIXTURES:
        predict_fixture(model, home, away)


def main() -> None:
    print("Fitting Dixon-Coles on the last decade of internationals "
          "(2026 World Cup excluded)...\n")
    model = fit_dixon_coles(load_training_matches())
    print()
    backtest(model)
    predict_round_of_16(model)


if __name__ == "__main__":
    main()
