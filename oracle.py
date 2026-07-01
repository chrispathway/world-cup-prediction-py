#!/usr/bin/env python3
"""World Cup Oracle — terminal edition.

Fits a Dixon-Coles goal model on the last decade of international results
(recency- and competition-weighted, with the 2026 World Cup held out), runs a
Monte Carlo simulation of the 2026 tournament, prints every team's title odds,
then drops into an interactive prompt where you can type two teams to get the
head-to-head prediction (win/draw/loss, expected goals, likely score).

Usage:
    python oracle.py                 # 10,000 simulations (default)
    python oracle.py --sims 2000     # faster, slightly noisier
"""

import argparse
import sys

from dataset import load_training_matches
from dixon_coles import fit_dixon_coles
from simulation import match_probabilities, run_simulations, set_goal_model
from worldcup2026 import WC2026_TEAMS, find_team


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%"


def rating(team) -> int:
    """A compact display 'strength index' derived from the Dixon-Coles fit."""
    return round(1000 + 150 * MODEL.strength(team.csv_name))


def print_title_odds(sim: dict, num_sims: int) -> None:
    rows = sorted(sim["titles"].items(), key=lambda kv: kv[1], reverse=True)
    team_by_name = {t.name: t for t in WC2026_TEAMS}

    print()
    print("=" * 62)
    print("  TITLE ODDS — 2026 FIFA WORLD CUP")
    print(f"  (Monte Carlo, {num_sims:,} simulated tournaments)")
    print("=" * 62)
    print(f"  {'#':>2}  {'Team':<16} {'Rtg':>5}  {'Champion':>9}  {'Final':>7}  {'Semi':>7}")
    print("  " + "-" * 58)
    for i, (name, titles) in enumerate(rows, 1):
        t = team_by_name[name]
        print(
            f"  {i:>2}  {t.flag} {name:<14} {rating(t):>5}  "
            f"{pct(titles, num_sims):>9}  "
            f"{pct(sim['finals'][name], num_sims):>7}  "
            f"{pct(sim['semi_finals'][name], num_sims):>7}"
        )
    print("=" * 62)


def print_match(a_name: str, b_name: str) -> None:
    a = find_team(a_name)
    b = find_team(b_name)
    if a is None:
        print(f"  ! Unknown team: '{a_name}'. Type 'teams' to list them.")
        return
    if b is None:
        print(f"  ! Unknown team: '{b_name}'. Type 'teams' to list them.")
        return
    if a.name == b.name:
        print("  ! Pick two different teams.")
        return

    r = match_probabilities(a, b)  # neutral ground, Dixon-Coles

    print()
    print(f"  {a.flag} {a.name}  (Rtg {rating(a)})   vs   {b.flag} {b.name}  (Rtg {rating(b)})")
    print("  " + "-" * 50)
    print(f"  {a.name} win : {r['p_win_a'] * 100:5.1f}%")
    print(f"  Draw      : {r['p_draw'] * 100:5.1f}%")
    print(f"  {b.name} win : {r['p_win_b'] * 100:5.1f}%")
    print(f"  Expected goals : {a.name} {r['xg_a']}  –  {r['xg_b']} {b.name}")
    print(f"  Most likely score : {r['most_likely_score']}  ({a.name} – {b.name})")
    print()


def list_teams() -> None:
    print()
    for t in sorted(WC2026_TEAMS, key=lambda x: x.group):
        print(f"  {t.group}  {t.flag} {t.name:<16} ({t.code})")
    print()


def interactive_loop() -> None:
    print()
    print("Head-to-head predictor. Enter two teams, e.g.  Brazil vs France")
    print("Commands:  titles | teams | quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            print("Bye.")
            return
        if low == "titles":
            print_title_odds(SIM, NUM_SIMS)
            continue
        if low == "teams":
            list_teams()
            continue

        # Accept "A vs B", "A v B", "A - B", "A x B", "A,B"
        sep = next((s for s in (" vs ", " v ", " - ", " x ", ",") if s in low), None)
        if sep is None:
            print("  ! Format: <team> vs <team>   (or 'titles', 'teams', 'quit')")
            continue
        a_part, b_part = line.split(sep, 1)
        print_match(a_part.strip(), b_part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="World Cup Oracle (terminal)")
    parser.add_argument("--sims", type=int, default=10_000, help="number of tournament simulations")
    args = parser.parse_args()

    global MODEL, SIM, NUM_SIMS
    NUM_SIMS = args.sims

    # Fit the Dixon-Coles goal model on the recency-weighted last decade (the
    # 2026 World Cup is held out) and use it as the engine for every fixture.
    MODEL = fit_dixon_coles(load_training_matches())
    set_goal_model(MODEL)

    print(f"\nRunning {NUM_SIMS:,} Monte Carlo tournament simulations...")
    SIM = run_simulations(NUM_SIMS)

    print_title_odds(SIM, NUM_SIMS)
    interactive_loop()


if __name__ == "__main__":
    sys.exit(main())
