#!/usr/bin/env python3
"""Generate the README 'Live scorecard' markdown table from the fitted model.

Scores every played 2026 World Cup match (loaded from the cached dataset, none
of which the model was trained on) against the model's most-likely outcome and
emits the markdown rows used in README.md, plus the running accuracy tally.
"""

from __future__ import annotations

from dataset import load_world_cup_2026_played
from dixon_coles import fit_dixon_coles, load_training_matches
from worldcup2026 import WC2026_TEAMS

_DISPLAY = {t.csv_name: t.name for t in WC2026_TEAMS}
_FLAG = {t.csv_name: t.flag for t in WC2026_TEAMS}


def display(name: str) -> str:
    return _DISPLAY.get(name, name)


def flag(name: str) -> str:
    return _FLAG.get(name, "")


def outcome(hg: int, ag: int) -> str:
    return "H" if hg > ag else "A" if hg < ag else "D"


def main() -> None:
    model = fit_dixon_coles(load_training_matches(), verbose=False)
    played = load_world_cup_2026_played()

    hits = 0
    rows = []
    for m in played:
        r = model.match_report(m.home, m.away, neutral=m.neutral)
        probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        pred = max(probs, key=probs.get)
        actual = outcome(m.home_goals, m.away_goals)
        correct = pred == actual
        hits += correct

        home, away = display(m.home), display(m.away)
        if pred == "H":
            label = f"{home} win"
        elif pred == "A":
            label = f"{away} win"
        else:
            label = "Draw"
        wdl = f"{r['p_home']*100:.0f} / {r['p_draw']*100:.0f} / {r['p_away']*100:.0f}"
        mark = "✅" if correct else "❌"
        rows.append(
            f"| {flag(m.home)} {home} v {away} {flag(m.away)} "
            f"| {m.home_goals}–{m.away_goals} | {label} ({wdl}) | {mark} |"
        )

    n = len(played)
    print(f"ACCURACY: {hits} / {n} = {hits/n*100:.1f}%")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
