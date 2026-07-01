"""Match sampling + Monte Carlo tournament simulation, driven by Dixon-Coles.

Every fixture's goals come from the fitted Dixon-Coles model (see
dixon_coles.py): a scoreline is drawn from the joint distribution of the two
teams' goals, which preserves the draw / low-score dependence that plain
independent Poissons miss. World Cup matches are simulated on neutral ground,
so no side gets a home-advantage boost.
"""

from __future__ import annotations

import numpy as np

from worldcup2026 import WC2026_TEAMS, GROUPS, WCTeam

NUM_SIMULATIONS = 10_000

# The fitted Dixon-Coles model, injected at runtime by oracle.py so this module
# has no hard dependency on scipy when imported on its own.
_MODEL = None
_RNG = np.random.default_rng()


def set_goal_model(model) -> None:
    """Install the fitted Dixon-Coles model as the goal engine."""
    global _MODEL
    _MODEL = model


# The model is keyed on CSV team names; map each WC display team to that key.
def _key(team: WCTeam) -> str:
    return team.csv_name


def simulate_match(a: WCTeam, b: WCTeam) -> tuple[int, int]:
    """Draw one neutral-venue scoreline (a_goals, b_goals) for the fixture."""
    return _MODEL.sample_score(_RNG, _key(a), _key(b), neutral=True)


def match_probabilities(a: WCTeam, b: WCTeam) -> dict:
    """Exact win/draw/loss probabilities, expected goals and modal scoreline.

    Computed straight from the Dixon-Coles scoreline grid — no sampling needed.
    """
    r = _MODEL.match_report(_key(a), _key(b), neutral=True)
    sx, sy = r["score"]
    return {
        "p_win_a": r["p_home"],
        "p_draw": r["p_draw"],
        "p_win_b": r["p_away"],
        "xg_a": round(r["xg_home"], 2),
        "xg_b": round(r["xg_away"], 2),
        "most_likely_score": f"{sx}-{sy}",
    }


# ---------- Tournament ----------

# Fixed Round-of-32 bracket template (faithful to the real World Cup format,
# without FIFA's full third-place lookup table). Each entry is one R32 match as
# (slot_a, slot_b); slots are filled in after the group stage. Codes:
#   ("W", group) -> winner of that group
#   ("R", group) -> runner-up of that group
#   ("T", i)     -> the i-th best third-place team (0 = best)
# Matches are listed in bracket order, so consecutive matches feed the same
# Round-of-16 tie, those feed the same quarter-final, and so on.
#
# Properties that mirror the real draw:
#   * No group winner can meet another group winner in the Round of 32.
#   * Each group's winner and runner-up sit in opposite halves of the bracket,
#     so two teams from the same group can only meet again in the final.
R32_BRACKET: list[tuple[tuple, tuple]] = [
    # --- Top half ---
    (("W", "A"), ("T", 0)),
    (("R", "C"), ("R", "D")),
    (("W", "E"), ("T", 1)),
    (("W", "G"), ("R", "H")),
    (("W", "B"), ("T", 2)),
    (("R", "F"), ("R", "L")),
    (("W", "I"), ("T", 3)),
    (("W", "K"), ("R", "J")),
    # --- Bottom half ---
    (("W", "C"), ("T", 4)),
    (("R", "A"), ("R", "B")),
    (("W", "F"), ("T", 5)),
    (("W", "H"), ("R", "G")),
    (("W", "D"), ("T", 6)),
    (("R", "E"), ("R", "I")),
    (("W", "J"), ("T", 7)),
    (("W", "L"), ("R", "K")),
]

# Group whose winner occupies the other side of each ("T", i) slot above, used
# to keep a third-place team from being drawn against its own group's winner.
_TSLOT_WINNER_GROUP = {0: "A", 1: "E", 2: "B", 3: "I", 4: "C", 5: "F", 6: "D", 7: "L"}


def _assign_thirds(thirds: list[WCTeam]) -> list[WCTeam]:
    """Place the 8 best third-place teams into the 8 ("T", i) slots, avoiding
    (where possible) a third-place team facing its own group's winner."""
    remaining = list(range(len(thirds)))
    assigned: list[WCTeam] = []
    for i in range(len(thirds)):
        winner_group = _TSLOT_WINNER_GROUP[i]
        pick = next(
            (ti for ti in remaining if thirds[ti].group != winner_group),
            remaining[0],
        )
        assigned.append(thirds[pick])
        remaining.remove(pick)
    return assigned


class _Standing:
    __slots__ = ("team", "points", "gf", "ga")

    def __init__(self, team: WCTeam):
        self.team = team
        self.points = 0
        self.gf = 0
        self.ga = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _simulate_group(group_teams: list[WCTeam]) -> list[_Standing]:
    standings = [_Standing(t) for t in group_teams]
    n = len(standings)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = standings[i], standings[j]
            ga, gb = simulate_match(a.team, b.team)
            a.gf += ga
            a.ga += gb
            b.gf += gb
            b.ga += ga
            if ga > gb:
                a.points += 3
            elif gb > ga:
                b.points += 3
            else:
                a.points += 1
                b.points += 1
    standings.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
    return standings


def _simulate_knockout(a: WCTeam, b: WCTeam) -> bool:
    """Return True if A advances. Ties go to a strength-weighted shootout."""
    ga, gb = simulate_match(a, b)
    if ga > gb:
        return True
    if gb > ga:
        return False
    # Coin-flip nudged by the two sides' overall strength (attack - defence).
    edge_a = _MODEL.strength(_key(a)) - _MODEL.strength(_key(b))
    return _RNG.random() < min(0.65, max(0.35, 0.5 + edge_a / 4))


def run_simulations(num_simulations: int = NUM_SIMULATIONS) -> dict:
    names = [t.name for t in WC2026_TEAMS]
    result = {
        stage: {n: 0 for n in names}
        for stage in (
            "titles",
            "finals",
            "semi_finals",
            "quarter_finals",
            "round_of_16",
            "group_wins",
            "group_advances",
        )
    }

    groups_index = {g: [t for t in WC2026_TEAMS if t.group == g] for g in GROUPS}

    for _ in range(num_simulations):
        group_results = [_simulate_group(groups_index[g]) for g in GROUPS]

        winners_by_group: dict[str, WCTeam] = {}
        runners_by_group: dict[str, WCTeam] = {}
        third_placers: list[_Standing] = []

        for g, standings in zip(GROUPS, group_results):
            winner, second, third = standings[0], standings[1], standings[2]
            result["group_wins"][winner.team.name] += 1
            result["group_advances"][winner.team.name] += 1
            result["group_advances"][second.team.name] += 1
            winners_by_group[g] = winner.team
            runners_by_group[g] = second.team
            third_placers.append(third)

        # Best 8 third-place teams complete the 32-team knockout bracket.
        third_placers.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
        best_thirds = third_placers[:8]
        for s in best_thirds:
            result["group_advances"][s.team.name] += 1
        thirds = _assign_thirds([s.team for s in best_thirds])

        # Fill the fixed bracket template instead of a random draw, so that
        # finishing position and seeding protection actually shape the path.
        def resolve(slot: tuple) -> WCTeam:
            kind, key = slot
            if kind == "W":
                return winners_by_group[key]
            if kind == "R":
                return runners_by_group[key]
            return thirds[key]  # ("T", i)

        pool: list[WCTeam] = []
        for slot_a, slot_b in R32_BRACKET:
            pool.append(resolve(slot_a))
            pool.append(resolve(slot_b))

        def knockout_round(teams: list[WCTeam]) -> list[WCTeam]:
            winners: list[WCTeam] = []
            for i in range(0, len(teams), 2):
                a, b = teams[i], teams[i + 1]
                winners.append(a if _simulate_knockout(a, b) else b)
            return winners

        r16 = knockout_round(pool)
        for t in r16:
            result["round_of_16"][t.name] += 1

        qf = knockout_round(r16)
        for t in qf:
            result["quarter_finals"][t.name] += 1

        sf = knockout_round(qf)
        for t in sf:
            result["semi_finals"][t.name] += 1

        finalists = knockout_round(sf)
        for t in finalists:
            result["finals"][t.name] += 1

        champion = knockout_round(finalists)[0]
        result["titles"][champion.name] += 1

    return result
