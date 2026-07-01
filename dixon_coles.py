"""Dixon-Coles goal model — the statistical core.

This replaces the old "Elo difference -> two independent Poissons" engine with
the model from Dixon & Coles (1997), the standard bivariate approach in
football forecasting. It keeps the goal-based, Poisson-style foundation but
fixes its best-known flaw: treating the two teams' goals as independent
under-predicts draws and low-scoring games.

The model gives every team an **attack** and a **defence** rating and, for a
fixture between home team *i* and away team *j*, expected goals

    lambda (home) = exp(attack_i - defence_j + home_advantage)
    mu     (away) = exp(attack_j - defence_i)

The joint probability of an exact scoreline (x, y) is

    P(x, y) = tau(x, y) * Poisson(x; lambda) * Poisson(y; mu)

where ``tau`` is the Dixon-Coles low-score correction, governed by a single
dependence parameter ``rho``. tau nudges the probabilities of 0-0, 1-0, 0-1 and
1-1 up or down, coupling the two scores exactly where independence fails.

Every parameter — all the attack/defence ratings, the home advantage and rho —
is fit *together* by maximising the recency- and competition-weighted
log-likelihood of the historical matches (see dataset.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dataset import Match, load_training_matches

# Scorelines above this are astronomically unlikely for international football;
# capping the grid keeps probability sums and sampling fast and exact enough.
MAX_GOALS = 12

# Teams with fewer than this many matches in the window (mostly non-FIFA
# regional or novelty sides) can't support their own reliable rating, so they
# are pooled into one shared "Other" strength. All 2026 qualifiers are far
# above this threshold, so predictions are unaffected.
MIN_APPEARANCES = 8
OTHER = "Other"


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction, vectorised over arrays of scorelines.

    Only the four lowest joint scores are adjusted; everything else is left at
    1.0 (i.e. plain independence). ``rho`` > 0 inflates draws / low scores.
    """
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class DixonColesModel:
    """A fitted Dixon-Coles model: team ratings plus home advantage and rho."""

    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    n_matches: int
    total_weight: float

    def strength(self, team: str) -> float:
        """Overall strength in log-goal units: attack plus defence (a higher
        defence rating means opponents score fewer, so both add to quality).
        A single 'how good is this team' number for ranking and tie-breaks."""
        att, deff = self._ratings(team)
        return att + deff

    def _ratings(self, team: str) -> tuple[float, float]:
        """(attack, defence) for a team, falling back to the pooled "Other"
        rating for anyone not fitted individually."""
        if team in self.attack:
            return self.attack[team], self.defence[team]
        return self.attack.get(OTHER, 0.0), self.defence.get(OTHER, 0.0)

    # --- Expected goals ----------------------------------------------------
    def expected_goals(
        self, home: str, away: str, neutral: bool = True
    ) -> tuple[float, float]:
        """Expected goals (lambda, mu) for the home and away side.

        World Cup knockout ties are played on neutral ground, so ``neutral``
        defaults to True and neither side gets the home-advantage term. Pass
        ``neutral=False`` for a genuine host-nation fixture.
        """
        a_home, d_home = self._ratings(home)
        a_away, d_away = self._ratings(away)
        adv = 0.0 if neutral else self.home_advantage
        lam = np.exp(a_home - d_away + adv)
        mu = np.exp(a_away - d_home)
        return float(lam), float(mu)

    # --- Full scoreline distribution --------------------------------------
    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Return the (MAX_GOALS+1) x (MAX_GOALS+1) joint scoreline probability
        grid, ``P[x, y]`` = probability the home side scores x and away y.
        """
        lam, mu = self.expected_goals(home, away, neutral)
        goals = np.arange(MAX_GOALS + 1)
        # Independent Poisson marginals via the log-pmf (stable, no factorials
        # overflowing): log p(k) = k*log(rate) - rate - log(k!).
        log_fact = np.cumsum(np.log(np.maximum(goals, 1)))  # log(k!) with log(0!)=0
        log_fact[0] = 0.0
        log_home = goals * np.log(lam) - lam - log_fact
        log_away = goals * np.log(mu) - mu - log_fact
        grid = np.exp(log_home[:, None] + log_away[None, :])

        # Apply the Dixon-Coles correction to the four low-score cells.
        xs, ys = np.meshgrid(goals, goals, indexing="ij")
        grid *= _tau(xs, ys, np.full_like(grid, lam), np.full_like(grid, mu), self.rho)

        grid /= grid.sum()  # renormalise (tau makes the raw grid sum ~1, not 1)
        return grid

    def match_report(self, home: str, away: str, neutral: bool = True) -> dict:
        """Win/draw/loss probabilities, expected goals and the modal scoreline."""
        lam, mu = self.expected_goals(home, away, neutral)
        grid = self.score_matrix(home, away, neutral)

        p_home = float(np.tril(grid, -1).sum())  # x > y
        p_away = float(np.triu(grid, 1).sum())   # x < y
        p_draw = float(np.trace(grid))           # x == y

        x, y = np.unravel_index(int(grid.argmax()), grid.shape)
        return {
            "xg_home": lam,
            "xg_away": mu,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "score": (int(x), int(y)),
            "p_score": float(grid[x, y]),
        }

    def sample_score(self, rng: np.random.Generator, home: str, away: str,
                     neutral: bool = True) -> tuple[int, int]:
        """Draw one scoreline from the joint Dixon-Coles distribution.

        Sampling from the joint grid (rather than two independent Poissons)
        preserves the draw/low-score dependence the model exists to capture.
        """
        grid = self.score_matrix(home, away, neutral)
        flat = grid.ravel()
        idx = rng.choice(flat.size, p=flat)
        x, y = np.unravel_index(idx, grid.shape)
        return int(x), int(y)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_dixon_coles(
    matches: list[Match] | None = None, verbose: bool = True
) -> DixonColesModel:
    """Fit attack/defence, home advantage and rho by weighted maximum likelihood.

    All parameters are estimated jointly: we minimise the negative
    recency/competition-weighted log-likelihood of every training scoreline
    with L-BFGS-B.
    """
    if matches is None:
        matches = load_training_matches()

    # Pool rarely-seen teams into a single "Other" rating (see MIN_APPEARANCES).
    appearances: dict[str, int] = {}
    for m in matches:
        appearances[m.home] = appearances.get(m.home, 0) + 1
        appearances[m.away] = appearances.get(m.away, 0) + 1

    def label(team: str) -> str:
        return team if appearances[team] >= MIN_APPEARANCES else OTHER

    teams = sorted({label(m.home) for m in matches} | {label(m.away) for m in matches})
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Pack each match into flat numpy arrays for a fast vectorised objective.
    hi = np.array([index[label(m.home)] for m in matches])
    ai = np.array([index[label(m.away)] for m in matches])
    x = np.array([m.home_goals for m in matches], dtype=float)
    y = np.array([m.away_goals for m in matches], dtype=float)
    w = np.array([m.weight for m in matches], dtype=float)
    is_home = np.array([0.0 if m.neutral else 1.0 for m in matches])

    # Parameter vector layout: [attack(n), defence(n), home_advantage, rho].
    ATT, DEF, HOME, RHO = slice(0, n), slice(n, 2 * n), 2 * n, 2 * n + 1
    PENALTY = 1e3  # weight of the mean-zero identifiability constraint

    def _pieces(p: np.ndarray):
        """Shared forward pass used by both the objective and its gradient."""
        attack, defence = p[ATT], p[DEF]
        lam = np.exp(attack[hi] - defence[ai] + p[HOME] * is_home)
        mu = np.exp(attack[ai] - defence[hi])
        tau = np.clip(_tau(x, y, lam, mu, p[RHO]), 1e-10, None)
        return attack, defence, lam, mu, tau

    def neg_log_likelihood(p: np.ndarray) -> float:
        attack, defence, lam, mu, tau = _pieces(p)
        # Poisson log-pmf without the constant -log(x!)-log(y!) terms (they do
        # not depend on the parameters, so they drop out of the argmin).
        loglik = np.log(tau) + x * np.log(lam) - lam + y * np.log(mu) - mu
        nll = -np.sum(w * loglik)
        # Identifiability: attack/defence are only defined up to a shared
        # constant, so pin their means to zero with a light quadratic penalty.
        nll += PENALTY * (attack.mean() ** 2 + defence.mean() ** 2)
        return nll

    def gradient(p: np.ndarray) -> np.ndarray:
        attack, defence, lam, mu, tau = _pieces(p)
        rho = p[RHO]
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)

        # Derivatives of tau w.r.t. lam, mu and rho (non-zero only on the four
        # corrected cells).
        dtau_dlam = np.where(m00, -mu * rho, np.where(m01, rho, 0.0))
        dtau_dmu = np.where(m00, -lam * rho, np.where(m10, rho, 0.0))
        dtau_drho = np.where(m00, -lam * mu, np.where(m01, lam,
                    np.where(m10, mu, np.where(m11, -1.0, 0.0))))

        # dLL/d(log lam) and dLL/d(log mu); the chain rule folds in dlam=lam.
        g_lam = w * (x - lam + lam / tau * dtau_dlam)
        g_mu = w * (y - mu + mu / tau * dtau_dmu)

        grad = np.zeros_like(p)
        # lam = exp(attack[home] - defence[away] + home_adv*is_home)
        np.add.at(grad, ATT.start + hi, g_lam)
        np.add.at(grad, DEF.start + ai, -g_lam)
        grad[HOME] += np.sum(g_lam * is_home)
        # mu = exp(attack[away] - defence[home])
        np.add.at(grad, ATT.start + ai, g_mu)
        np.add.at(grad, DEF.start + hi, -g_mu)
        grad[RHO] += np.sum(w * dtau_drho / tau)

        grad = -grad  # we minimise the negative log-likelihood
        # Gradient of the mean-zero penalty.
        grad[ATT] += PENALTY * 2 * attack.mean() / n
        grad[DEF] += PENALTY * 2 * defence.mean() / n
        return grad

    # Start from level ratings, a small home edge and mild draw inflation.
    p0 = np.zeros(2 * n + 2)
    p0[HOME] = 0.25
    p0[RHO] = -0.05

    # rho must stay in a range that keeps tau positive for realistic rates;
    # everything else is unbounded.
    bounds = [(None, None)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]

    result = minimize(
        neg_log_likelihood, p0, jac=gradient, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000},
    )
    p = result.x

    # Centre the ratings exactly (the penalty gets us close; this makes them
    # cleanly interpretable — a positive attack is above the world average).
    attack = p[ATT] - p[ATT].mean()
    defence = p[DEF] - p[DEF].mean()

    model = DixonColesModel(
        teams=teams,
        attack={t: float(attack[i]) for t, i in index.items()},
        defence={t: float(defence[i]) for t, i in index.items()},
        home_advantage=float(p[HOME]),
        rho=float(p[RHO]),
        n_matches=len(matches),
        total_weight=float(w.sum()),
    )

    if verbose:
        # Weighted mean goals per side in the training data — a calibration
        # sanity check the fitted rates should reproduce on average.
        avg_goals = float(np.sum(w * (x + y)) / (2 * w.sum()))
        print(
            f"Fitted Dixon-Coles on {len(matches):,} matches "
            f"({len(teams)} teams, effective weight {w.sum():,.0f})."
        )
        print(
            f"  home advantage = x{np.exp(model.home_advantage):.2f} on goals, "
            f"rho = {model.rho:+.3f} (negative => extra draws / low scores)."
        )
        print(f"  weighted average scoring in training: {avg_goals:.2f} goals per side.")

    return model


if __name__ == "__main__":
    # Standalone: fit and print the strongest attacks/defences. Handy for a
    # quick sanity check or a screen capture.
    m = fit_dixon_coles()
    top_att = sorted(m.attack.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_def = sorted(m.defence.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nBest attacks: ", ", ".join(f"{t} {v:+.2f}" for t, v in top_att))
    print("Best defences:", ", ".join(f"{t} {v:+.2f}" for t, v in top_def))
