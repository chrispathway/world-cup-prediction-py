# World Cup Oracle — Terminal Edition

A Python CLI that predicts the 2026 FIFA World Cup with a **Dixon-Coles** goal
model fitted on the last decade of international results.

It gives every team an **attack** and a **defence** rating, models a match as
two coupled goal distributions (with the Dixon-Coles low-score correction that
properly handles draws and 0-0 / 1-1 games), and runs a **Monte Carlo**
simulation of the full 48-team tournament (10,000 runs by default) to estimate
every team's odds of winning the title, reaching the final, and reaching the
semis.

Then it drops you into an interactive prompt where you can type two teams and
get the head-to-head prediction: win/draw/loss probabilities, expected goals,
and the most likely scoreline.

## Requirements

Python 3.10+ and a couple of scientific packages:

```bash
pip install -r requirements.txt   # numpy, scipy, certifi
```

## Run

```bash
python oracle.py                 # fit the model, simulate, then the prompt
python oracle.py --sims 2000     # fewer simulations = faster, a bit noisier
```

On first run it downloads the historical results dataset and caches it locally
(`.cache_results.csv`); later runs are offline. Delete that file to refresh.

### Backtest + Round-of-16 predictions

```bash
python evaluate_2026.py
```

This scores the model out-of-sample against every 2026 World Cup match already
played (accuracy, log-loss, Brier), then predicts three upcoming Round-of-16
fixtures. The model is trained only on data **before** the tournament, so none
of the games it is scored on ever leaked into training.

## No data leakage

Training is restricted to internationals from **11 June 2016 to 10 June 2026** —
the decade ending the day before the 2026 finals kick off. The 2026 World Cup's
own matches are excluded outright, so every 2026 game is a genuine out-of-sample
prediction.

## The model in brief

- **Recency weighting.** Each training match is weighted by an exponential
  time-decay with a ~3-year half-life, so last year's results count far more
  than results from ten years ago.
- **Competition weighting.** Competitive fixtures (World Cups, continental
  championships, qualifiers) outweigh friendlies.
- **Dixon-Coles goals.** For home team *i* vs away team *j*, expected goals are
  `λ = exp(attack_i − defence_j + home_adv)` and `μ = exp(attack_j − defence_i)`.
  The joint scoreline probability is `τ(x,y) · Poisson(x;λ) · Poisson(y;μ)`,
  where `τ` (governed by a single dependence parameter `ρ`) corrects the four
  lowest scorelines — fixing plain Poisson's habit of under-predicting draws.
- **Joint fit.** All attack/defence ratings, the home advantage and `ρ` are
  estimated *together* by maximising the weighted log-likelihood (L-BFGS-B with
  an analytic gradient).
- **Neutral knockouts.** World Cup matches are simulated on neutral ground, so
  neither side gets the home-advantage term.
- **Tournament.** 12 groups of 4 play round-robin; the top two of each group
  plus the eight best third-place teams advance to a 32-team knockout bracket.
  The bracket uses a **fixed template** that follows the real format — group
  winners are protected from each other in the Round of 32, and a group's winner
  and runner-up sit in opposite halves so they can only meet again in the final.
  Knockout ties are decided by a lightly strength-weighted shootout. Repeat
  10,000 times and count how often each team reaches each stage.

## Teams & groups

The 48 teams and their groups follow the **official FIFA final draw** held on
5 December 2025 in Washington, D.C.

| | | | |
|---|---|---|---|
| **A** Mexico · South Africa · South Korea · Czech Republic | **B** Canada · Bosnia & Herz. · Qatar · Switzerland | **C** Brazil · Morocco · Haiti · Scotland | **D** USA · Paraguay · Australia · Turkey |
| **E** Germany · Curaçao · Ivory Coast · Ecuador | **F** Netherlands · Japan · Sweden · Tunisia | **G** Belgium · Egypt · Iran · New Zealand | **H** Spain · Cape Verde · Saudi Arabia · Uruguay |
| **I** France · Senegal · Iraq · Norway | **J** Argentina · Algeria · Austria · Jordan | **K** Portugal · DR Congo · Uzbekistan · Colombia | **L** England · Croatia · Ghana · Panama |

## Using the prompt

```
> Brazil vs France      # head-to-head match prediction
> titles                # reprint the full title-odds table
> teams                 # list all 48 qualified teams + groups
> quit
```

Team names are matched loosely — `Brazil`, `BRA`, or `bra` all work.

## How it works

| File | Responsibility |
|------|----------------|
| `dataset.py` | Downloads results, restricts them to the training window, and computes the recency + competition weights. |
| `dixon_coles.py` | The Dixon-Coles goal model: weighted maximum-likelihood fit, scoreline grid, sampling. |
| `simulation.py` | Monte Carlo group stage and knockout bracket, driven by the fitted model. |
| `worldcup2026.py` | The 48 qualified teams, group assignments, and dataset name mapping. |
| `oracle.py` | The CLI — wires it together and renders the tables / prompt. |
| `evaluate_2026.py` | Out-of-sample backtest against played 2026 matches + Round-of-16 predictions. |

> Predictions are a probabilistic model for entertainment, not betting advice.
