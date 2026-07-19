# 2026 World Cup Prediction (Live)

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


## 🏆 The Final — 🇪🇸 Spain vs Argentina 🇦🇷

The model makes the final a near coin-flip, with Argentina a whisker ahead:

| Outcome | 🇪🇸 Spain win | Draw (90') | Argentina win 🇦🇷 |
|---------|:------------:|:----------:|:----------------:|
| **Probability** | 32.5% | 33.2% | **34.2%** |

Expected goals: **Spain 0.93 – 0.96 Argentina**. Most likely scoreline: **0–0**
(15.7%) — the two meanest defences left standing, separated by almost nothing.
Among decisive outcomes it splits **Argentina 51% / Spain 49%**. A true toss-up.


## Live scorecard — model vs reality

Every 2026 World Cup match played so far, scored against the model. The model
was trained **only on data from before the tournament**, so each of these is a
genuine out-of-sample prediction. **Model prediction** shows the most likely
outcome with the home-win / draw / away-win probabilities (%); **Result** is a
✅ when the model's most likely outcome matched the actual winner (or a draw),
❌ otherwise. Knockout ties are scored on the 90-minute result, so a game level
after 90 counts as a draw regardless of who won the shootout.

**Outcome accuracy: 72 / 103 = 69.9%** correct calls, covering every match of
the tournament bar the final still to be played — the full group stage, the
Round of 32, the Round of 16, the quarter-finals, the semi-finals and the
third-place play-off.

| Match | Score | Model prediction (W / D / L %) | Result |
|-------|:-----:|--------------------------------|:------:|
| 🇲🇽 Mexico v South Africa 🇿🇦 | 2–0 | Mexico win (61 / 26 / 14) | ✅ |
| 🇰🇷 South Korea v Czech Republic 🇨🇿 | 2–1 | Czech Republic win (33 / 30 / 36) | ❌ |
| 🇨🇦 Canada v Bosnia & Herz. 🇧🇦 | 1–1 | Canada win (59 / 25 / 16) | ❌ |
| 🇺🇸 USA v Paraguay 🇵🇾 | 4–1 | USA win (38 / 32 / 30) | ✅ |
| 🇶🇦 Qatar v Switzerland 🇨🇭 | 1–1 | Switzerland win (7 / 15 / 78) | ❌ |
| 🇧🇷 Brazil v Morocco 🇲🇦 | 1–1 | Brazil win (41 / 36 / 23) | ❌ |
| 🇭🇹 Haiti v Scotland 🏴󠁧󠁢󠁳󠁣󠁴󠁿 | 0–1 | Scotland win (11 / 20 / 68) | ✅ |
| 🇦🇺 Australia v Turkey 🇹🇷 | 2–0 | Turkey win (34 / 30 / 36) | ❌ |
| 🇩🇪 Germany v Curaçao 🇨🇼 | 7–1 | Germany win (91 / 7 / 2) | ✅ |
| 🇨🇮 Ivory Coast v Ecuador 🇪🇨 | 1–0 | Ecuador win (20 / 37 / 42) | ❌ |
| 🇳🇱 Netherlands v Japan 🇯🇵 | 2–2 | Netherlands win (44 / 29 / 27) | ❌ |
| 🇸🇪 Sweden v Tunisia 🇹🇳 | 5–1 | Sweden win (42 / 32 / 27) | ✅ |
| 🇧🇪 Belgium v Egypt 🇪🇬 | 1–1 | Belgium win (56 / 29 / 16) | ❌ |
| 🇮🇷 Iran v New Zealand 🇳🇿 | 2–2 | Iran win (58 / 29 / 13) | ❌ |
| 🇪🇸 Spain v Cape Verde 🇨🇻 | 0–0 | Spain win (87 / 11 / 3) | ❌ |
| 🇸🇦 Saudi Arabia v Uruguay 🇺🇾 | 1–1 | Uruguay win (11 / 29 / 60) | ❌ |
| 🇫🇷 France v Senegal 🇸🇳 | 3–1 | France win (51 / 30 / 19) | ✅ |
| 🇮🇶 Iraq v Norway 🇳🇴 | 1–4 | Norway win (11 / 23 / 65) | ✅ |
| 🇦🇷 Argentina v Algeria 🇩🇿 | 3–0 | Argentina win (63 / 25 / 12) | ✅ |
| 🇦🇹 Austria v Jordan 🇯🇴 | 3–1 | Austria win (62 / 24 / 14) | ✅ |
| 🇵🇹 Portugal v DR Congo 🇨🇩 | 1–1 | Portugal win (66 / 24 / 10) | ❌ |
| 🇺🇿 Uzbekistan v Colombia 🇨🇴 | 1–3 | Colombia win (12 / 27 / 61) | ✅ |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England v Croatia 🇭🇷 | 4–2 | England win (49 / 30 / 20) | ✅ |
| 🇬🇭 Ghana v Panama 🇵🇦 | 1–0 | Ghana win (37 / 31 / 32) | ✅ |
| 🇨🇿 Czech Republic v South Africa 🇿🇦 | 1–1 | Czech Republic win (49 / 30 / 21) | ❌ |
| 🇲🇽 Mexico v South Korea 🇰🇷 | 1–0 | Mexico win (48 / 29 / 23) | ✅ |
| 🇨🇭 Switzerland v Bosnia & Herz. 🇧🇦 | 4–1 | Switzerland win (67 / 22 / 11) | ✅ |
| 🇨🇦 Canada v Qatar 🇶🇦 | 6–0 | Canada win (71 / 18 / 10) | ✅ |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland v Morocco 🇲🇦 | 0–1 | Morocco win (18 / 33 / 48) | ✅ |
| 🇧🇷 Brazil v Haiti 🇭🇹 | 3–0 | Brazil win (92 / 6 / 2) | ✅ |
| 🇺🇸 USA v Australia 🇦🇺 | 2–0 | USA win (37 / 30 / 33) | ✅ |
| 🇹🇷 Turkey v Paraguay 🇵🇾 | 0–1 | Turkey win (37 / 32 / 31) | ❌ |
| 🇩🇪 Germany v Ivory Coast 🇨🇮 | 2–1 | Germany win (58 / 26 / 16) | ✅ |
| 🇪🇨 Ecuador v Curaçao 🇨🇼 | 0–0 | Ecuador win (77 / 19 / 5) | ❌ |
| 🇳🇱 Netherlands v Sweden 🇸🇪 | 5–1 | Netherlands win (56 / 24 / 20) | ✅ |
| 🇹🇳 Tunisia v Japan 🇯🇵 | 0–4 | Japan win (20 / 33 / 48) | ✅ |
| 🇧🇪 Belgium v Iran 🇮🇷 | 0–0 | Belgium win (54 / 27 / 19) | ❌ |
| 🇳🇿 New Zealand v Egypt 🇪🇬 | 1–3 | Egypt win (15 / 35 / 50) | ✅ |
| 🇪🇸 Spain v Saudi Arabia 🇸🇦 | 4–0 | Spain win (83 / 13 / 4) | ✅ |
| 🇺🇾 Uruguay v Cape Verde 🇨🇻 | 2–2 | Uruguay win (65 / 26 / 9) | ❌ |
| 🇫🇷 France v Iraq 🇮🇶 | 3–0 | France win (75 / 19 / 6) | ✅ |
| 🇳🇴 Norway v Senegal 🇸🇳 | 3–2 | Norway win (39 / 31 / 29) | ✅ |
| 🇦🇷 Argentina v Austria 🇦🇹 | 2–0 | Argentina win (58 / 28 / 14) | ✅ |
| 🇯🇴 Jordan v Algeria 🇩🇿 | 1–2 | Algeria win (17 / 25 / 58) | ✅ |
| 🇵🇹 Portugal v Uzbekistan 🇺🇿 | 5–0 | Portugal win (66 / 24 / 10) | ✅ |
| 🇨🇴 Colombia v DR Congo 🇨🇩 | 1–0 | Colombia win (61 / 27 / 12) | ✅ |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England v Ghana 🇬🇭 | 0–0 | England win (77 / 17 / 5) | ❌ |
| 🇵🇦 Panama v Croatia 🇭🇷 | 0–1 | Croatia win (10 / 20 / 70) | ✅ |
| 🇲🇽 Mexico v Czech Republic 🇨🇿 | 3–0 | Mexico win (47 / 28 / 25) | ✅ |
| 🇿🇦 South Africa v South Korea 🇰🇷 | 1–0 | South Korea win (22 / 31 / 47) | ❌ |
| 🇨🇦 Canada v Switzerland 🇨🇭 | 1–2 | Switzerland win (28 / 28 / 44) | ✅ |
| 🇧🇦 Bosnia & Herz. v Qatar 🇶🇦 | 3–1 | Bosnia & Herz. win (43 / 29 / 27) | ✅ |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland v Brazil 🇧🇷 | 0–3 | Brazil win (12 / 23 / 66) | ✅ |
| 🇲🇦 Morocco v Haiti 🇭🇹 | 4–2 | Morocco win (79 / 16 / 5) | ✅ |
| 🇺🇸 USA v Turkey 🇹🇷 | 2–3 | USA win (38 / 27 / 36) | ❌ |
| 🇵🇾 Paraguay v Australia 🇦🇺 | 0–0 | Draw (30 / 36 / 33) | ✅ |
| 🇨🇼 Curaçao v Ivory Coast 🇨🇮 | 0–2 | Ivory Coast win (9 / 24 / 67) | ✅ |
| 🇪🇨 Ecuador v Germany 🇩🇪 | 2–1 | Germany win (27 / 32 / 41) | ❌ |
| 🇯🇵 Japan v Sweden 🇸🇪 | 1–1 | Japan win (44 / 29 / 27) | ❌ |
| 🇹🇳 Tunisia v Netherlands 🇳🇱 | 1–3 | Netherlands win (14 / 27 / 59) | ✅ |
| 🇪🇬 Egypt v Iran 🇮🇷 | 1–1 | Iran win (28 / 35 / 37) | ❌ |
| 🇳🇿 New Zealand v Belgium 🇧🇪 | 1–5 | Belgium win (6 / 17 / 77) | ✅ |
| 🇨🇻 Cape Verde v Saudi Arabia 🇸🇦 | 0–0 | Draw (29 / 36 / 35) | ✅ |
| 🇺🇾 Uruguay v Spain 🇪🇸 | 0–1 | Spain win (18 / 31 / 51) | ✅ |
| 🇳🇴 Norway v France 🇫🇷 | 1–4 | France win (23 / 27 / 50) | ✅ |
| 🇸🇳 Senegal v Iraq 🇮🇶 | 5–0 | Senegal win (55 / 31 / 14) | ✅ |
| 🇩🇿 Algeria v Austria 🇦🇹 | 3–3 | Austria win (31 / 30 / 39) | ❌ |
| 🇯🇴 Jordan v Argentina 🇦🇷 | 1–3 | Argentina win (4 / 14 / 82) | ✅ |
| 🇨🇴 Colombia v Portugal 🇵🇹 | 0–0 | Portugal win (31 / 30 / 39) | ❌ |
| 🇨🇩 DR Congo v Uzbekistan 🇺🇿 | 3–1 | Draw (29 / 40 / 31) | ❌ |
| 🇵🇦 Panama v England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 0–2 | England win (4 / 14 / 82) | ✅ |
| 🇭🇷 Croatia v Ghana 🇬🇭 | 2–1 | Croatia win (65 / 23 / 11) | ✅ |
| 🇿🇦 South Africa v Canada 🇨🇦 | 0–1 | Canada win (21 / 31 / 48) | ✅ |
| 🇧🇷 Brazil v Japan 🇯🇵 | 2–1 | Brazil win (53 / 29 / 19) | ✅ |
| 🇩🇪 Germany v Paraguay 🇵🇾 | 1–1 | Germany win (55 / 28 / 18) | ❌ |
| 🇳🇱 Netherlands v Morocco 🇲🇦 | 1–1 | Draw (34 / 34 / 31) | ✅ |
| 🇨🇮 Ivory Coast v Norway 🇳🇴 | 1–2 | Norway win (22 / 30 / 49) | ✅ |
| 🇫🇷 France v Sweden 🇸🇪 | 3–0 | France win (61 / 24 / 16) | ✅ |
| 🇲🇽 Mexico v Ecuador 🇪🇨 | 2–0 | Draw (30 / 35 / 35) | ❌ |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England v DR Congo 🇨🇩 | 2–1 | England win (66 / 26 / 8) | ✅ |
| 🇧🇪 Belgium v Senegal 🇸🇳 | 3–2 | Belgium win (47 / 30 / 23) | ✅ |
| 🇺🇸 USA v Bosnia & Herz. 🇧🇦 | 2–0 | USA win (58 / 25 / 17) | ✅ |
| 🇪🇸 Spain v Austria 🇦🇹 | 3–0 | Spain win (62 / 24 / 14) | ✅ |
| 🇵🇹 Portugal v Croatia 🇭🇷 | 2–1 | Portugal win (48 / 28 / 24) | ✅ |
| 🇨🇭 Switzerland v Algeria 🇩🇿 | 2–0 | Switzerland win (45 / 28 / 26) | ✅ |
| 🇦🇺 Australia v Egypt 🇪🇬 | 1–1 | Draw (36 / 36 / 28) | ✅ |
| 🇦🇷 Argentina v Cape Verde 🇨🇻 | 3–2 | Argentina win (82 / 15 / 3) | ✅ |
| 🇨🇴 Colombia v Ghana 🇬🇭 | 1–0 | Colombia win (73 / 20 / 8) | ✅ |
| 🇨🇦 Canada v Morocco 🇲🇦 | 0–3 | Morocco win (17 / 34 / 49) | ✅ |
| 🇵🇾 Paraguay v France 🇫🇷 | 0–1 | France win (14 / 29 / 57) | ✅ |
| 🇧🇷 Brazil v Norway 🇳🇴 | 1–2 | Brazil win (54 / 26 / 20) | ❌ |
| 🇲🇽 Mexico v England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 2–3 | England win (18 / 29 / 53) | ✅ |
| 🇵🇹 Portugal v Spain 🇪🇸 | 0–1 | Spain win (27 / 28 / 44) | ✅ |
| 🇺🇸 USA v Belgium 🇧🇪 | 1–4 | Belgium win (21 / 25 / 54) | ✅ |
| 🇦🇷 Argentina v Egypt 🇪🇬 | 3–2 | Argentina win (64 / 27 / 9) | ✅ |
| 🇨🇭 Switzerland v Colombia 🇨🇴 | 0–0 | Colombia win (27 / 29 / 43) | ❌ |
| 🇫🇷 France v Morocco 🇲🇦 | 2–0 | France win (38 / 36 / 26) | ✅ |
| 🇪🇸 Spain v Belgium 🇧🇪 | 2–1 | Spain win (51 / 27 / 22) | ✅ |
| 🇳🇴 Norway v England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 1–2 | England win (20 / 28 / 52) | ✅ |
| 🇦🇷 Argentina v Switzerland 🇨🇭 | 3–1 | Argentina win (54 / 29 / 17) | ✅ |
| 🇫🇷 France v Spain 🇪🇸 | 0–2 | Spain win (27 / 29 / 44) | ✅ |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England v Argentina 🇦🇷 | 1–2 | Draw (28 / 36 / 36) | ❌ |
| 🇫🇷 France v England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 4–6 | England win (30 / 33 / 37) | ✅ |

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

### Backtest + final prediction

```bash
python evaluate_2026.py
```

This scores the model out-of-sample against every 2026 World Cup match already
played (accuracy, log-loss, Brier), then predicts the one fixture left — the
final. The model is trained only on data **before** the tournament, so none of
the games it is scored on ever leaked into training.

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

