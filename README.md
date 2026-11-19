# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aryatrifebrian423/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/aryatrifebrian423/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python stdlib forecaster for GTA VI trophy earn rates on Exophase: it reads a case's recorded reads of the game's trophy page and forecasts each trophy's rate at the target as five quantiles.

## How it works

Earners vs players, in counts space. A trophy's rate is holders / players tracked, and the two sides move for different reasons, so the program forecasts them separately and divides:

- **Players tracked** is fit as log players against log(1 + hours since release) on the distinct values the page has shown.
- **Holders** per trophy are fit two ways and blended: a log-log line/quadratic (short 24-hour window, so a saturating rise is not extrapolated forever) and a relaxing rate curve with an asymptote, which carries most of the weight because a rate is bounded and adoption relaxes toward an asymptote.
- **Stale page:** when a case's history shows no refresh at all (launch night), the program applies the first-refresh prior published in the rules (638 players / 943 trophies earned, then one refresh to 3,868 / 4,276) instead of persisting the frozen page.
- **Spread:** a normal fan on the logit scale whose width comes from the fits' residuals plus a horizon term, so a far target is honestly wider than a near one.

## Run it

```sh
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`). Standard library only, deterministic, about 0.1 s a case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2tg9044f&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
