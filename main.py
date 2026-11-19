"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at a case's target, as five quantiles, from the house's reads
of the game's trophy page before the case's checkpoint.

Model: earners vs players, in counts space
-------------------------------------------
A trophy's rate is a ratio, holders / players_tracked, and the two sides move
for different reasons, so the program forecasts them separately and divides:

* players_tracked P(t): how many players Exophase tracks. It only grows, in
  jumps: Exophase refreshes a game's counts on its own schedule, hours apart,
  and between refreshes the page is frozen. The program fits log P against
  log(1 + hours since release) on the distinct values the page has shown.
* holders H_i(t): how many tracked players hold trophy i. Holders only grow
  too, as players progress and as newly tracked players bring the trophies
  they already hold. The program fits log H_i against log(1 + hours) per
  trophy on the last 24 hours (a quadratic when the points bend down, so a
  saturating rise is not extrapolated forever), with each slope shrunk a
  little toward its tier's median slope, and blends that view with a
  relaxing rate curve fitted on the last 48 hours, which carries most of
  the weight because a rate is bounded and adoption relaxes toward an
  asymptote.

The forecast rate is H_i(target) / P(target). Working in counts keeps the
two forces apart: a trophy whose holders grow slower than the tracked
population sees its rate fall (dilution), one whose holders grow faster sees
it rise (progression), and the split is estimated from the case's own reads.

Stale page: when a case's history shows no refresh at all (every read the
same players and rates, as on launch night), the fits above are flat and say
"nothing changes" -- but a page that has sat unchanged for hours in launch
week is stale, not stable: the next refresh reveals hours of accumulated
players and holders at once. For that case the program uses the first
refresh the competition rules publish (launch night: 638 players and 943
trophies earned, then one refresh to 3,868 players and 4,276 earned, every
rate but the 100% one lower) as a prior: it scales each trophy's holders by
the refresh's holder multiple for the trophies the new cohort barely holds,
keeps the 100% trophy at 100%, and spreads the quantiles wide enough that
the 95% end still allows the page not to have refreshed by the target.

Spread: the quantiles are a normal fan on the logit scale around the median
rate, with a per-trophy width from the fits' residuals plus a horizon term
that grows with the forecast distance, so a far target is honestly wider
than a near one. The 100% trophy is forecast at ~100% with a narrow fan.

Only the Python standard library is used. The program is deterministic: no
randomness, no wall clock, no network; the same case always writes the same
bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- First-refresh prior (launch night, as published in the rules) -----------
# The house saw the page unchanged at 638 players tracked and 943 trophies
# earned from 2026-11-18T23:48Z through its 04:00Z read, then one refresh to
# 3,868 players and 4,276 trophies earned by its 04:30Z read, every rate but
# the one at 100% lower. These are the multiples a first refresh applies.
FIRST_REFRESH_PLAYERS = 3868.0 / 638.0   # ~6.06x players tracked
FIRST_REFRESH_EARNED = 4276.0 / 943.0    # ~4.53x trophies earned

# Spread (logit standard deviation) used when the history is flat: the
# refresh is expected but not certain, so the fan stays wide.
FLAT_SPREAD = 0.5

# Spread floor / cap for the data-driven branch (logit SD).
SPREAD_MIN = 0.45
SPREAD_MAX = 1.25
# Horizon term of the spread: 0.30 plus 0.10 per root day of horizon.
HORIZON_SPREAD_BASE = 0.30
HORIZON_SPREAD_PER_ROOT_DAY = 0.10


def clamp(x, lo=0.01, hi=99.99):
    """Clamp a percent into the range the logit is defined on."""
    return min(hi, max(lo, x))


def logit(pct):
    """logit of a percent (0..100), clamped first."""
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    """Percent (0..100) for a logit value."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z) into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def least_squares(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_sd).

    resid_sd is the standard deviation of the residuals (0 when the fit is
    exact or has fewer than 3 points). Returns None when the slope is
    undefined (fewer than 2 points or no spread in x).
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    if n < 3:
        return (a, b, 0.0)
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    return (a, b, math.sqrt(var))


def least_squares_quad(xs, ys):
    """Least-squares quadratic y = a + b*x + c*x^2. Returns (a, b, c, sd).

    Returns None with fewer than 4 points or when the normal equations
    are singular. Solves the 3x3 system by Gaussian elimination.
    """
    n = len(xs)
    if n < 4:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * (x ** k) for x, y in zip(xs, ys)) for k in range(3)]
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for cc in range(col, 4):
                m[r][cc] -= f * m[col][cc]
    coef = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        coef[r] = (m[r][3] - sum(m[r][cc] * coef[cc] for cc in range(r + 1, 3))) / m[r][r]
    a, b, c = coef
    var = sum((y - (a + b * x + c * x * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 3, 1)
    return (a, b, c, math.sqrt(var))


def saturating_rate_fit(hours, rates, h_target):
    """Fit a relaxing rate curve f(h) = L + (f0 - L) * exp(-(h - h0) / tau).

    A trophy's rate rises toward an asymptote as players finish what it
    asks for, and falls toward one when the tracked population outgrows
    its holders, so the curve is allowed to relax up (L above the first
    rate) or down (L below it). tau and L are grid-searched; the anchor
    f0 is the first rate. Returns (f_star, resid_sd) in percent / log
    units, or None when there are too few points.
    """
    if len(hours) < 3 or len({h for h in hours}) < 2:
        return None
    h0 = hours[0]
    f0 = max(rates[0], 0.01)
    fmax = max(rates)
    fmin = min(rates)
    levels = []
    for m in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0):
        levels.append(min(99.99, fmax * m))
    for m in (0.9, 0.75, 0.5, 0.25, 0.1):
        levels.append(max(0.01, fmin * m))
    taus = (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0)
    best = None
    for tau in taus:
        for level in levels:
            if abs(level - f0) < 1e-9:
                continue
            sse = 0.0
            for h, r in zip(hours, rates):
                pred = level + (f0 - level) * math.exp(-(h - h0) / tau)
                pred = max(pred, 0.01)
                d = math.log(max(r, 0.01)) - math.log(pred)
                sse += d * d
            if best is None or sse < best[0]:
                best = (sse, tau, level)
    if best is None:
        return None
    sse, tau, level = best
    f_star = level + (f0 - level) * math.exp(-(h_target - h0) / tau)
    resid = math.sqrt(sse / max(len(hours) - 2, 1))
    return (min(max(f_star, 0.01), 99.99), resid)

def load_case(path):
    """Read one case file and return (case, release, steps).

    release is the release instant (game.release_date at 00:00 UTC).
    steps is the history compressed to distinct values: consecutive reads
    that print the same players and rates are one step (the page frozen
    between refreshes), kept at the time the values first appeared. Each
    step is (hours_since_release, players, [rate per trophy]).
    """
    case = json.loads(path.read_text(encoding="utf-8"))
    release_text = case.get("game", {}).get("release_date", "2026-11-19")
    release = datetime.fromisoformat(release_text + "T00:00:00+00:00")
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)
    trophies = case["trophies"]
    n = len(trophies)
    steps = []
    for read in case.get("history", []):
        rates = read.get("rates") or []
        # Carry the last non-null rate forward for a trophy a read lacked.
        if steps:
            prev = steps[-1][2]
            rates = [
                (rates[i] if i < len(rates) and rates[i] is not None else prev[i])
                for i in range(n)
            ]
        else:
            rates = [
                (rates[i] if i < len(rates) and rates[i] is not None else 0.0)
                for i in range(n)
            ]
        players = read.get("players_tracked")
        if players is None:
            continue
        at = parse_time(read["at"])
        hours = (at - release).total_seconds() / 3600.0
        step = (hours, float(players), rates)
        if steps and steps[-1][1] == step[1] and steps[-1][2] == rates:
            continue  # same page as the last step: still frozen
        steps.append(step)
    return case, release, steps


def holders_of(rates, players):
    """Integer holder counts implied by rates (percent) and players tracked.

    Rates print to 2 decimals, so rate * players / 100 rounds to the whole
    count the page's numbers came from.
    """
    return [int(round(r * players / 100.0)) for r in rates]


def quantiles(median_pct, spread):
    """Five quantiles for a median rate: a normal fan on the logit scale.

    The 100% trophy gets a narrow fan at the top: it has stayed at 100%
    through every refresh, so only a small dip is hedged at the 5% end.
    """
    if median_pct >= 99.99:
        return [99.95, 99.99, 99.99, 99.99, 99.99]
    centre = logit(median_pct)
    out = [round4(clamp(inv_logit(centre + z * spread))) for z in Z]
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_flat(last_rates, last_players):
    """Forecast when the history shows no refresh at all.

    Applies the first-refresh prior: the tracked population jumps by
    FIRST_REFRESH_PLAYERS and the trophies earned by FIRST_REFRESH_EARNED.
    The 100% trophy keeps every new player, so it stays at 100%; the other
    trophies' holders grow only by the refresh's non-100% holder multiple,
    (earned' - players') / (earned - players), which is how much the newly
    tracked players hold of each trophy on average. Every other trophy's
    rate therefore falls to roughly a fifth of where it stood.
    """
    holders = holders_of(last_rates, last_players)
    earned = sum(holders)
    players_star = last_players * FIRST_REFRESH_PLAYERS
    earned_star = earned * FIRST_REFRESH_EARNED
    full = sum(1 for r in last_rates if r >= 99.995)
    other_scale = (earned_star - full * players_star) / max(
        earned - full * last_players, 1
    )
    rows = []
    for i, rate in enumerate(last_rates):
        if rate >= 99.995:
            rows.append((i, quantiles(99.99, FLAT_SPREAD)))
            continue
        holders_star = holders[i] * other_scale
        median_pct = 100.0 * holders_star / players_star
        rows.append((i, quantiles(median_pct, FLAT_SPREAD)))
    return rows


def forecast_trend(case, release, steps, x_target, horizon_days):
    """Forecast from a history that shows refreshes: fit log counts.

    Fits log players and log holders against log(1 + hours since release)
    on the steps of the last 48 hours before the checkpoint (all steps when
    there are fewer than three), then reads both fits at the target. Each
    trophy's holder slope is shrunk 20% toward its tier's median slope.
    Holders are floored at their last count (holders only grow) and capped
    at the forecast players (a rate cannot pass 100%).
    """
    trophies = case["trophies"]
    n = len(trophies)
    hours_at = steps[-1][0]
    hours_target = math.exp(x_target) - 1.0
    window = [s for s in steps if s[0] >= hours_at - 48.0]
    if len(window) < 3:
        window = steps
    xs = [math.log(1.0 + max(h, 0.0)) for h, _, _ in window]

    # Holder fits use a shorter window than the players fit: a trophy's
    # earn rate saturates as players finish what it asks for, so its
    # holder growth bends down over time, and a 48-hour line through the
    # steep early rise would extrapolate the rise forever. The last 24
    # hours track the growth regime the trophy is in now; fall back to
    # the 48-hour window when there are too few steps to fit on.
    window_h = [s for s in window if s[0] >= hours_at - 24.0]
    if len(window_h) < 4:
        window_h = window
    xs_h = [math.log(1.0 + max(h, 0.0)) for h, _, _ in window_h]

    fit_p = least_squares(xs, [math.log(s[1]) for s in window])
    if fit_p is None:
        return forecast_flat(window[-1][2], window[-1][1])
    a_p, b_p, resid_p = fit_p
    players_star = math.exp(a_p + b_p * x_target)
    players_star = max(players_star, window[-1][1])

    # Per-trophy holder fits, first pass (raw slopes), on the short window.
    # Each trophy gets a line and, when the points bend down, a quadratic:
    # holder growth saturates as players finish what a trophy asks for, so
    # log holders against log time is concave, and a line through the steep
    # early rise would extrapolate the rise forever. The quadratic is used
    # only when it bends down (c < 0) and fits at least as well as the line.
    raw = []
    for i in range(n):
        points = []
        for k, (_, players, rates) in enumerate(window_h):
            h = int(round(rates[i] * players / 100.0))
            if h > 0:
                points.append((xs_h[k], math.log(h)))
        fit = None
        if len(points) >= 2 and len({p[0] for p in points}) >= 2:
            px = [p[0] for p in points]
            py = [p[1] for p in points]
            lin = least_squares(px, py)
            quad = least_squares_quad(px, py)
            mean_x = sum(px) / len(px)
            mean_y = sum(py) / len(py)
            if quad is not None and quad[2] < 0 and quad[3] <= lin[2]:
                fit = ("quad", quad[0], quad[1], quad[2], quad[3], mean_x, mean_y)
            else:
                fit = ("lin", lin[0], lin[1], 0.0, lin[2], mean_x, mean_y)
        raw.append(fit)
    tier_slopes = {}
    for i, trophy in enumerate(trophies):
        fit = raw[i]
        if fit is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(fit[2])
    tier_median = {t: median(v) for t, v in tier_slopes.items()}

    horizon_spread = HORIZON_SPREAD_BASE + HORIZON_SPREAD_PER_ROOT_DAY * math.sqrt(
        max(horizon_days, 0.0)
    )
    fit_spread_p = min(0.5, max(0.10, resid_p))
    last_players = window[-1][1]
    last_rates = window[-1][2]
    rows = []
    for i, trophy in enumerate(trophies):
        fit = raw[i]
        last_h = int(round(last_rates[i] * last_players / 100.0))
        if fit is None:
            log_h_star = math.log(max(last_h, 1))
            resid = 0.15
        else:
            _, _, b_i, c_i, resid, mean_x, mean_y = fit
            # Shrink the slope 20% toward the tier median (own slope when
            # the tier has too few trophies to take a median from), then
            # re-anchor the fit at the data centre so the shrinkage pivots
            # there instead of at x = 0.
            own = tier_median.get(trophy["tier"], b_i)
            b_i = 0.8 * b_i + 0.2 * own
            b_i = min(3.0, max(-0.5, b_i))
            a_i = mean_y - b_i * mean_x - c_i * mean_x * mean_x
            log_h_star = a_i + b_i * x_target + c_i * x_target * x_target
            resid = min(0.7, max(0.15, resid))
        # Saturating-curve view of the same trophy, on the 48-hour window:
        # a relaxing rate has an asymptote, so it cannot run away the way
        # a log-log line can. The saturating view carries most of the
        # forecast (85%) because a rate is bounded and adoption relaxes
        # toward an asymptote; the log-log view keeps a 15% weight as a
        # hedge against a shape the curve family misses. When the views
        # disagree the spread widens by the disagreement.
        if last_rates[i] < 99.995:
            sat = saturating_rate_fit(
                [s[0] for s in window],
                [s[2][i] for s in window],
                hours_target,
            )
            if sat is not None:
                f_sat, resid_sat = sat
                log_h_sat = math.log(max(f_sat * players_star / 100.0, 1.0))
                disagree = abs(log_h_star - log_h_sat)
                log_h_star = 0.15 * log_h_star + 0.85 * log_h_sat
                resid = max(resid, min(0.7, resid_sat), min(0.5, 0.3 * disagree))
        holders_star = math.exp(log_h_star)
        holders_star = min(max(holders_star, last_h), players_star)
        if last_rates[i] >= 99.995 and last_h >= last_players:
            rows.append((i, quantiles(99.99, SPREAD_MIN)))
            continue
        median_pct = 100.0 * holders_star / players_star
        spread = math.sqrt(resid ** 2 + fit_spread_p ** 2 + horizon_spread ** 2)
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
        rows.append((i, quantiles(median_pct, spread)))
    return rows


def forecast_case(path):
    """Forecast one case file; returns the {"trophies": [...]} object."""
    case, release, steps = load_case(path)
    if case["levels"] != LEVELS:
        raise ValueError(f"{path.name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    if not steps:
        # No reads at all: say 1% everywhere with the widest fan. This
        # should not happen; it only keeps the case valid.
        return {
            "trophies": [
                {"id": t["id"], "q": quantiles(1.0, SPREAD_MAX)} for t in trophies
            ]
        }
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    hours_target = (target - release).total_seconds() / 3600.0
    x_target = math.log(1.0 + max(hours_target, 0.0))
    last_rates = steps[-1][2]
    last_players = steps[-1][1]
    if len(steps) < 2:
        rows = forecast_flat(last_rates, last_players)
    else:
        rows = forecast_trend(case, release, steps, x_target, horizon_days)
    return {
        "trophies": [
            {"id": trophies[i]["id"], "q": q} for i, q in sorted(rows)
        ]
    }

def main():
    """Read every <id>.in in the input directory, write <id>.out for each."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        forecast = forecast_case(path)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
