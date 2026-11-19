# GTA VI Trophy Rarity Forecast

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dunoil/makefun-gta6-trophy-rarity-forecast-v2/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dunoil/makefun-gta6-trophy-rarity-forecast-v2/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for the make.fun competition **gta6-trophy-rarity-forecast-01**: forecast every
Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z,
as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Program

`main.ts` (Node 22, standard library only, no network, deterministic) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out`
forecast to `$MAKEFUN_OUTPUT_DIR` (else `/output/`).

Run it locally on the practice cases:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Approach

Exophase prints each trophy's rate as earners divided by players tracked, and a refresh
grows players tracked in a jump while earners move smoothly, so a refresh dilutes most
rates at once (launch night: 638 tracked players, unchanged for hours, then 3,868 at
the first refresh, every rate but the 100% one lower). Forecasting the rate directly
with a trend misses that mechanism, so this program forecasts earners and tracked
players separately and takes their ratio:

- Per trophy, least squares fits of `log(earners)` against `ln(1 + hours since
  2026-11-19T00:00:00Z)` over the last 24 hours of reads, plus one shared fit of
  `log(players_tracked)` against the same axis. The median at the target is
  `100 * exp(logE*) / exp(logP*)`, with slopes clamped so a short window cannot
  extrapolate into absurdity. The 100% trophy falls out naturally: its earners
  equal players tracked in every read, so its ratio stays at 100%.
- The spread (logit sigma) combines the residual std of a logit(rate) trend fit, a
  random walk of 0.5 logit units per square-root day, and a refresh allowance that
  applies only while players tracked has never changed in the history (a stale page
  with a first refresh pending). Once the history shows a refresh, the trend fits
  carry that process and the allowance switches off.
- Fallbacks for trophies with too few usable reads, null rates, and missing history;
  rates clamped to [0.01, 99.99] before any logit; quantiles non-decreasing.

## Validation

The practice answers are not in the kit, so the refresh test is synthetic, built only
from the facts the rules state (at 04:30Z players tracked was 3,868, earners frozen
gives answer = last rate * 638/3868):

- Holdout on the practice histories (fit to 01:30Z, forecast held-out reads at 02:00Z
  and 02:30Z): mean pinball loss **0.200**, the honest cost of the refresh allowance
  when no refresh happens.
- Synthetic-refresh test on p00/p01: this program **0.761**, house baseline **1.688**.
  It also wins under milder assumptions (3x dilution: 0.326 vs 0.996; 6.06x dilution
  with earners up 20%: 0.634 vs 1.508).
- Synthetic multi-refresh case (earners growing, players jumping 2x twice, 4.5-day
  horizon): median absolute logit error **0.013**, worst trophy 0.070, pinball 0.214.
- Two runs on both practice cases are byte-identical; every output is valid (55 rows,
  five non-decreasing quantiles in [0, 100], under 65,536 bytes); runs in about 0.2s
  per case.
