# FloodLens validation log

Every validation run so far, in order, with what we changed after each. Kept because the
useful findings here are the negative ones, and they are easy to forget once a table looks
better.

**Read this first.** All of it rests on **four** flood sites and a **30-point** random
baseline in one city. Nothing below is significant. It is enough to rule things out, which
is what it has mostly been used for.

- Sites: `data/validation_sites.csv`. Command: `scripts/validate.py --baseline 30 --history`.
- Baseline: random points in 6.40-6.70 N, 3.10-3.60 E with WorldCover built-up >= 50%
  within 250 m, centre pixel not open water, seed 20261027.
- AUC here means: over every (flood site, baseline point) pair, how often the flood site
  scores higher. 0.5 is chance.

## The circularity caveat, which applies to every row below

The four flood sites were **found in news reports**, and one of the three scoring groups
**searches news reports**. Of course they are found. A high score on these sites is close
to meaningless on its own.

The only honest number is how often a **random** address also turns up a street-level
report. If that rate were as high as the flood sites', the signal would be worthless no
matter how good the AUC looked.

## Run 1 - satellite only, water-proximity features

| site | score | tier |
| --- | --- | --- |
| Ago Palace Way, Okota | 0.0 | Low |
| James Kunle St, Isashi | 2.1 | Low |
| Ojota-Maryland bridge | 1.2 | Low |
| Lekki Phase 1 | 30.9 | Moderate |

Four flood-verified sites, three scored Low. Every feature measured proximity to a body of
water, and three of the four sites have **no water within 2 km**: `m_dist_water` and
`m_height_above_water` were not merely low, they were *missing*, so 35% of the weight
silently rescaled away. The one site that scored was the one beside a lagoon.

**Diagnosis.** The formula could only see riverine and coastal flooding. These sites flood
because rain cannot drain away. That is pluvial, and it carried zero weight.

**Changed:** "no water in view" became a measurement (`m_dist_water` saturates at the 1 km
grid edge; only `m_height_above_water` is dropped). Added `m_local_low_spot`,
`m_built_up_500`, `m_wetland_500`. Rebalanced 0.45 pluvial / 0.45 riverine / 0.10 rain.
Radar window widened 6 -> 12 days.

## Run 2 - satellite, rebalanced. **AUC 0.578**

| site | score | tier | percentile in baseline |
| --- | --- | --- | --- |
| Ago Palace Way, Okota | 28.0 | Moderate | 43.3 |
| Ojota-Maryland bridge | 36.7 | Moderate | 60.0 |
| Lekki Phase 1 | 39.4 | Moderate | 70.0 |

Baseline (n=30): min 14.3, p25 20.7, median 31.0, p75 39.9, **max 60.2**.

Mean flood score rose from 8.5 to 34.8 and this looked like progress. **It was not.** The
baseline rose too. A random built-up point outscored a flood site in 42.2% of pairs, Okota
sat *below* the baseline median, and the highest-scoring point in Lagos was a random one.

Per feature, nothing separated: `m_water_history_500`, `m_wet_water_250` and
`m_wetland_500` were ~0 across the whole city (0.35 of the weight doing no work),
`m_dist_water` was saturated for 21 of 30 baseline points, and the two features that did
vary varied as much in the baseline as among the flood sites (`m_local_low_spot` baseline
p25 -3.5 m vs flood -1.0 / -1.7 / -9.2).

**Diagnosis.** Open satellite data cannot tell a flooding Lagos street from a dry one.
Drainage decides, and drainage is not visible from orbit.

**Changed:** what the product claims. The satellite score was demoted to an **area screen**
for coastal and riverine exposure, and local flood history from news was added as a
separate group. Weights: local history 0.40, area screen 0.50, rainfall 0.10.

## Run 3 - with local flood history. **AUC 0.622**

| site | score | tier | reports | percentile |
| --- | --- | --- | --- | --- |
| Ago Palace Way, Okota | 56.5 | High | 3.0 | 100.0 |
| James Kunle St, Isashi | 35.9 | Moderate | 1.2 | (excluded, weak pin) |
| Ojota-Maryland bridge | 33.9 | Moderate | 1.0 | 56.7 |
| Lekki Phase 1 | 21.9 | Low | 0.0 | 30.0 |

Baseline (n=30): min 7.1, p25 18.4, median 30.4, p75 37.2, max 55.1.

**The circularity check:**

| | any report | street-level report |
| --- | --- | --- |
| flood sites | 3/4 | **2/4** |
| random baseline | 27/30 (90%) | **5/30 (17%)** |

AUC improved, but it rests on one site. Two findings matter more:

1. **Area-level matching is worthless.** 27 of 30 random addresses have *some* flood
   report, because an article naming a district matches everywhere inside it.
2. **Lekki's zero was our bug.** Nominatim at zoom 18 returned
   `Olarenwaju Ninalowo Crescent, Maroko`, so the search never looked for "Lekki Phase 1"
   and missed the documented 2024 flood.

Verification earned its place: across 70 reports it caught one URL the model typed that
the search tool never retrieved. Per-place caching took a second full run from $1.71 to
$0.05.

**Also found, by reading the output rather than trusting it:** transient Planetary Computer
failures had silently dropped the whole coastal/riverine group on two sites, rescaling 0.25
of the weight and inflating Okota from 56.5 to 69.2 with no warning anywhere. `validate.py`
now prints a `!` per degraded site.

**Changed:** area match weight -> 0 (reports still listed as context); page-text
verification against **our** reverse-geocoded names rather than the model's wording, with
abbreviation and whole-word handling, so `match_level` is evidence not opinion; a coarser
second reverse geocode for every site; event merging window 7 -> 3 days; baseline points
within 1 km of a validation site resampled; one retry per data layer, with a whole missing
evidence group forcing a visible note and low confidence; and the mandatory sentence "No
street-level flood reports found. This does not mean the site is safe." enforced by the
verifier.

## Run 4 - frozen scoring. **AUC 0.856**

| site | score | tier | reports | percentile |
| --- | --- | --- | --- | --- |
| Ago Palace Way, Okota | 42.5 | Moderate | 2.0 | 93.3 |
| James Kunle St, Isashi | 19.1 | Low | 0.0 | (excluded, weak pin) |
| Ojota-Maryland bridge | 56.4 | High | 2.6 | 100.0 |
| Lekki Phase 1 | 29.9 | Moderate | 0.6 | 63.3 |

Baseline (n=30): min 7.1, p25 19.1, median 23.4, p75 33.5, max 53.1. A random point
outscores a flood site in 14.4% of 90 pairs. No site was degraded or abandoned.

**Read the jump from 0.622 carefully. Three reasons it is not as good as it looks.**

1. **The baseline is a different sample.** 5 points were rejected for being in Ogun State,
   not Lagos — the bounding box reaches past the state line, so earlier baselines were
   partly not Lagos at all. Run 3 and Run 4 AUCs are not strictly comparable.
2. **Most of the gain is the baseline falling, not the flood sites rising.** Baseline
   median went 30.4 -> 23.4, because area-level matches now score 0 and most random
   addresses only ever had area-level reports.
3. **One site carries a lot of it.** Ojota went 33.9 -> 56.4, promoted to street level when
   the page check found our street name on the page.

**The number that actually improved, and the one worth trusting:**

| | any report | street-level report |
| --- | --- | --- |
| flood sites | 4/4 | **2/4** |
| random baseline | 23/30 | **2/30** (was 5/30) |

Checking the page against *our* reverse-geocoded street names rather than the model's
wording rejected 3 of the 5 baseline street-level matches from Run 3. That is a real gain
in selectivity, independent of the weights.

**Lekki is now actively hurt by its pin.** Two genuine Lekki flood reports were *rejected*
because the page named none of our place names: the reverse geocode returns `Ilado` and
`Eti Osa` for that point, never "Lekki Phase 1". The page check is working exactly as
designed and throwing away true positives, because the names it is given are wrong. A
hand-placed pin fixes this; the coarse zoom-14 lookup did not (it also returns `Ilado`).

**Isashi fell to 0 reports**, its estate-level matches demoted to area by the page check.

Cost: 34 places, 22 from cache, 488 searches, **$0.74**.

**A reporting bug found while reading this run.** The summary said "27 dropped", but that
list mixed real rejections with "kept, but demoted" match-level corrections, overstating
rejections roughly threefold. On the 12 places newly searched this run, 13 such lines split
**4 rejections / 9 corrections**. `FloodHistory` now keeps `dropped` and `corrections`
apart and the summary counts them separately.

## Scoring is frozen as of Run 4

No weight, ramp or match-level change without new, independent validation sites. See
CLAUDE.md. Everything below is a measurement problem, not a weighting problem.

## Standing findings

- **Radar cannot see urban flooding.** With `wet_date` pointing Sentinel-1 at the flood
  date *itself*, every flood site read 0.0-0.9% water. Building double-bounce defeats it.
  `m_wet_water_250` holds 0.10 weight and contributes almost nothing in the city. A
  wet-minus-dry change measure is the next thing to try.
- **Rainfall cannot separate addresses.** Open-Meteo is a ~10 km grid: 50-79 mm across the
  entire baseline. It separates years, not sites. Its 60 mm threshold is probably wrong for
  Lagos (these floods happened on 56-75 mm days) but lowering it on this evidence would be
  fitting to the validation set.
- **No dry sites yet.** Every labelled site in the set is a flood site. A random built-up
  point is **not** a dry site: Lagos floods widely, so the negative class is contaminated
  and every AUC here understates performance by an unknown amount.
- **Two weak pins, and they now cost real evidence.** `isashi_light_estate` is a rounded
  OSM village node (marked `geocode_quality=approximate` and excluded from summaries);
  `lekki_phase_1` is a neighbourhood centroid whose reverse geocode never yields "Lekki
  Phase 1", so the page check rejects genuine Lekki flood reports. Hand-placed pins are now
  the single cheapest improvement available.
- **The Lagos bounding box is not Lagos.** 6.40-6.70 N reaches into Ogun State; baseline
  points are now kept only if their reverse geocode says Lagos, and 5 of 35 draws were
  rejected on that test.

## What would actually move this

Not weights. In rough order of expected value:

1. Real dry sites, and hand-placed pins for the two weak ones.
2. A drainage measurement: distance to a drain or canal, or upslope contributing area from
   the DEM. The thing that decides the outcome is the thing we do not measure.
3. Wet-minus-dry radar change detection, to make `m_wet_water_250` earn its weight.
4. More sites, from a source that is not news, so the history group can be tested without
   circularity.
