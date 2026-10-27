# Pre-build test: can SAM 3.1 find water in 10 m satellite imagery?

Run on Oct 26, 2026 (ET), after kickoff and before the first commit, with a throwaway script.

**Setup:** central Lagos (6.4541 N, 3.3947 E). Sentinel-2 L2A scene `S2C_MSIL2A_20260124T100311_R122_T31NEH_20260124T132918` (24 Jan 2026), picked as the clearest of 12 dry-season dates over the tile. Sentinel-1 RTC scene from 22 Jul 2025. Each image resampled to 1024 px and sent to `sam-3.1` with one noun phrase per request. Reference water: Sentinel-2 scene classification water class plus clear-sky NDWI > 0.1.

| Tile | Image | Prompt | Masks | Cover | IoU vs reference |
| --- | --- | --- | --- | --- | --- |
| 2 km (9.6% cloud) | true colour | water, river, lagoon, flooded area | 0 | 0% | 0 |
| 2 km | true colour | building, road | 0 | 0% | n/a |
| 2 km | water index, radar | all water prompts | 0 | 0% | 0 |
| 5 km (7.8% cloud) | true colour | water | 1 | 29.7% (ref 44.2%) | **0.66** |
| 5 km | true colour | river, lagoon, flooded area, building, road | 0 | 0% | 0 |
| 5 km | water index, radar | all water prompts | 0 | 0% | 0 |

**Cloud:** of 12 dry-season dates checked (Nov 2025 to Feb 2026), only 24 Jan 2026 had under 15% cloud over the 2 km tile.

**Radar:** a fixed -18 dB threshold found 8.5% water at 5 km where optical found 44%. FloodLens now fits an Otsu threshold per scene, kept between -24 and -14 dB.

## Decisions

1. Water at the site is measured in code (Sentinel-2 SCL + NDWI, Sentinel-1 radar, JRC history), not by SAM.
2. SAM 3.1 is used where it is strong: the officer's site photos, and a 5 km true-colour context map with the prompt "water".
3. Radar and JRC history are core inputs, because a clear optical image is rare in Lagos.
