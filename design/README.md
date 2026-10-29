# FloodLens design reference

Static HTML mockups of the four screens. Open any file in a browser; the links between them work. They are a **visual reference** for the Next.js app, not code to copy: build real components with the same layout, type, colours and spacing.

| File | Screen |
| --- | --- |
| `01-new-assessment.html` | Address or pin, loan amount and tenor, site note, site photos, map with 250 m / 500 m rings |
| `02-agent-at-work.html` | Live steps, evidence images, agent log, elapsed time and cost |
| `03-risk-report.html` | Tier, score, action, verified badge, drivers, reported flooding nearby, site photos, evidence, limits |
| `04-portfolio.html` | CSV upload, counts and exposure per tier, loans by risk, map of loans by tier |

Text in [brackets] is a placeholder for real data.

## Design tokens

- Type: IBM Plex Sans (UI), IBM Plex Mono (numbers, coordinates, dates)
- Ink `#0F1E2D`, secondary text `#4D5B69`, ground `#F5F6F4`, cards `#FFFFFF`, borders `#D9DEE3`, row rules `#EDF0F2`
- Accent (actions, links, current step) `#1D5FA8`, hover `#123E70`, tint `#E8F0F9`
- Tier chips (background / text): Low `#E3F2EC` / `#1A6B4F` · Moderate `#FBEFD5` / `#7A4F00` · High `#FCE4D6` / `#9A3B0B` · Severe `#9B1C1C` / `#FFFFFF`
- Mask colours: water `#3FA9F5`, buildings and drains `#F2A33A`
- Cards radius 12 px, buttons 10 px, chips 999 px; touch targets at least 44 px

## Where the mockups are out of date (follow the current design in CLAUDE.md instead)

The screens were drawn before validation changed the approach. When building:

1. **Agent at work (02):** the steps should follow the real pipeline: Area screen → Flood history → Site photos → Score → Explain → Verify. The two images show SAM masks on satellite tiles; FloodLens does not segment satellite tiles. Show the code-computed water maps (water now, wet date, water history) instead, and SAM masks only on site photos.
2. **Risk report (03):** the drivers list should come from the real metrics and their contributions, grouped as Area screen, Local flood history and Site evidence. Show a clear banner when an evidence group is missing, with confidence capped at low. When there are no street-level reports, show exactly: "No street-level flood reports found. This does not mean the site is safe."
3. **All screens:** tier and action labels come from the API's `Tier` and `Action` values; never hard-code a tier.
