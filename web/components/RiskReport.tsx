"use client";

import { useState } from "react";
import {
  GROUP_BLURB,
  NO_STREET_REPORTS,
  type AssessmentView,
  type EventGroup,
  type FloodReport,
  type GroupView,
  type MatchLevel,
  metricText,
} from "@/lib/types";
import {
  Card,
  Label,
  MissingEvidenceBanner,
  Mono,
  TierChip,
  VerifiedBadge,
  coords,
  money,
} from "@/components/ui";

const ACTION_WORDS: Record<string, string> = {
  approve: "Approve",
  approve_note_risk: "Approve, noting the flood risk",
  approve_with_insurance: "Approve with flood insurance",
  adjust_terms: "Adjust the terms",
  decline_or_change_collateral: "Decline, or change the collateral",
};

const MATCH_WORDS: Record<MatchLevel, string> = {
  street: "This street",
  estate: "This estate",
  area: "This area only",
};

const MATCH_STYLE: Record<MatchLevel, string> = {
  street: "bg-high-bg text-high-ink",
  estate: "bg-moderate-bg text-moderate-ink",
  area: "bg-rule text-muted",
};

const EVIDENCE_CAPTION: Record<string, string> = {
  optical: "Satellite view, clearest recent date",
  water_now: "Water detected on that clear date",
  water_history: "How often water has been here since 1984",
  water_wet: "Radar water on the wettest recent date",
};

/** The three water views are masks: white is water, black is none. On a dry inland site
 *  they are legitimately almost entirely black, which reads as a broken image unless the
 *  caption says what black means. */
const MASK_VIEWS = new Set(["water_now", "water_history", "water_wet"]);

function Metrics({ group }: { group: GroupView }) {
  if (!group.present)
    return <p className="text-sm text-muted">Not measured for this site.</p>;
  return (
    <ul className="flex flex-col">
      {group.metrics.map((m, i) => (
        <li key={m.id} className={`flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-2.5 ${i ? "border-t border-rule" : ""}`}>
          <span className="min-w-0 flex-1 text-sm">{m.label}</span>
          <Mono className="text-sm font-semibold">{metricText(m)}</Mono>
          {group.contributions[m.id] != null && (
            <Mono className="w-full text-xs text-muted sm:w-auto">
              +{group.contributions[m.id].toFixed(1)} pts
            </Mono>
          )}
        </li>
      ))}
    </ul>
  );
}

function EventCard({ group }: { group: EventGroup }) {
  const n = group.reports.length;
  return (
    <section className="rounded-[12px] border border-line">
      <header className="flex flex-wrap items-baseline justify-between gap-2 border-b border-rule px-3.5 py-2.5">
        <Mono className="text-sm font-semibold">{group.event_date}</Mono>
        <span className="text-xs text-muted">
          {n} {n === 1 ? "report" : "reports"}
        </span>
      </header>
      <div className="flex flex-col gap-3 p-3.5">
        {group.reports.map((r) => (
          <ReportCard key={`${r.url}-${r.publisher}`} report={r} />
        ))}
      </div>
    </section>
  );
}

function ReportCard({ report }: { report: FloodReport }) {
  return (
    <article>
      <div className="flex flex-wrap items-center gap-2">
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${MATCH_STYLE[report.match_level]}`}>
          {MATCH_WORDS[report.match_level]}
        </span>
        {report.text_verified === "unknown" && (
          <span className="rounded-full bg-rule px-2.5 py-0.5 text-xs text-muted">
            page could not be read
          </span>
        )}
      </div>
      <blockquote className="mt-2.5 border-l-2 border-line pl-3 text-sm italic text-ink">
        “{report.quote}”
      </blockquote>
      <p className="mt-2.5 text-xs text-muted">
        {report.publisher} ·{" "}
        <a href={report.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
          Read the source
        </a>
      </p>
    </article>
  );
}

export function RiskReport({ view }: { view: AssessmentView }) {
  const [openPhoto, setOpenPhoto] = useState<string | null>(null);
  const a = view.assessment;
  if (!a) return null;
  const { score } = a;
  const report = view.report;
  const byId = Object.fromEntries(a.metrics.map((m) => [m.id, m]));

  const layers = Object.values(view.water_layers);
  const noWater = layers.length > 0 && layers.every((found) => !found);
  const opticalCaption = view.optical_date
    ? `, ${view.optical_date}${
        view.optical_cloud_pct != null ? `, ${view.optical_cloud_pct.toFixed(0)}% cloud` : ""
      }`
    : "";

  // One list, no repeats. The model paraphrases the standing limits in its own words, so
  // matching on wording is not enough: both lists are bucketed by topic and the
  // code-written line wins, because it is the precise one. The mandatory sentence is
  // pinned first and never deduped away.
  const TOPICS: [string, RegExp][] = [
    ["height", /height above/i],
    ["news", /news|coverage|reported|reports/i],
    ["satellite", /satellit/i],
  ];
  const topicOf = (line: string) => TOPICS.find(([, re]) => re.test(line))?.[0] ?? "";

  const standing: string[] = [];
  if (a.score.missing.includes("m_height_above_water")) {
    standing.push(
      "Height above the nearest water: not applicable, no regular water within 1 km.",
    );
  }
  standing.push(
    "Satellite data can say whether an area is near rivers or the coast. It cannot tell a " +
      "street that floods from one that does not: we measured that, and it was no better " +
      "than guessing.",
  );
  standing.push(
    "Local news covers some parts of Lagos far better than others, so finding no reports is " +
      "weak evidence rather than a clean bill of health.",
  );
  for (const note of a.notes) if (note.startsWith("NO ")) standing.push(note);

  const covered = new Set(standing.map(topicOf).filter(Boolean));
  const required = (report?.caveats ?? []).filter((c) => c.includes(NO_STREET_REPORTS));
  const fromAgent = (report?.caveats ?? []).filter(
    (c) => !c.includes(NO_STREET_REPORTS) && !covered.has(topicOf(c)),
  );

  const seen = new Set<string>();
  const limits: string[] = [];
  for (const line of [...required, ...fromAgent, ...standing]) {
    const k = line.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter(Boolean)
      .slice(0, 6).join(" ");
    if (!k || seen.has(k)) continue;
    seen.add(k);
    limits.push(line);
  }

  return (
    <div className="flex flex-col gap-4">
      <MissingEvidenceBanner groups={score.missing_groups} />

      <Card aria-label="Verdict" className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-6">
        <div className="flex-1">
          <Label>Flood risk</Label>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <TierChip tier={score.tier} large />
            {!score.weighted_tier && (
              <div>
                <Mono className="text-2xl font-semibold">
                  {score.score.toFixed(1)}
                  <span className="text-base text-muted">/100</span>
                </Mono>
                <p className="text-xs text-muted">Confidence: {score.confidence}</p>
              </div>
            )}
          </div>
          {score.weighted_tier && score.tier_floor && (
            <>
              <p className="mt-2 text-sm">
                <Mono className="font-semibold">Weighted score {score.score.toFixed(1)}</Mono> (
                {score.weighted_tier}) · held at {score.tier} by the evidence rule
              </p>
              <p className="mt-2 rounded-[10px] bg-tint px-3 py-2 text-sm text-ink">
                <strong className="font-semibold">Why:</strong> {score.tier_floor}. Direct
                evidence that this street floods outranks the weighted measurements.
              </p>
              <p className="mt-1 text-xs text-muted">Confidence: {score.confidence}</p>
            </>
          )}
        </div>
        <div className="flex-1 border-t border-rule pt-4 sm:border-l sm:border-t-0 sm:pl-6 sm:pt-0">
          <Label>Recommended action</Label>
          <p className="mt-2 text-lg font-semibold">
            {report ? ACTION_WORDS[report.action] ?? report.action : "—"}
          </p>
          {report && <p className="mt-1.5 text-sm text-muted">{report.rationale}</p>}
          <div className="mt-3">{report && <VerifiedBadge explained={report.explained} />}</div>
        </div>
      </Card>

      {report && (
        <Card aria-label="Summary">
          <p className="text-[15px] leading-relaxed">{report.summary}</p>
        </Card>
      )}

      <Card aria-label="Risk drivers">
        <h2 className="text-base font-semibold">What drives this score</h2>
        <p className="mt-1 text-sm text-muted">
          Every number is measured in code. Muse Spark only explains what it means.
        </p>
        <div className="mt-4 flex flex-col gap-5">
          {view.groups.map((group) => (
            <div key={group.name}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="text-sm font-semibold first-letter:uppercase">{group.name}</h3>
                {!group.present && (
                  <span className="rounded-full bg-high-bg px-2.5 py-0.5 text-xs font-semibold text-high-ink">
                    missing
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-xs text-muted">{GROUP_BLURB[group.name] ?? ""}</p>
              <div className="mt-1.5">
                <Metrics group={group} />
              </div>
              {report?.drivers
                .filter((d) => group.metrics.some((m) => m.id === d.metric_id))
                .map((d) => (
                  <p key={d.metric_id} className="mt-2 rounded-[10px] bg-tint px-3 py-2 text-sm">
                    <span className="font-semibold">{byId[d.metric_id]?.label ?? d.metric_id}:</span>{" "}
                    {d.why}
                  </p>
                ))}
            </div>
          ))}
        </div>
      </Card>

      <Card aria-label="Reported flooding nearby">
        <h2 className="text-base font-semibold">Reported flooding nearby</h2>
        {view.reports.length === 0 || !a.has_street_reports ? (
          <p className="mt-2 rounded-[10px] bg-moderate-bg px-3 py-2.5 text-sm font-semibold text-moderate-ink">
            {NO_STREET_REPORTS}
          </p>
        ) : null}
        {view.report_groups.length > 0 && (
          <div className="mt-3 flex flex-col gap-3">
            {view.report_groups.map((g) => (
              <EventCard key={g.event_date} group={g} />
            ))}
          </div>
        )}
      </Card>

      {view.photos.length > 0 && (
        <Card aria-label="Site photos">
          <h2 className="text-base font-semibold">Site photos</h2>
          <p className="mt-1 text-sm text-muted">
            Shown as evidence only. Nothing here changes the score.
          </p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {view.photos.map((photo) => {
              const hits = photo.findings.filter((f) => f.instances > 0);
              const shown = openPhoto === photo.name ? openPhoto : null;
              const overlay = hits.find((f) => f.overlay_url);
              return (
                <figure key={photo.name}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={shown && overlay ? overlay.overlay_url : photo.url}
                    alt={`Site photo${shown && overlay ? `, ${overlay.prompt} highlighted` : ""}`}
                    className="w-full rounded-[12px] border border-line"
                  />
                  <figcaption className="mt-2 flex flex-wrap items-center gap-1.5">
                    {hits.length === 0 ? (
                      <span className="text-xs text-muted">Nothing detected</span>
                    ) : (
                      hits.map((f) => (
                        <span key={f.prompt} className="rounded-full bg-tint px-2.5 py-0.5 text-xs text-accent">
                          {f.prompt} · {(f.coverage * 100).toFixed(0)}%
                        </span>
                      ))
                    )}
                    {overlay && (
                      <button
                        type="button"
                        onClick={() => setOpenPhoto(shown ? null : photo.name)}
                        className="ml-auto rounded-[10px] px-2 text-xs font-semibold text-accent hover:bg-tint"
                      >
                        {shown ? "Hide overlay" : "Show overlay"}
                      </button>
                    )}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </Card>
      )}

      <Card aria-label="Evidence images">
        <h2 className="text-base font-semibold">Evidence</h2>
        {noWater ? (
          <p className="mt-2 text-sm text-muted">
            None of the three water layers — Sentinel-2 on a clear date, Sentinel-1 radar on
            the wettest recent date, and JRC water history since 1984 — found any water
            within 1 km of this site, so there is nothing to show on a map.
          </p>
        ) : (
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            {view.evidence_urls.water_overlay && (
              <figure className="sm:col-span-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={view.evidence_urls.water_overlay}
                  alt="Satellite view with detected water shaded blue"
                  className="w-full rounded-[12px] border border-line"
                />
                <figcaption className="mt-1.5 text-xs text-muted">
                  Water found by any layer, shaded blue on the satellite view{opticalCaption}
                </figcaption>
              </figure>
            )}
            {view.evidence_urls.optical && !view.evidence_urls.water_overlay && (
              <figure className="sm:col-span-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={view.evidence_urls.optical}
                  alt="Satellite view of the site"
                  className="w-full rounded-[12px] border border-line"
                />
                <figcaption className="mt-1.5 text-xs text-muted">
                  Satellite view{opticalCaption}
                </figcaption>
              </figure>
            )}
            {Object.entries(view.water_layers)
              .filter(([name, hasWater]) => hasWater && view.evidence_urls[name])
              .map(([name]) => (
                <figure key={name}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={view.evidence_urls[name]}
                    alt={EVIDENCE_CAPTION[name]}
                    className="w-full rounded-[12px] border border-line"
                  />
                  <figcaption className="mt-1.5 text-xs text-muted">
                    {EVIDENCE_CAPTION[name]} · white is water, black is none
                  </figcaption>
                </figure>
              ))}
          </div>
        )}
      </Card>

      <Card aria-label="Limits">
        <h2 className="text-base font-semibold">What this report cannot tell you</h2>
        <ul className="mt-2 flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted">
          {limits.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <p className="mt-3 text-xs text-muted">
          <Mono>{coords(a.lat, a.lon)}</Mono>
          {view.loan_amount ? ` · ${money(view.loan_amount)}` : ""}
          {view.tenor_months ? ` · ${view.tenor_months} months` : ""}
          {view.from_cache ? " · cached demo result" : ""}
          {` · $${view.cost_usd.toFixed(4)} to produce`}
        </p>
      </Card>
    </div>
  );
}
