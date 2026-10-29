"use client";

import { useState } from "react";
import {
  GROUP_BLURB,
  NO_STREET_REPORTS,
  type AssessmentView,
  type FloodReport,
  type GroupView,
  type MatchLevel,
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

function Metrics({ group }: { group: GroupView }) {
  if (!group.present)
    return <p className="text-sm text-muted">Not measured for this site.</p>;
  return (
    <ul className="flex flex-col">
      {group.metrics.map((m, i) => (
        <li key={m.id} className={`flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-2.5 ${i ? "border-t border-rule" : ""}`}>
          <span className="min-w-0 flex-1 text-sm">{m.label}</span>
          <Mono className="text-sm font-semibold">
            {m.value.toFixed(m.decimals)} {m.unit}
          </Mono>
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

function ReportCard({ report }: { report: FloodReport }) {
  return (
    <article className="rounded-[12px] border border-line p-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${MATCH_STYLE[report.match_level]}`}>
          {MATCH_WORDS[report.match_level]}
        </span>
        <Mono className="text-xs text-muted">{report.event_date}</Mono>
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

  return (
    <div className="flex flex-col gap-4">
      <MissingEvidenceBanner groups={score.missing_groups} />

      <Card aria-label="Verdict" className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-6">
        <div className="flex-1">
          <Label>Flood risk</Label>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <TierChip tier={score.tier} large />
            <div>
              <Mono className="text-2xl font-semibold">
                {score.score.toFixed(1)}
                <span className="text-base text-muted">/100</span>
              </Mono>
              <p className="text-xs text-muted">Confidence: {score.confidence}</p>
            </div>
          </div>
          {score.tier_floor && (
            <p className="mt-3 rounded-[10px] bg-tint px-3 py-2 text-sm text-ink">
              <strong className="font-semibold">Evidence rule applied.</strong> {score.tier_floor}.
            </p>
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
                <h3 className="text-sm font-semibold capitalize">{group.name}</h3>
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
        {view.reports.length > 0 && (
          <div className="mt-3 flex flex-col gap-3">
            {view.reports.map((r) => (
              <ReportCard key={`${r.url}-${r.event_date}`} report={r} />
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
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          {Object.entries(view.evidence_urls)
            .filter(([name]) => name in EVIDENCE_CAPTION)
            .map(([name, url]) => (
              <figure key={name}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={url} alt={EVIDENCE_CAPTION[name]} className="w-full rounded-[12px] border border-line" />
                <figcaption className="mt-1.5 text-xs text-muted">{EVIDENCE_CAPTION[name]}</figcaption>
              </figure>
            ))}
        </div>
      </Card>

      <Card aria-label="Limits">
        <h2 className="text-base font-semibold">What this report cannot tell you</h2>
        <ul className="mt-2 flex list-disc flex-col gap-1.5 pl-5 text-sm text-muted">
          {report?.caveats.map((c) => <li key={c}>{c}</li>)}
          <li>
            Satellite data screens the <em>area</em> for coastal and river flooding. It cannot
            tell a flooding street from a dry one: that was measured, and it performed no
            better than chance.
          </li>
          <li>
            News coverage is uneven across Lagos, so finding no reports is weak evidence, not
            a clean bill of health.
          </li>
          {a.notes.filter((n) => n.startsWith("NO ")).map((n) => <li key={n}>{n}</li>)}
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
