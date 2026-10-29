"use client";

import { STEP_ORDER, type StepName, type StepStatus, type StepView } from "@/lib/types";
import { Mono } from "@/components/ui";

const DESCRIPTION: Record<StepName, string> = {
  "Area screen": "Satellite water history, terrain and land cover around the site",
  "Flood history": "News reports of flooding at this street, checked against the source page",
  "Site photos": "SAM looks for standing water, doors, roads and drains",
  Score: "The fixed formula, in code",
  Explain: "Muse Spark writes the report",
  Verify: "Every number is checked against a measured value",
};

function Dot({ status }: { status: StepStatus }) {
  if (status === "done")
    return (
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-low-bg text-low-ink">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M4 12.5l5 5L20 6.5" />
        </svg>
      </span>
    );
  if (status === "failed")
    return (
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-high-bg text-high-ink">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </span>
    );
  if (status === "skipped")
    return <span className="flex h-6 w-6 items-center justify-center rounded-full border border-line text-muted">–</span>;
  if (status === "running")
    return (
      <span className="flex h-6 w-6 items-center justify-center" aria-label="running">
        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      </span>
    );
  return <span className="h-6 w-6 rounded-full border border-line" />;
}

/** The step list. Driven by whatever the stream has sent so far, keyed by step name:
 *  a replayed run arrives already complete, with no `running` event for any step, and a
 *  reconnecting phone receives the whole backlog again. */
export function Steps({ steps }: { steps: Record<string, StepView> }) {
  return (
    <ol className="flex flex-col">
      {STEP_ORDER.map((name, i) => {
        const step = steps[name];
        const status: StepStatus = step?.status ?? "pending";
        return (
          <li
            key={name}
            className={`flex gap-3 py-3 ${i ? "border-t border-rule" : ""} ${
              status === "pending" ? "opacity-55" : ""
            }`}
          >
            <Dot status={status} />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                <span className="font-semibold">{name}</span>
                {step && (status === "done" || status === "failed") && step.ms > 0 && (
                  <Mono className="text-xs text-muted">{(step.ms / 1000).toFixed(1)}s</Mono>
                )}
              </div>
              <p className="mt-0.5 text-sm text-muted">
                {step?.detail || DESCRIPTION[name]}
              </p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
