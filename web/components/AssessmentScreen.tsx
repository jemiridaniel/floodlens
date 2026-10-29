"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { getAssessment, watchAssessment } from "@/lib/api";
import { STEP_ORDER, type AssessmentView, type StepView } from "@/lib/types";
import { RiskReport } from "@/components/RiskReport";
import { Steps } from "@/components/Steps";
import { Button, Card, Label, Mono, coords } from "@/components/ui";

const EVIDENCE_CAPTION: Record<string, string> = {
  optical: "Satellite view",
  water_now: "Water on that date",
  water_history: "Water since 1984",
  water_wet: "Radar, wettest date",
};

export function AssessmentScreen({ id }: { id: string }) {
  const [steps, setSteps] = useState<Record<string, StepView>>({});
  const [cost, setCost] = useState(0);
  const [view, setView] = useState<AssessmentView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const started = useRef(Date.now());

  const load = useCallback(async () => {
    try {
      const got = await getAssessment(id);
      setView(got);
      // Seed from the stored steps too: a replayed run may have finished before we
      // subscribed, and the stream will simply confirm what is already here.
      setSteps((prev) => {
        const next = { ...prev };
        for (const s of got.steps) if (s.status !== "pending") next[s.step] = s;
        return next;
      });
      setCost((c) => Math.max(c, got.cost_usd));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load this assessment");
    }
  }, [id]);

  useEffect(() => {
    let live = true;
    const stop = watchAssessment(
      id,
      (event) => {
        if (!live) return;
        if (event.type === "step") {
          // Keyed by step name, never by arrival order: a reconnecting phone replays the
          // whole backlog, and a cached run sends every step already `done`.
          setSteps((prev) => ({ ...prev, [event.step]: event }));
          setCost((c) => Math.max(c, event.cost_usd));
        } else {
          setCost((c) => Math.max(c, event.cost_usd));
        }
      },
      (status, streamError) => {
        if (!live) return;
        if (streamError) setError(streamError);
        void load();
      },
    );
    void load(); // also covers the case where the job finished before this page opened
    return () => {
      live = false;
      stop();
    };
  }, [id, load]);

  const finished = view?.status === "done" || view?.status === "failed";

  useEffect(() => {
    if (finished) return;
    const t = setInterval(() => setElapsed(Math.round((Date.now() - started.current) / 1000)), 500);
    return () => clearInterval(t);
  }, [finished]);

  if (view?.status === "done" && view.assessment) {
    return (
      <>
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <Link href="/" className="text-sm font-medium text-accent">
              ← New assessment
            </Link>
            <h1 className="mt-1 text-xl font-semibold sm:text-2xl">
              {view.label || view.address || "Risk report"}
            </h1>
            <p className="mt-1 text-xs text-muted">
              <Mono>{coords(view.lat, view.lon)}</Mono> · Report <Mono>{view.id}</Mono>
            </p>
          </div>
        </div>
        <RiskReport view={view} />
      </>
    );
  }

  const failed = view?.status === "failed" || error;

  return (
    <>
      <div className="mb-5">
        <Link href="/" className="text-sm font-medium text-accent">
          ← New assessment
        </Link>
        <h1 className="mt-1 text-xl font-semibold sm:text-2xl">
          {failed ? "Assessment failed" : "Working on it"}
        </h1>
        <p className="mt-1 text-sm text-muted">
          {failed
            ? view?.error ?? error
            : "The area screen and the local flood history run at the same time."}
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        <Card>
          <Steps steps={steps} />
        </Card>

        <div className="flex flex-col gap-4">
          <Card className="flex items-center justify-between gap-4">
            <div>
              <Label>Elapsed</Label>
              <p>
                <Mono className="text-lg font-semibold">
                  {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}
                </Mono>
              </p>
            </div>
            <div className="text-right">
              <Label>Cost so far</Label>
              <p>
                <Mono className="text-lg font-semibold">${cost.toFixed(4)}</Mono>
              </p>
            </div>
          </Card>

          {view && Object.keys(view.evidence_urls).length > 0 && (
            <Card>
              <Label>Evidence so far</Label>
              <div className="mt-3 grid grid-cols-2 gap-3">
                {Object.entries(view.evidence_urls)
                  .filter(([name]) => name in EVIDENCE_CAPTION)
                  .map(([name, url]) => (
                    <figure key={name}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={url} alt={EVIDENCE_CAPTION[name]} className="w-full rounded-[10px] border border-line" />
                      <figcaption className="mt-1 text-[11px] text-muted">
                        {EVIDENCE_CAPTION[name]}
                      </figcaption>
                    </figure>
                  ))}
              </div>
            </Card>
          )}

          {failed && (
            <Button variant="secondary" onClick={() => void load()}>
              Try loading again
            </Button>
          )}
        </div>
      </div>

      {!failed && Object.keys(steps).length === 0 && (
        <p className="mt-4 text-sm text-muted">Starting {STEP_ORDER.length} steps…</p>
      )}
    </>
  );
}
