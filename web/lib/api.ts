import type { AssessmentView, StepView } from "./types";

export const API =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export interface NewAssessment {
  lat: number;
  lon: number;
  address?: string;
  label?: string;
  loan_amount?: string;
  tenor_months?: string;
  note?: string;
  photos: File[];
}

export async function createAssessment(input: NewAssessment): Promise<string> {
  const form = new FormData();
  form.append("lat", String(input.lat));
  form.append("lon", String(input.lon));
  for (const key of ["address", "label", "loan_amount", "tenor_months", "note"] as const) {
    const value = input[key];
    if (value) form.append(key, String(value));
  }
  for (const photo of input.photos) form.append("photos", photo, photo.name);

  const r = await fetch(`${API}/assessments`, { method: "POST", body: form });
  if (!r.ok) throw new Error(await readError(r));
  return (await r.json()).id as string;
}

export async function getAssessment(id: string): Promise<AssessmentView> {
  const r = await fetch(`${API}/assessments/${id}`, { cache: "no-store" });
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

async function readError(r: Response): Promise<string> {
  try {
    const body = await r.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return detail.map((d) => d.msg).join("; ");
  } catch {
    /* fall through to the status line */
  }
  return `${r.status} ${r.statusText}`;
}

export type StreamEvent =
  | ({ type: "step"; seq: number } & StepView)
  | { type: "done"; seq: number; status: string; error: string | null; cost_usd: number };

/** Subscribe to a job's progress. Returns an unsubscribe function.
 *
 *  Two things this must tolerate, both real:
 *  - A replayed (demo-cached) run arrives already complete: every step lands as `done`
 *    with no `running` first, and the `done` event may be the second message we see.
 *  - A reconnecting phone gets the whole backlog again, so events repeat. Callers key by
 *    step name, not by arrival order. */
export function watchAssessment(
  id: string,
  onEvent: (event: StreamEvent) => void,
  onClose?: (status: string, error: string | null) => void,
): () => void {
  const source = new EventSource(`${API}/assessments/${id}/events`);
  let closed = false;

  const close = () => {
    if (!closed) {
      closed = true;
      source.close();
    }
  };

  const handle = (raw: MessageEvent) => {
    let event: StreamEvent;
    try {
      event = JSON.parse(raw.data);
    } catch {
      return;
    }
    onEvent(event);
    if (event.type === "done") {
      onClose?.(event.status, event.error);
      close();
    }
  };

  source.addEventListener("step", handle as EventListener);
  source.addEventListener("done", handle as EventListener);
  source.onerror = () => {
    // The stream ends by closing, which EventSource reports as an error. Only treat it
    // as a failure if the job never told us it finished.
    if (!closed) {
      closed = true;
      source.close();
      onClose?.("disconnected", null);
    }
  };
  return close;
}

export function evidenceUrl(id: string, name: string): string {
  return `${API}/assessments/${id}/evidence/${name}.png`;
}
