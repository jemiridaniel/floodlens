"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { createAssessment } from "@/lib/api";
import { SiteMap } from "@/components/SiteMap";
import { Button, Card, Field, Label, Mono, coords, inputClass } from "@/components/ui";

const LAGOS = { lat: 6.5026, lon: 3.3052 }; // Ago Palace Way, Okota
const MAX_PHOTOS = 5;

export function AssessmentForm() {
  const router = useRouter();
  const [pin, setPin] = useState(LAGOS);
  const [photos, setPhotos] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const form = useRef<HTMLFormElement>(null);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const data = new FormData(event.currentTarget);
    try {
      const id = await createAssessment({
        lat: pin.lat,
        lon: pin.lon,
        address: String(data.get("address") ?? ""),
        label: String(data.get("label") ?? ""),
        loan_amount: String(data.get("loan_amount") ?? ""),
        tenor_months: String(data.get("tenor_months") ?? ""),
        note: String(data.get("note") ?? ""),
        photos,
      });
      router.push(`/assessments/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start the assessment");
      setBusy(false);
    }
  }

  return (
    <form ref={form} onSubmit={submit} className="grid gap-4 lg:grid-cols-[1fr_380px]">
      <div className="order-2 flex flex-col gap-4 lg:order-1">
        <Card>
          <h2 className="mb-1 text-base font-semibold">Where is the collateral?</h2>
          <p className="mb-4 text-sm text-muted">
            Tap the map to place the pin. The rings show 250 m and 500 m, the distances the
            area screen measures.
          </p>
          <SiteMap lat={pin.lat} lon={pin.lon} onPick={(lat, lon) => setPin({ lat, lon })} />
          <p className="mt-3 text-sm text-muted">
            Pin at <Mono className="text-ink">{coords(pin.lat, pin.lon)}</Mono>
          </p>
        </Card>
      </div>

      <div className="order-1 flex flex-col gap-4 lg:order-2">
        <Card className="flex flex-col gap-4">
          <Field
            label="Address"
            hint="Used to search local news as well as the map. The street name helps most."
          >
            <input name="address" className={inputClass} placeholder="24 Ago Palace Way, Okota" />
          </Field>
          <Field label="Site name">
            <input name="label" className={inputClass} placeholder="Okota shop" />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Loan amount">
              <input
                name="loan_amount"
                type="number"
                min="0"
                step="1000"
                inputMode="numeric"
                className={`${inputClass} font-mono`}
                placeholder="1500000"
              />
            </Field>
            <Field label="Tenor (months)">
              <input
                name="tenor_months"
                type="number"
                min="1"
                max="360"
                inputMode="numeric"
                className={`${inputClass} font-mono`}
                placeholder="12"
              />
            </Field>
          </div>
          <Field label="Site note">
            <textarea
              name="note"
              rows={3}
              className={inputClass}
              placeholder="Ground floor shop, concrete slab, drain at the back"
            />
          </Field>
        </Card>

        <Card>
          <Label>Site photos</Label>
          <p className="mb-3 mt-1.5 text-sm text-muted">
            Up to {MAX_PHOTOS}. Location data is removed from every photo before it is stored.
            Photos are shown as evidence and do not change the score.
          </p>
          <input
            type="file"
            accept="image/*"
            multiple
            className="w-full text-sm file:mr-3 file:rounded-[10px] file:border-0 file:bg-tint file:px-4 file:py-2.5 file:text-sm file:font-semibold file:text-accent"
            onChange={(e) => setPhotos(Array.from(e.target.files ?? []).slice(0, MAX_PHOTOS))}
          />
          {photos.length > 0 && (
            <ul className="mt-3 flex flex-col gap-1 text-sm text-muted">
              {photos.map((p) => (
                <li key={p.name} className="truncate">
                  {p.name} <Mono>({Math.round(p.size / 1024)} KB)</Mono>
                </li>
              ))}
            </ul>
          )}
        </Card>

        {error && (
          <p role="alert" className="rounded-[12px] bg-high-bg px-4 py-3 text-sm text-high-ink">
            {error}
          </p>
        )}

        <Button type="submit" disabled={busy} className="w-full">
          {busy ? "Starting…" : "Run assessment"}
        </Button>
        <p className="text-center text-xs text-muted">
          Takes about a minute. You can watch each step as it runs.
        </p>
      </div>
    </form>
  );
}
