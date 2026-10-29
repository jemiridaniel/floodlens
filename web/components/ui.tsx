import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import type { Tier } from "@/lib/types";

const TIER_STYLE: Record<Tier, string> = {
  Low: "bg-low-bg text-low-ink",
  Moderate: "bg-moderate-bg text-moderate-ink",
  High: "bg-high-bg text-high-ink",
  Severe: "bg-severe-bg text-severe-ink",
};

/** Tier labels come from the API. Never hard-code one (design/README.md). */
export function TierChip({ tier, large = false }: { tier: Tier; large?: boolean }) {
  return (
    <span
      className={`inline-flex items-center rounded-full font-semibold ${TIER_STYLE[tier]} ${
        large ? "px-4 py-1.5 text-2xl" : "px-3 py-1 text-sm"
      }`}
    >
      {tier}
    </span>
  );
}

export function Card({
  children,
  className = "",
  as: Tag = "section",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article";
} & HTMLAttributes<HTMLElement>) {
  return (
    <Tag
      className={`rounded-[12px] border border-line bg-card p-4 sm:p-5 ${className}`}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-xs font-semibold uppercase tracking-wide text-muted">{children}</span>
  );
}

export function Mono({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`font-mono tabular-nums ${className}`}>{children}</span>;
}

export function Button({
  children,
  variant = "primary",
  className = "",
  ...rest
}: {
  children: ReactNode;
  variant?: "primary" | "secondary";
  className?: string;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-[10px] px-4 py-2.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50";
  const look =
    variant === "primary"
      ? "bg-accent text-white hover:bg-accent-hover"
      : "border border-line bg-card text-ink hover:bg-tint";
  return (
    <button className={`${base} ${look} ${className}`} {...rest}>
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-semibold text-ink">{label}</span>
      {children}
      {hint ? <span className="mt-1 block text-xs text-muted">{hint}</span> : null}
    </label>
  );
}

export const inputClass =
  "w-full rounded-[10px] border border-line bg-card px-3 py-2.5 text-sm text-ink outline-none placeholder:text-muted/70 focus:border-accent focus:ring-2 focus:ring-tint";

/** Shown when a whole kind of evidence is missing. Driven by score.missing_groups, not
 *  `missing`: one absent metric is routine, an absent group means the score is resting on
 *  evidence it was never weighted to carry. */
export function MissingEvidenceBanner({ groups }: { groups: string[] }) {
  if (!groups.length) return null;
  return (
    <div
      role="status"
      className="rounded-[12px] border border-high-ink/30 bg-high-bg px-4 py-3 text-sm text-high-ink"
    >
      <p className="font-semibold">
        {groups.length === 1 ? "One kind of evidence is missing" : "Some evidence is missing"}:{" "}
        {groups.join(", ")}.
      </p>
      <p className="mt-1">
        The score rests on the rest, so confidence is held at low. Treat this report as
        incomplete rather than reassuring.
      </p>
    </div>
  );
}

export function VerifiedBadge({ explained }: { explained: boolean }) {
  return explained ? (
    <span className="inline-flex items-center gap-2 rounded-full bg-low-bg px-3 py-1.5 text-sm font-semibold text-low-ink">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" />
        <path d="M8.5 12l2.5 2.5 4.5-5" />
      </svg>
      Verified: every number matches a measured value
    </span>
  ) : (
    <span className="inline-flex items-center gap-2 rounded-full bg-moderate-bg px-3 py-1.5 text-sm font-semibold text-moderate-ink">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
        <path d="M12 8v5M12 16.5v.5M10.3 3.9 2.6 17.3A1.6 1.6 0 0 0 4 19.7h16a1.6 1.6 0 0 0 1.4-2.4L13.7 3.9a1.6 1.6 0 0 0-2.8 0z" />
      </svg>
      Written from the measurements only
    </span>
  );
}

export function money(amount: number | null): string {
  if (amount == null) return "—";
  return new Intl.NumberFormat("en-NG", {
    style: "currency",
    currency: "NGN",
    maximumFractionDigits: 0,
  }).format(amount);
}

export function coords(lat: number, lon: number): string {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(4)}° ${
    lon >= 0 ? "E" : "W"
  }`;
}
