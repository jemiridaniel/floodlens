// Mirrors backend/app/jobs.py (AssessmentView) and app/schemas.py. Kept narrow on
// purpose: anything the UI does not read is not declared, so a backend addition cannot
// silently become a frontend assumption.

export type Tier = "Low" | "Moderate" | "High" | "Severe";
export type StepStatus = "pending" | "running" | "done" | "failed" | "skipped";
export type MatchLevel = "street" | "estate" | "area";

export const STEP_ORDER = [
  "Area screen",
  "Flood history",
  "Site photos",
  "Score",
  "Explain",
  "Verify",
] as const;
export type StepName = (typeof STEP_ORDER)[number];

export interface StepView {
  seq: number;
  step: StepName;
  status: StepStatus;
  ms: number;
  detail: string;
  cost_usd: number;
}

export interface Metric {
  id: string;
  label: string;
  value: number;
  unit: string;
  source: string;
  decimals: number;
  /** Set when the bare number would mislead: "more than 1 km" for a distance that
   *  saturated, "1.0 m lower than the surrounding ground" for a signed difference. */
  display_value: string | null;
}

/** What a reader should see for a metric. */
export function metricText(m: Metric): string {
  if (m.display_value) return m.display_value;
  return `${m.value.toFixed(m.decimals)} ${m.unit}`.trim();
}

export interface Score {
  score: number;
  tier: Tier;
  confidence: string;
  contributions: Record<string, number>;
  missing: string[];
  /** Whole kinds of evidence that are absent. The banner keys off THIS, not `missing`:
   *  a single absent metric is normal, a whole absent group is not. */
  missing_groups: string[];
  /** Set when the street-level evidence rule held the tier above the weighted score. */
  tier_floor: string | null;
  /** The tier the weighted score alone would give. Only set when the rule raised it. */
  weighted_tier: Tier | null;
}

export interface Assessment {
  id: string;
  lat: number;
  lon: number;
  label: string | null;
  metrics: Metric[];
  score: Score;
  allowed_actions: string[];
  evidence: Record<string, string>;
  notes: string[];
  has_street_reports: boolean;
  street_flood_events: number;
}

export interface FloodReport {
  place_named: string;
  match_level: MatchLevel;
  event_date: string;
  publisher: string;
  url: string;
  quote: string;
  text_verified: "true" | "false" | "unknown";
  quote_found: boolean | null;
  phrase_found: boolean | null;
}

export interface Driver {
  metric_id: string;
  why: string;
}

export interface Report {
  summary: string;
  drivers: Driver[];
  action: string;
  rationale: string;
  caveats: string[];
  /** false = the model never produced a report the verifier accepted, and this is the
   *  metrics-only fallback. The badge must reflect it. */
  explained: boolean;
}

export interface EventGroup {
  event_date: string;
  reports: FloodReport[];
}

export interface GroupView {
  name: string;
  present: boolean;
  metrics: Metric[];
  contributions: Record<string, number>;
}

export interface FindingView {
  prompt: string;
  coverage: number;
  instances: number;
  status: string;
  overlay_url: string;
}

export interface PhotoView {
  name: string;
  url: string;
  findings: FindingView[];
  error: string;
}

export interface AgentRun {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  errors: string[];
}

export interface AssessmentView {
  id: string;
  status: "running" | "done" | "failed";
  lat: number;
  lon: number;
  address: string | null;
  label: string | null;
  loan_amount: number | null;
  tenor_months: number | null;
  note: string | null;
  from_cache: boolean;
  assessment: Assessment | null;
  groups: GroupView[];
  reports: FloodReport[];
  report_groups: EventGroup[];
  photos: PhotoView[];
  optical_date: string | null;
  optical_cloud_pct: number | null;
  /** layer name -> whether it found any water within 1 km. All false means the three
   *  mask images are blank squares, and one sentence says more than three of those. */
  water_layers: Record<string, boolean>;
  report: Report | null;
  run: AgentRun | null;
  evidence_urls: Record<string, string>;
  steps: StepView[];
  cost_usd: number;
  error: string | null;
}

/** Required verbatim wherever no street-level report was found. The backend's verifier
 *  enforces it in the agent's text; the UI shows it in its own right, because a replayed
 *  or fallback report must say it too. */
export const NO_STREET_REPORTS =
  "No street-level flood reports found. This does not mean the site is safe.";

export const GROUP_BLURB: Record<string, string> = {
  "local history": "News reports of flooding at this street or area, each checked against the page it came from.",
  "coastal and riverine": "How close the site is to water that is there most years, from 37 years of satellite history.",
  "terrain and surface": "Whether rain has somewhere to collect here, and whether it can soak away.",
  rainfall: "The heaviest day of rain on record nearby. A ~10 km grid, so it separates years, not addresses.",
};
