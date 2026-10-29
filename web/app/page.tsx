import { Suspense } from "react";
import { AssessmentForm } from "@/components/AssessmentForm";

export default function NewAssessmentPage() {
  return (
    <>
      <div className="mb-5">
        <h1 className="text-xl font-semibold sm:text-2xl">New assessment</h1>
        <p className="mt-1 text-sm text-muted">
          Check one piece of collateral against satellite data and local flood reports.
        </p>
      </div>
      {/* useSearchParams needs a boundary when the page is statically rendered. */}
      <Suspense fallback={<p className="text-sm text-muted">Loading the map…</p>}>
        <AssessmentForm />
      </Suspense>
    </>
  );
}
