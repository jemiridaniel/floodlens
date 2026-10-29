import { Card } from "@/components/ui";

export default function PortfolioPage() {
  return (
    <>
      <h1 className="text-xl font-semibold sm:text-2xl">Portfolio</h1>
      <p className="mt-1 text-sm text-muted">Screen a whole loan book at once.</p>
      <Card className="mt-5">
        <p className="text-sm text-muted">
          Not built yet. This screen will take a CSV of loans, score each site, and show
          exposure by tier with a map of the book. It needs the batch screener, which comes
          after the single-site flow is solid.
        </p>
      </Card>
    </>
  );
}
