import Link from "next/link";

export function Header() {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-card/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 text-ink">
          <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
            <circle cx="12" cy="12" r="9" fill="none" stroke="#1D5FA8" strokeWidth="2.2" />
            <path d="M6 13c2-2 4-2 6 0s4 2 6 0" fill="none" stroke="#1D5FA8" strokeWidth="2.2" strokeLinecap="round" />
            <path d="M19 19l6 6" stroke="#0F1E2D" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
          <span className="text-base font-semibold">FloodLens</span>
        </Link>
        <nav aria-label="Main" className="flex items-center gap-1 text-sm">
          <Link href="/" className="rounded-[10px] px-3 py-2 font-medium text-accent hover:bg-tint">
            New assessment
          </Link>
          <Link href="/portfolio" className="rounded-[10px] px-3 py-2 font-medium text-accent hover:bg-tint">
            Portfolio
          </Link>
        </nav>
      </div>
    </header>
  );
}
