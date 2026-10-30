import type { Metadata } from "next";
import type { ReactNode } from "react";
import localFont from "next/font/local";
import { Header } from "@/components/Header";
import "./globals.css";

// Self-hosted rather than next/font/google. The Google loader fetches font CSS during
// the build, and that build step crashed on Vercel while succeeding locally. A deploy
// should not depend on a third-party request at build time for something this static.
const sans = localFont({
  variable: "--font-plex-sans",
  display: "swap",
  src: [
    { path: "./fonts/plex-sans-400.woff2", weight: "400", style: "normal" },
    { path: "./fonts/plex-sans-500.woff2", weight: "500", style: "normal" },
    { path: "./fonts/plex-sans-600.woff2", weight: "600", style: "normal" },
    { path: "./fonts/plex-sans-700.woff2", weight: "700", style: "normal" },
  ],
});
const mono = localFont({
  variable: "--font-plex-mono",
  display: "swap",
  src: [
    { path: "./fonts/plex-mono-400.woff2", weight: "400", style: "normal" },
    { path: "./fonts/plex-mono-500.woff2", weight: "500", style: "normal" },
    { path: "./fonts/plex-mono-600.woff2", weight: "600", style: "normal" },
  ],
});

export const metadata: Metadata = {
  title: "FloodLens",
  description: "Flood-risk checks on loan collateral, from open data and local reports.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="font-sans antialiased">
        <Header />
        <main className="mx-auto w-full max-w-6xl px-4 py-5 sm:py-7">{children}</main>
      </body>
    </html>
  );
}
