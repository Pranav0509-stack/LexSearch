import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";

// Display: Fraunces — variable serif with optical-size axis. More modern
// than Playfair, less stiff than Tiempos. The wordmark and section
// headings render with `opsz=144` so the high-display cut shows through.
//
// Note: `axes` requires the variable-weight axis (i.e. `weight` omitted).
// next/font errors otherwise: "Axes can only be defined for variable fonts
// when the weight property is nonexistent or set to `variable`."
const fraunces = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
  style: ["normal", "italic"],
  axes: ["opsz", "SOFT", "WONK"],
  display: "swap",
});

// Body: Inter — the workhorse. Slightly wider than Inter Tight, easier on
// long-form legal prose. We keep the variable name `--font-inter-tight`
// so existing Tailwind classes (`font-sans`, `font-body`) don't break.
const inter = Inter({
  variable: "--font-inter-tight",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sanhita — research counsel for Asian jurisdictions",
  description:
    "Grounded legal answers across India, Japan, Singapore, and 14 more Asian markets. Every claim cited.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      // Backwards-compat: keep the old `--font-playfair` alias pointing at
      // Fraunces so any stray `var(--font-playfair)` reference still
      // renders. New code should reach for `--font-display`.
      className={`${fraunces.variable} ${inter.variable} h-full antialiased`}
      style={{ ["--font-playfair" as string]: "var(--font-display)" }}
    >
      <body className="min-h-full flex flex-col bg-[var(--bg)] text-[var(--ink)]">
        {children}
      </body>
    </html>
  );
}
