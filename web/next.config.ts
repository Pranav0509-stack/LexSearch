import type { NextConfig } from "next";

// Proxy all /api/* calls to the FastAPI backend running on :8080 so that
// the session cookie (`ls_session`) stays first-party and the browser sees
// a single origin. Set BACKEND_ORIGIN at deploy time to point elsewhere.
const BACKEND = process.env.BACKEND_ORIGIN || "http://localhost:8080";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      // Static PDF passthrough so source cards can deep-link to a judgment.
      { source: "/pdf/:path*", destination: `${BACKEND}/pdf/:path*` },
      { source: "/sc-pdf/:path*", destination: `${BACKEND}/sc-pdf/:path*` },
    ];
  },
};

export default nextConfig;
