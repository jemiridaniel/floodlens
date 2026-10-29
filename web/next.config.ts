import type { NextConfig } from "next";

// The only configuration the frontend has is where the API lives. No keys ever reach the
// browser: the Meta and Planetary Computer credentials stay in the backend.
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
