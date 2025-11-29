import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker
  output: 'standalone',

  // Disable telemetry
  reactStrictMode: true,

  // Fix Turbopack path resolution for Vercel monorepo
  experimental: {
    turbo: {
      resolveAlias: {
        '@': './src',
      },
    },
  },
};

export default nextConfig;
