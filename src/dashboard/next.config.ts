import type { NextConfig } from "next";
import createMDX from "@next/mdx";

const nextConfig: NextConfig = {
  pageExtensions: ["js", "jsx", "md", "mdx", "ts", "tsx"],
  async rewrites() {
    return [
      {
        source: "/api/engine/:path*",
        destination: "http://127.0.0.1:8000/api/engine/:path*",
      },
      {
        source: "/api/training/:path*",
        destination: "http://127.0.0.1:8000/api/training/:path*",
      },
      {
        source: "/api/causal_dag/:path*",
        destination: "http://127.0.0.1:8000/api/causal_dag/:path*",
      },
      {
        source: "/api/causal_dag",
        destination: "http://127.0.0.1:8000/api/causal_dag",
      },
      {
        source: "/api/factory/:path*",
        destination: "http://127.0.0.1:8000/api/factory/:path*",
      },
      {
        source: "/v1/:path*",
        destination: "http://127.0.0.1:8000/v1/:path*",
      },
      {
        source: "/events",
        destination: "http://127.0.0.1:8000/events",
      },
    ];
  },
};

const withMDX = createMDX({
  extension: /\.(md|mdx)$/,
});

export default withMDX(nextConfig);

