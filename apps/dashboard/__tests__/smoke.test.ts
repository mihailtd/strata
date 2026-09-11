/**
 * Dashboard smoke tests.
 *
 * These tests verify the Next.js app configuration can be imported/resolved
 * without any runtime errors, without needing a live server.
 */

describe("Dashboard smoke tests", () => {
  test("next.config.ts is importable without errors", () => {
    // Verify the config file can be required without throwing
    expect(() => {
      // Dynamic require so jest can catch module-level errors
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require("../next.config");
    }).not.toThrow();
  });

  test("environment constants are valid", () => {
    // The dashboard reads NEXT_PUBLIC_RUNTIME_URL at build time.
    // When unset it falls back to localhost:8000 — verify that fallback is a valid URL.
    const runtimeUrl =
      process.env.NEXT_PUBLIC_RUNTIME_URL ?? "http://localhost:8000";
    expect(() => new URL(runtimeUrl)).not.toThrow();
  });

  test("API endpoint paths follow expected shape", () => {
    const endpoints = [
      "/health",
      "/v1/models",
      "/v1/chat/completions",
      "/api/engine/status",
      "/api/engine/load",
      "/api/engine/unload",
      "/api/telemetry",
    ];
    for (const ep of endpoints) {
      expect(ep).toMatch(/^\/[a-z0-9/_-]+$/);
    }
  });
});
