import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET /api/engine/status/stream needed its own Route Handler, not the generic
// `/api/engine/:path*` entry in next.config.ts's rewrites(). Same reason
// app/api/chat/stream/route.ts exists instead of relying on a rewrite for
// /v1/chat/completions: config-level rewrites here do not reliably stream a
// long-lived response through -- chat SSE already had to work around this, and
// engine-status SSE hit the identical wall (POST toggles returned 200 from the
// Python server, but the browser's EventSource never received a frame, so the UI
// only ever updated on a manual reload). Explicitly piping `response.body` through
// a Route Handler, with the streaming-specific headers set by hand, is what
// actually flushes chunks to the client instead of buffering the whole response.
export async function GET(req: NextRequest) {
  try {
    const upstream = await fetch("http://127.0.0.1:8000/api/engine/status/stream", {
      headers: { Accept: "text/event-stream" },
      signal: req.signal,
    });

    if (!upstream.ok || !upstream.body) {
      return new Response(JSON.stringify({ error: `Backend returned ${upstream.status}` }), {
        status: upstream.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return new Response(JSON.stringify({ error: message || "Streaming error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
