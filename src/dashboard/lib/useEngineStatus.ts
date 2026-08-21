"use client";

import { useEffect, useRef, useState } from "react";
import { EngineStatus } from "@/lib/types";

/**
 * One live subscription to GET /api/engine/status/stream (SSE), shared by every
 * component that needs engine status.
 *
 * Replaces two independent poll loops that used to exist -- Header.tsx on a 3s
 * setInterval, chat/page.tsx on refetch-after-action -- which had no way to agree
 * with each other: a toggle made from one view could take up to 3s to show up in
 * the other, or the two could show genuinely different snapshots mid-flight. One
 * EventSource per mounted component here is still two connections, not one shared
 * one, but both now read the SAME backend event stream on the SAME cadence
 * (push-on-change, not a fixed poll interval), so they can't disagree the way two
 * independently-timed polls could.
 */
export function useEngineStatus() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Belt-and-suspenders: fire a one-shot GET immediately alongside opening the
    // SSE connection. If the stream is slow to establish (or, in a setup this new,
    // fails outright -- e.g. the Next.js rewrite mishandling a chunked response is
    // a real possibility, unverified) this still gets a first status instead of
    // leaving every status-gated control stuck on a null read indefinitely.
    fetch("/api/engine/status")
      .then((r) => r.json())
      .then((data) => setStatus((prev) => prev ?? data))
      .catch(() => {});

    const es = new EventSource("/api/engine/status/stream");
    esRef.current = es;

    es.addEventListener("status", (ev: MessageEvent) => {
      try {
        setStatus(JSON.parse(ev.data));
        setConnected(true);
      } catch {
        // malformed frame -- skip it, keep the last good status rather than crash
      }
    });

    es.onerror = () => {
      // EventSource auto-reconnects on its own; this just reflects that the last
      // attempt dropped, so the UI can show "reconnecting" instead of stale-but-live.
      setConnected(false);
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  const patchStatus = (partial: Partial<EngineStatus>) => {
    setStatus((prev) => (prev ? { ...prev, ...partial } : (partial as EngineStatus)));
  };

  const refetch = async () => {
    try {
      const r = await fetch("/api/engine/status");
      const data = await r.json();
      setStatus(data);
      return data as EngineStatus;
    } catch {
      return null;
    }
  };

  return { status, connected, patchStatus, refetch };
}
