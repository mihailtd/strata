"use client";

import { Dispatch, SetStateAction, useEffect, useState } from "react";

/**
 * useState that survives a page refresh via localStorage.
 *
 * Initializes with `defaultValue` on every render (server and first client render
 * match, so no hydration mismatch), then a client-only effect overwrites it from
 * localStorage if a persisted value exists. This means the UI briefly shows the
 * default before snapping to the persisted value on mount -- for settings like
 * routing mode / thinking effort / max tokens, that's a single frame, not
 * something worth trading hydration-safety away for.
 */
export function usePersistentState<T>(
  key: string,
  defaultValue: T
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(defaultValue);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(key);
      if (stored !== null) setValue(JSON.parse(stored));
    } catch {
      // corrupt or inaccessible storage -- keep the default, don't crash the page
    }
     
  }, [key]);

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage full / disabled (private browsing etc.) -- setting still works
      // for this session, it just won't survive a refresh
    }
  }, [key, value]);

  return [value, setValue];
}
