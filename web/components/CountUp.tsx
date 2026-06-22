"use client";

import { useEffect, useRef, useState } from "react";

const reduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

// 从 0 平滑滚动到 target（easeOutCubic）。尊重 prefers-reduced-motion。
// 用于结果出现时的关键读数（修正幅度、边界评分 B），与 reveal/bar 入场节奏呼应。
export function useCountUp(target: number, durationMs = 650): number {
  const [val, setVal] = useState(0);
  const fromRef = useRef(0);
  const rafRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (reduced() || durationMs <= 0) {
      setVal(target);
      fromRef.current = target;
      return;
    }
    const from = fromRef.current;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setVal(from + (target - from) * eased);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, durationMs]);

  return val;
}

export function CountUp({
  value,
  decimals = 3,
  suffix = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
}) {
  const v = useCountUp(value);
  return (
    <>
      {v.toFixed(decimals)}
      {suffix}
    </>
  );
}
