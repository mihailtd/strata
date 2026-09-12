"use client";

import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart, BarChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  DatasetComponent,
  TransformComponent,
  MarkLineComponent,
  MarkAreaComponent,
  MarkPointComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

// Register modular assets globally once inside this runtime boundary
echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  DatasetComponent,
  TransformComponent,
  MarkLineComponent,
  MarkAreaComponent,
  MarkPointComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

interface AdaptiveChartProps {
  options: echarts.EChartsCoreOption;
  theme?: "light" | "dark";
  className?: string;
}

export default function AdaptiveChart({
  options,
  theme = "dark",
  className = "w-full h-full min-h-[350px]",
}: AdaptiveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);

  // Effect 1: Handle Initializing and Options updates
  useEffect(() => {
    if (!containerRef.current) return;

    if (!chartInstanceRef.current) {
      chartInstanceRef.current = echarts.init(containerRef.current, theme);
    } else {
      // Re-initialize if the theme changes dynamically
      chartInstanceRef.current.dispose();
      chartInstanceRef.current = echarts.init(containerRef.current, theme);
    }

    chartInstanceRef.current.setOption(options, { notMerge: true });
  }, [options, theme]);

  // Effect 2: Isolate window resizing event streams via ResizeObserver
  useEffect(() => {
    const handleResize = () => {
      chartInstanceRef.current?.resize();
    };

    const resizeObserver = new ResizeObserver(() => handleResize());
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
      chartInstanceRef.current?.dispose();
      chartInstanceRef.current = null;
    };
  }, []);

  return <div ref={containerRef} className={`${className} transition-colors duration-200`} />;
}
