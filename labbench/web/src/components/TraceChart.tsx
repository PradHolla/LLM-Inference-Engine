import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Trace } from "../types";

export function TraceChart({ traces, theme }: { traces: Trace[]; theme: "cream" | "dark" }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<uPlot | null>(null);
  const dataRef = useRef<uPlot.AlignedData>([[], [], []]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const foreground = theme === "dark" ? "#c1c0b7" : "#5c605b";
    const grid = theme === "dark" ? "#41433d" : "#d8d3c7";
    const options: uPlot.Options = {
      width: Math.max(280, host.clientWidth), height: 150,
      padding: [8, 8, 0, 8],
      scales: { x: { time: false }, ttft: { auto: true }, prompt: { auto: true } },
      axes: [
        { stroke: foreground, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "10px ui-monospace, monospace" },
        { scale: "ttft", stroke: "#b86748", grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 }, font: "10px ui-monospace, monospace", values: (_u, values) => values.map((v) => `${Math.round(v)}ms`) },
        { scale: "prompt", side: 1, stroke: "#357b70", grid: { show: false }, ticks: { stroke: grid, width: 1 }, font: "10px ui-monospace, monospace", values: (_u, values) => values.map((v) => `${Math.round(v)}`) },
      ],
      series: [
        {},
        { label: "TTFT", scale: "ttft", stroke: "#b86748", width: 1.8, spanGaps: false, points: { show: true, size: 4, fill: "#b86748" } },
        { label: "Prompt", scale: "prompt", stroke: "#357b70", width: 1.6, spanGaps: false, points: { show: true, size: 3, fill: "#357b70" } },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false, setScale: false } },
    };
    chartRef.current = new uPlot(options, dataRef.current, host);
    const observer = new ResizeObserver(() => {
      if (chartRef.current && host.clientWidth > 0) chartRef.current.setSize({ width: host.clientWidth, height: 150 });
    });
    observer.observe(host);
    return () => { observer.disconnect(); chartRef.current?.destroy(); chartRef.current = null; };
  }, [theme]);

  useEffect(() => {
    const sorted = traces.slice(-20);
    dataRef.current = [
      sorted.map((_trace, index) => index + 1),
      sorted.map((trace) => typeof trace.ttft_ms === "number" ? trace.ttft_ms : null),
      sorted.map((trace) => typeof trace.prompt_tokens === "number" ? trace.prompt_tokens : null),
    ];
    chartRef.current?.setData(dataRef.current);
  }, [traces]);

  return <div className="chart-wrap">
    <div className="chart-legend"><span><i className="legend-ttft" /> TTFT, ms</span><span><i className="legend-prompt" /> Prompt tokens</span></div>
    <div className="trace-chart" ref={hostRef} aria-label="Measured TTFT and prompt tokens per turn" />
  </div>;
}
