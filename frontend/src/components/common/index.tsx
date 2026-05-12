/**
 * Common shared UI components
 *
 * LoadingSpinner   – animated loader for async states
 * LoadingScreen    – full-page loading overlay (Suspense fallback)
 * ErrorBoundary    – React error boundary wrapper
 * RiskBadge        – coloured risk level pill
 * EmptyState       – illustrated empty-state placeholder
 * ProgressBar      – horizontal percentage bar
 * StatusDot        – coloured status indicator
 */

import React, { Component, ReactNode } from "react";
import { AlertTriangle, RefreshCcw, Building2 } from "lucide-react";
import clsx from "clsx";

// ── LoadingSpinner ─────────────────────────────────────────────────────────────

export function LoadingSpinner({
  size = 24,
  className = "",
  fullPage = false,
}: {
  size?: number;
  className?: string;
  fullPage?: boolean;
}) {
  const spinner = (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={clsx("animate-spin text-blue-500", className)}
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"
              strokeDasharray="31.4" strokeDashoffset="10" strokeLinecap="round" />
    </svg>
  );

  if (fullPage) {
    return (
      <div className="flex items-center justify-center h-full w-full min-h-64">
        {spinner}
      </div>
    );
  }
  return spinner;
}

// ── LoadingScreen ──────────────────────────────────────────────────────────────

export function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-slate-950 flex flex-col items-center justify-center gap-4 z-50">
      <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center">
        <Building2 size={24} className="text-white" />
      </div>
      <LoadingSpinner size={28} />
      <p className="text-slate-500 text-sm">Loading Reality Intelligence…</p>
    </div>
  );
}

// ── ErrorBoundary ──────────────────────────────────────────────────────────────

interface EBState { hasError: boolean; error: Error | null }

export class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  EBState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="flex flex-col items-center justify-center h-full min-h-64
                        text-center p-8 space-y-4">
          <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
            <AlertTriangle size={24} className="text-red-400" />
          </div>
          <div>
            <p className="text-white font-semibold">Something went wrong</p>
            <p className="text-slate-400 text-sm mt-1 max-w-xs">
              {this.state.error?.message ?? "An unexpected error occurred."}
            </p>
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            <RefreshCcw size={14} /> Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── RiskBadge ─────────────────────────────────────────────────────────────────

const RISK_STYLES: Record<string, string> = {
  critical: "bg-red-500/20 text-red-400 border-red-500/30",
  high:     "bg-amber-500/20 text-amber-400 border-amber-500/30",
  medium:   "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  low:      "bg-green-500/20 text-green-400 border-green-500/30",
  on_track: "bg-green-500/20 text-green-400 border-green-500/30",
  at_risk:  "bg-amber-500/20 text-amber-400 border-amber-500/30",
  delayed:  "bg-red-500/20 text-red-400 border-red-500/30",
};

export function RiskBadge({
  risk, className = "",
}: {
  risk: string; className?: string;
}) {
  const style = RISK_STYLES[risk] ?? RISK_STYLES.low;
  const label = risk.replace("_", " ");
  return (
    <span
      className={clsx(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border capitalize",
        style, className
      )}
    >
      {label}
    </span>
  );
}

// ── EmptyState ────────────────────────────────────────────────────────────────

export function EmptyState({
  icon: Icon = Building2,
  title,
  description,
  action,
}: {
  icon?: React.ElementType;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-8 text-center">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700
                      flex items-center justify-center mb-4">
        <Icon size={24} className="text-slate-500" />
      </div>
      <p className="text-white font-semibold">{title}</p>
      {description && (
        <p className="text-slate-400 text-sm mt-2 max-w-sm">{description}</p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

// ── ProgressBar ────────────────────────────────────────────────────────────────

export function ProgressBar({
  value,
  max = 100,
  colour = "blue",
  height = "h-2",
  showLabel = false,
  className = "",
}: {
  value: number; max?: number; colour?: string;
  height?: string; showLabel?: boolean; className?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const colours: Record<string, string> = {
    blue:   "bg-blue-500",
    green:  "bg-green-500",
    amber:  "bg-amber-500",
    red:    "bg-red-500",
    purple: "bg-purple-500",
  };

  return (
    <div className={clsx("space-y-1", className)}>
      {showLabel && (
        <div className="flex justify-between text-xs text-slate-400">
          <span>{value.toFixed(1)}{max === 100 ? "%" : `/${max}`}</span>
        </div>
      )}
      <div className={clsx("w-full bg-slate-700 rounded-full overflow-hidden", height)}>
        <div
          className={clsx("h-full rounded-full transition-all duration-500", colours[colour] ?? colours.blue)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── StatusDot ─────────────────────────────────────────────────────────────────

export function StatusDot({
  status, pulse = false,
}: {
  status: "active" | "idle" | "error" | "warning" | "success";
  pulse?: boolean;
}) {
  const colours: Record<string, string> = {
    active:  "bg-green-500",
    idle:    "bg-slate-500",
    error:   "bg-red-500",
    warning: "bg-amber-500",
    success: "bg-blue-500",
  };
  return (
    <span
      className={clsx(
        "inline-block w-2 h-2 rounded-full",
        colours[status],
        pulse && "animate-pulse",
      )}
    />
  );
}

// ── SiteHeatmap (placeholder for Plotly heatmap) ──────────────────────────────

export function SiteHeatmap({
  data, height = 300,
}: {
  data: any; height?: number;
}) {
  if (!data?.cells?.length) {
    return (
      <div
        className="flex items-center justify-center bg-slate-700/30 rounded-lg text-slate-500 text-sm"
        style={{ height }}
      >
        Heatmap data loading…
      </div>
    );
  }

  // Render via Plotly heatmap
  const { default: Plot } = require("react-plotly.js");
  const grid: number[][] = [];
  const { grid_rows: rows, grid_cols: cols, cells } = data;

  for (let r = 0; r < rows; r++) {
    grid[r] = new Array(cols).fill(0);
  }
  cells.forEach((c: any) => {
    if (grid[c.y]) grid[c.y][c.x] = c.value;
  });

  return (
    <Plot
      data={[{
        type: "heatmap",
        z: grid,
        colorscale: "YlOrRd",
        showscale: false,
      }]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor:  "transparent",
        margin: { t: 0, l: 0, r: 0, b: 0 },
        xaxis: { visible: false },
        yaxis: { visible: false },
      }}
      style={{ width: "100%", height }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
