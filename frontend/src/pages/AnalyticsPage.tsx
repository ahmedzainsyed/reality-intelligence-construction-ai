/**
 * AnalyticsPage.tsx
 *
 * Main construction analytics dashboard.
 * Shows progress trends, equipment utilisation, delay risk,
 * activity heatmaps, and BIM comparison summary.
 */

import React, { useMemo } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import {
  TrendingUp, TrendingDown, AlertTriangle, CheckCircle2,
  HardHat, Truck, Clock, BarChart3, Activity, Calendar,
} from "lucide-react";

import { api } from "../services/api";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import { ProgressGauge } from "../components/dashboard/ProgressGauge";
import { RiskBadge } from "../components/dashboard/RiskBadge";
import { SiteHeatmap } from "../components/visualization/SiteHeatmap";

// ── API calls ─────────────────────────────────────────────────────────────────

const useProjectAnalytics = (projectId: string) =>
  useQuery({
    queryKey: ["analytics", "summary", projectId],
    queryFn:  () => api.get(`/analytics/summary/${projectId}`).then(r => r.data),
    enabled:  !!projectId,
    refetchInterval: 60_000,
  });

const useTimeline = (projectId: string) =>
  useQuery({
    queryKey: ["analytics", "timeline", projectId],
    queryFn:  () => api.get(`/analytics/timeline/${projectId}?granularity=weekly`).then(r => r.data),
    enabled:  !!projectId,
  });

const useDelayPrediction = (projectId: string) =>
  useQuery({
    queryKey: ["analytics", "delays", projectId],
    queryFn:  () => api.get(`/analytics/delays/${projectId}`).then(r => r.data),
    enabled:  !!projectId,
    refetchInterval: 300_000,
  });

const useHeatmap = (projectId: string) =>
  useQuery({
    queryKey: ["analytics", "heatmap", projectId],
    queryFn:  () => api.get(`/analytics/heatmap/${projectId}?heatmap_type=activity`).then(r => r.data),
    enabled:  !!projectId,
  });

// ── KPI Card ──────────────────────────────────────────────────────────────────

interface KPICardProps {
  title: string;
  value: string | number;
  unit?: string;
  delta?: number;
  icon: React.ReactNode;
  colour?: string;
}

function KPICard({ title, value, unit, delta, icon, colour = "blue" }: KPICardProps) {
  const colourMap: Record<string, string> = {
    blue:   "from-blue-500/20 to-blue-600/10 border-blue-500/30",
    green:  "from-green-500/20 to-green-600/10 border-green-500/30",
    amber:  "from-amber-500/20 to-amber-600/10 border-amber-500/30",
    red:    "from-red-500/20 to-red-600/10   border-red-500/30",
    purple: "from-purple-500/20 to-purple-600/10 border-purple-500/30",
  };
  const isDeltaPositive = delta !== undefined && delta >= 0;

  return (
    <div className={`rounded-xl bg-gradient-to-br ${colourMap[colour]} border p-4 backdrop-blur`}>
      <div className="flex items-start justify-between mb-3">
        <p className="text-slate-400 text-sm">{title}</p>
        <span className="text-slate-500">{icon}</span>
      </div>
      <div className="flex items-end gap-2">
        <span className="text-3xl font-bold text-white">{value}</span>
        {unit && <span className="text-slate-400 text-sm mb-1">{unit}</span>}
      </div>
      {delta !== undefined && (
        <div className={`flex items-center gap-1 mt-2 text-xs ${isDeltaPositive ? "text-green-400" : "text-red-400"}`}>
          {isDeltaPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
          <span>{Math.abs(delta).toFixed(1)}% vs last week</span>
        </div>
      )}
    </div>
  );
}

// ── Progress Timeline Chart ────────────────────────────────────────────────────

function ProgressTimelineChart({ data }: { data: any }) {
  const dates   = data?.snapshots?.map((s: any) => s.snapshot_date) ?? [];
  const actual  = data?.snapshots?.map((s: any) => s.overall_progress_percent) ?? [];
  const planned = data?.snapshots?.map((s: any) => s.planned_progress_percent) ?? [];

  const traces: Plotly.Data[] = [
    {
      x: dates, y: actual, type: "scatter", mode: "lines+markers",
      name: "Actual Progress", line: { color: "#3b82f6", width: 2 },
      marker: { size: 4 },
    },
    {
      x: dates, y: planned, type: "scatter", mode: "lines",
      name: "Planned Progress", line: { color: "#64748b", width: 2, dash: "dash" },
    },
    {
      x: dates, y: actual.map((a: number, i: number) => a - (planned[i] ?? 0)),
      type: "bar", name: "Variance", yaxis: "y2",
      marker: {
        color: actual.map((a: number, i: number) =>
          a >= (planned[i] ?? 0) ? "#22c55e44" : "#ef444444"
        ),
      },
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#94a3b8", size: 11 },
    legend: { orientation: "h", y: -0.15 },
    margin: { t: 10, l: 40, r: 40, b: 40 },
    yaxis: {
      title: "Progress (%)", range: [0, 105],
      gridcolor: "#1e293b", zeroline: false,
    },
    yaxis2: {
      title: "Variance (%)", overlaying: "y", side: "right",
      gridcolor: "#1e293b", zeroline: true, zerolinecolor: "#334155",
    },
    xaxis: { gridcolor: "#1e293b" },
  };

  return (
    <Plot
      data={traces} layout={layout}
      style={{ width: "100%", height: "300px" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

// ── Equipment Utilisation Chart ───────────────────────────────────────────────

function EquipmentChart({ data }: { data: any }) {
  const items = data?.equipment_types ?? [];
  const traces: Plotly.Data[] = [
    {
      type: "bar", orientation: "h",
      x: items.map((e: any) => e.utilisation_pct),
      y: items.map((e: any) => e.type_name),
      marker: {
        color: items.map((e: any) =>
          e.utilisation_pct > 70 ? "#22c55e" : e.utilisation_pct > 40 ? "#f59e0b" : "#ef4444"
        ),
      },
      text: items.map((e: any) => `${e.utilisation_pct.toFixed(0)}%`),
      textposition: "outside",
    },
  ];

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: { color: "#94a3b8", size: 11 },
    margin: { t: 10, l: 120, r: 50, b: 30 },
    xaxis: { range: [0, 110], gridcolor: "#1e293b", ticksuffix: "%" },
    yaxis: { automargin: true },
  };

  return (
    <Plot data={traces} layout={layout}
      style={{ width: "100%", height: "250px" }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

// ── Delay Risk Gauge ──────────────────────────────────────────────────────────

function DelayRiskPanel({ delay }: { delay: any }) {
  if (!delay) return <LoadingSpinner />;

  const prob = Math.round((delay.delay_probability ?? 0) * 100);
  const days = delay.predicted_delay_days ?? 0;
  const risk = delay.risk_level ?? "low";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm">Predicted Delay</p>
          <p className="text-4xl font-bold text-white mt-1">
            {days.toFixed(0)}
            <span className="text-lg text-slate-400 ml-1">days</span>
          </p>
        </div>
        <RiskBadge risk={risk} />
      </div>

      {/* Probability bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Delay probability</span><span>{prob}%</span>
        </div>
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              prob >= 70 ? "bg-red-500" : prob >= 40 ? "bg-amber-500" : "bg-green-500"
            }`}
            style={{ width: `${prob}%` }}
          />
        </div>
      </div>

      {/* Top risk factors */}
      <div>
        <p className="text-slate-400 text-xs mb-2">Top risk factors</p>
        <div className="space-y-1.5">
          {(delay.risk_factors ?? []).slice(0, 4).map((rf: any) => (
            <div key={rf.feature} className="flex items-center gap-2 text-xs">
              <div className="flex-1 text-slate-300 truncate">{rf.factor}</div>
              <div className="w-20 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500 rounded-full"
                  style={{ width: `${Math.round(rf.importance * 100)}%` }}
                />
              </div>
              <span className="text-slate-500 w-10 text-right">
                {Math.round(rf.importance * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {delay.revised_completion_date && (
        <p className="text-xs text-slate-500 border-t border-slate-700 pt-3">
          Revised completion:{" "}
          <span className="text-amber-400 font-medium">
            {new Date(delay.revised_completion_date).toLocaleDateString("en-GB", {
              day: "2-digit", month: "short", year: "numeric",
            })}
          </span>
        </p>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const { id: projectId } = useParams<{ id: string }>();

  const { data: summary, isLoading: summaryLoading } = useProjectAnalytics(projectId!);
  const { data: timeline }  = useTimeline(projectId!);
  const { data: delay }     = useDelayPrediction(projectId!);
  const { data: heatmap }   = useHeatmap(projectId!);

  if (summaryLoading) return <LoadingSpinner fullPage />;

  const progress = summary?.progress;
  const equipment = summary?.equipment;

  return (
    <div className="p-6 space-y-6">
      {/* ── Page header ───────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Site Analytics</h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {summary?.project?.name ?? "Loading…"} · Real-time construction intelligence
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Activity size={12} className="text-green-400 animate-pulse" />
          Live · Updated {new Date().toLocaleTimeString()}
        </div>
      </div>

      {/* ── KPI row ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <KPICard
          title="Overall Progress" icon={<CheckCircle2 size={18} />}
          value={`${(progress?.overall_progress_percent ?? 0).toFixed(1)}`}
          unit="%" delta={progress?.progress_velocity_7d} colour="blue"
        />
        <KPICard
          title="Active Workers" icon={<HardHat size={18} />}
          value={progress?.active_workers ?? 0} colour="green"
        />
        <KPICard
          title="Equipment On-site" icon={<Truck size={18} />}
          value={progress?.active_equipment ?? 0} colour="purple"
        />
        <KPICard
          title="Predicted Delay" icon={<Clock size={18} />}
          value={`${(delay?.predicted_delay_days ?? 0).toFixed(0)}`}
          unit="days" colour={delay?.risk_level === "low" ? "green" : "amber"}
        />
        <KPICard
          title="Schedule Variance" icon={<BarChart3 size={18} />}
          value={`${(progress?.progress_variance_percent ?? 0).toFixed(1)}`}
          unit="%"
          colour={(progress?.progress_variance_percent ?? 0) >= 0 ? "green" : "red"}
        />
        <KPICard
          title="Safety Alerts" icon={<AlertTriangle size={18} />}
          value={progress?.safety_violations_detected ?? 0}
          colour={progress?.safety_violations_detected > 0 ? "red" : "green"}
        />
      </div>

      {/* ── Main charts row ───────────────────────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Progress timeline – 2/3 width */}
        <div className="xl:col-span-2 bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4 flex items-center gap-2">
            <TrendingUp size={16} className="text-blue-400" />
            Progress Timeline
          </h2>
          <ProgressTimelineChart data={timeline} />
        </div>

        {/* Delay prediction – 1/3 width */}
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-400" />
            Delay Prediction
          </h2>
          <DelayRiskPanel delay={delay} />
        </div>
      </div>

      {/* ── Bottom row: equipment + heatmap + progress breakdown ─────── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4">Equipment Utilisation</h2>
          <EquipmentChart data={equipment} />
        </div>

        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4">Activity Heatmap</h2>
          <SiteHeatmap data={heatmap} height={250} />
        </div>

        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
          <h2 className="text-white font-semibold mb-4">Structural Progress</h2>
          <div className="space-y-3">
            {[
              { label: "Foundation",       key: "foundation_completion" },
              { label: "Structural Frame", key: "structural_frame_completion" },
              { label: "Slabs",            key: "slab_completion" },
              { label: "Walls",            key: "walls_completion" },
              { label: "MEP",              key: "mep_completion" },
              { label: "Finishing",        key: "finishing_completion" },
            ].map(({ label, key }) => {
              const pct = Math.round(progress?.[key] ?? 0);
              return (
                <div key={key}>
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>{label}</span><span>{pct}%</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
