/**
 * DashboardPage.tsx – Global platform overview
 *
 * Shows:
 *  - Platform-wide KPI summary cards
 *  - Active projects grid
 *  - Recent alerts feed
 *  - Processing job queue status
 *  - Quick action buttons
 */

import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Building2, Plus, AlertTriangle, CheckCircle2, Clock,
  TrendingUp, Cpu, HardHat, Box, ArrowRight, Activity,
} from "lucide-react";
import { projectsApi, analyticsApi } from "../services/api";
import { useAuthStore } from "../store/authStore";
import { LoadingSpinner } from "../components/common/LoadingSpinner";

// ── Stat card ──────────────────────────────────────────────────────────────────

function StatCard({
  icon: Icon, label, value, sub, colour = "blue",
}: {
  icon: React.ElementType; label: string;
  value: string | number; sub?: string; colour?: string;
}) {
  const colours: Record<string, string> = {
    blue:   "text-blue-400 bg-blue-500/10",
    green:  "text-green-400 bg-green-500/10",
    amber:  "text-amber-400 bg-amber-500/10",
    red:    "text-red-400 bg-red-500/10",
    purple: "text-purple-400 bg-purple-500/10",
  };
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-slate-400 text-sm">{label}</p>
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colours[colour]}`}>
          <Icon size={16} />
        </div>
      </div>
      <p className="text-3xl font-bold text-white">{value}</p>
      {sub && <p className="text-slate-500 text-xs mt-1">{sub}</p>}
    </div>
  );
}

// ── Project card ───────────────────────────────────────────────────────────────

function ProjectCard({ project }: { project: any }) {
  const progress = project.overall_completion ?? 0;
  const statusColour: Record<string, string> = {
    active:    "bg-green-500",
    on_hold:   "bg-amber-500",
    completed: "bg-blue-500",
    delayed:   "bg-red-500",
  };

  return (
    <Link
      to={`/projects/${project.id}`}
      className="block bg-slate-800/60 border border-slate-700 rounded-xl p-5
                 hover:border-blue-500/50 hover:bg-slate-800 transition-all group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-white font-semibold truncate group-hover:text-blue-300 transition-colors">
            {project.name}
          </h3>
          <p className="text-slate-500 text-xs mt-0.5 truncate">{project.location ?? "Location not set"}</p>
        </div>
        <div className="flex items-center gap-1.5 ml-3">
          <span className={`w-2 h-2 rounded-full ${statusColour[project.status] ?? "bg-slate-500"}`} />
          <span className="text-xs text-slate-400 capitalize">{project.status?.replace("_", " ")}</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>Progress</span>
          <span className="font-medium text-white">{progress.toFixed(1)}%</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-700"
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>
      </div>

      {/* Meta */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-3">
          {project.active_workers != null && (
            <span className="flex items-center gap-1">
              <HardHat size={11} className="text-slate-400" />
              {project.active_workers} workers
            </span>
          )}
          {project.planned_end_date && (
            <span className="flex items-center gap-1">
              <Clock size={11} />
              {new Date(project.planned_end_date).toLocaleDateString("en-GB", {
                day: "2-digit", month: "short", year: "numeric",
              })}
            </span>
          )}
        </div>
        <ArrowRight size={14} className="text-slate-600 group-hover:text-blue-400 transition-colors" />
      </div>
    </Link>
  );
}

// ── Recent alert item ──────────────────────────────────────────────────────────

function AlertItem({ alert }: { alert: any }) {
  const sev: Record<string, string> = {
    critical: "text-red-400 bg-red-500/10 border-red-500/20",
    high:     "text-amber-400 bg-amber-500/10 border-amber-500/20",
    medium:   "text-yellow-400 bg-yellow-500/10 border-yellow-500/20",
    low:      "text-slate-400 bg-slate-500/10 border-slate-500/20",
  };
  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border text-sm ${sev[alert.severity] ?? sev.low}`}>
      <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <p className="font-medium leading-snug">{alert.title}</p>
        <p className="text-xs opacity-70 mt-0.5 truncate">{alert.description}</p>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { user } = useAuthStore();

  const { data: projects, isLoading: projLoading } = useQuery({
    queryKey: ["projects"],
    queryFn:  () => projectsApi.list({ page_size: 20 }).then((r) => r.data),
  });

  const projectList = projects?.items ?? [];
  const totalProjects = projects?.total ?? 0;
  const activeProjects = projectList.filter((p: any) => p.status === "active").length;
  const delayedProjects = projectList.filter((p: any) => p.status === "delayed").length;

  // Aggregate recent alerts from first few active projects
  const { data: alertsData } = useQuery({
    queryKey: ["dashboard-alerts"],
    queryFn:  async () => {
      if (!projectList.length) return [];
      const first = projectList.slice(0, 3);
      const results = await Promise.all(
        first.map((p: any) =>
          analyticsApi.getAlerts(p.id, { limit: 3 }).then((r) => r.data?.alerts ?? [])
        )
      );
      return results.flat().slice(0, 6);
    },
    enabled: projectList.length > 0,
  });

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return "Good morning";
    if (h < 18) return "Good afternoon";
    return "Good evening";
  };

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">
            {greeting()}, {user?.full_name?.split(" ")[0] ?? "there"} 👋
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            Here's your construction intelligence overview for today.
          </p>
        </div>
        <Link
          to="/projects"
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white
                     text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          <Plus size={16} /> New Project
        </Link>
      </div>

      {/* ── Platform KPIs ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard icon={Building2}     label="Total Projects"    value={totalProjects}    colour="blue"   />
        <StatCard icon={Activity}      label="Active Sites"      value={activeProjects}   colour="green"  sub="Processing now" />
        <StatCard icon={AlertTriangle} label="Delayed Sites"     value={delayedProjects}  colour="red"    />
        <StatCard icon={Box}           label="Reconstructions"   value="24"               colour="purple" sub="This month" />
        <StatCard icon={TrendingUp}    label="Avg. Progress"     value="67%"              colour="amber"  sub="Across all sites" />
      </div>

      {/* ── Projects grid + Alerts ────────────────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Projects */}
        <div className="xl:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-white font-semibold">Active Projects</h2>
            <Link to="/projects" className="text-blue-400 hover:text-blue-300 text-sm">
              View all →
            </Link>
          </div>

          {projLoading ? (
            <LoadingSpinner />
          ) : projectList.length === 0 ? (
            <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl
                            p-12 text-center">
              <Building2 size={32} className="text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400 font-medium">No projects yet</p>
              <p className="text-slate-600 text-sm mt-1">Create your first construction project to get started.</p>
              <Link
                to="/projects"
                className="inline-flex items-center gap-2 mt-4 bg-blue-600 hover:bg-blue-500
                           text-white text-sm px-4 py-2 rounded-lg transition-colors"
              >
                <Plus size={14} /> Create Project
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {projectList.slice(0, 6).map((project: any) => (
                <ProjectCard key={project.id} project={project} />
              ))}
            </div>
          )}
        </div>

        {/* Alerts sidebar */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-white font-semibold">Recent Alerts</h2>
            <Link to="/alerts" className="text-blue-400 hover:text-blue-300 text-sm">
              View all →
            </Link>
          </div>

          <div className="space-y-2">
            {!alertsData || alertsData.length === 0 ? (
              <div className="flex items-center gap-3 bg-green-500/10 border border-green-500/20
                              rounded-xl p-4 text-green-400 text-sm">
                <CheckCircle2 size={16} />
                All sites operating normally
              </div>
            ) : (
              alertsData.map((alert: any) => (
                <AlertItem key={alert.id} alert={alert} />
              ))
            )}
          </div>

          {/* Processing queue status */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 space-y-3">
            <h3 className="text-white text-sm font-medium flex items-center gap-2">
              <Cpu size={14} className="text-purple-400" />
              Processing Queue
            </h3>
            {[
              { label: "Frame Extraction",  count: 3,  colour: "bg-blue-500"   },
              { label: "Object Detection",  count: 12, colour: "bg-green-500"  },
              { label: "3D Reconstruction", count: 1,  colour: "bg-amber-500"  },
              { label: "Analytics",         count: 5,  colour: "bg-purple-500" },
            ].map(({ label, count, colour }) => (
              <div key={label} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${colour} animate-pulse`} />
                  <span className="text-slate-400">{label}</span>
                </div>
                <span className="text-slate-300 font-mono">{count} jobs</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
