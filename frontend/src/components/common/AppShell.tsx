/**
 * AppShell.tsx
 *
 * Primary layout wrapper for authenticated pages.
 * Contains collapsible sidebar, top bar, and content area.
 */

import React, { useState } from "react";
import { Outlet, NavLink, useNavigate, useParams } from "react-router-dom";
import {
  Building2, LayoutDashboard, FolderOpen, Box, BarChart3,
  Clock, AlertTriangle, Upload, Settings, LogOut, ChevronLeft,
  ChevronRight, Bell, Search, Menu, GitCompare, HardHat,
} from "lucide-react";
import { useAuthStore } from "../../store/authStore";
import clsx from "clsx";

// ── Nav items ──────────────────────────────────────────────────────────────────

const TOP_NAV = [
  { to: "/dashboard",  icon: LayoutDashboard, label: "Dashboard" },
  { to: "/projects",   icon: FolderOpen,      label: "Projects"  },
  { to: "/alerts",     icon: AlertTriangle,   label: "Alerts",   badge: 3 },
];

const PROJECT_NAV = (id: string) => [
  { to: `/projects/${id}`,           icon: LayoutDashboard, label: "Overview"    },
  { to: `/projects/${id}/viewer`,    icon: Box,             label: "3D Viewer"   },
  { to: `/projects/${id}/progress`,  icon: HardHat,         label: "Progress"    },
  { to: `/projects/${id}/analytics`, icon: BarChart3,       label: "Analytics"   },
  { to: `/projects/${id}/timeline`,  icon: Clock,           label: "Timeline"    },
  { to: `/projects/${id}/bim`,       icon: GitCompare,      label: "BIM Compare" },
  { to: `/projects/${id}/uploads`,   icon: Upload,          label: "Uploads"     },
];

// ── Sidebar link ───────────────────────────────────────────────────────────────

function SidebarLink({
  to, icon: Icon, label, badge, collapsed,
}: {
  to: string; icon: React.ElementType; label: string;
  badge?: number; collapsed: boolean;
}) {
  return (
    <NavLink
      to={to}
      title={collapsed ? label : undefined}
      className={({ isActive }) =>
        clsx(
          "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors group",
          isActive
            ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
            : "text-slate-400 hover:text-white hover:bg-slate-700/60"
        )
      }
    >
      <Icon size={16} className="flex-shrink-0" />
      {!collapsed && <span className="flex-1 truncate">{label}</span>}
      {!collapsed && badge && badge > 0 && (
        <span className="text-xs bg-red-500 text-white rounded-full px-1.5 py-0.5 min-w-[20px] text-center">
          {badge}
        </span>
      )}
    </NavLink>
  );
}

// ── Top bar ────────────────────────────────────────────────────────────────────

function TopBar({ onToggleMobile }: { onToggleMobile: () => void }) {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 flex items-center px-4 gap-4 flex-shrink-0">
      <button onClick={onToggleMobile} className="lg:hidden text-slate-400 hover:text-white">
        <Menu size={20} />
      </button>

      {/* Search */}
      <div className="flex-1 max-w-md hidden md:flex items-center gap-2
                      bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5">
        <Search size={14} className="text-slate-500" />
        <input
          placeholder="Search projects, sites…"
          className="bg-transparent text-sm text-white placeholder-slate-500
                     focus:outline-none w-full"
        />
        <kbd className="text-xs text-slate-600 font-mono bg-slate-700 px-1.5 py-0.5 rounded">⌘K</kbd>
      </div>

      <div className="ml-auto flex items-center gap-3">
        {/* Notifications */}
        <button className="relative text-slate-400 hover:text-white p-1.5 rounded-lg hover:bg-slate-800">
          <Bell size={18} />
          <span className="absolute top-0.5 right-0.5 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        {/* User avatar */}
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-medium">
            {user?.full_name?.charAt(0) ?? "U"}
          </div>
          <div className="hidden md:block text-sm">
            <p className="text-white font-medium leading-none">{user?.full_name ?? "User"}</p>
            <p className="text-slate-500 text-xs capitalize mt-0.5">{user?.role?.replace("_", " ")}</p>
          </div>
        </div>

        <button
          onClick={handleLogout}
          className="text-slate-500 hover:text-red-400 p-1.5 rounded-lg hover:bg-slate-800 transition-colors"
          title="Sign out"
        >
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}

// ── Main shell ─────────────────────────────────────────────────────────────────

export function AppShell() {
  const [collapsed,   setCollapsed]   = useState(false);
  const [mobileOpen,  setMobileOpen]  = useState(false);
  const { id: projectId } = useParams<{ id?: string }>();

  const sidebarWidth = collapsed ? "w-14" : "w-56";

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      {/* ── Mobile overlay ────────────────────────────────────────────── */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-20 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ── Sidebar ───────────────────────────────────────────────────── */}
      <aside
        className={clsx(
          "flex-shrink-0 flex flex-col bg-slate-900 border-r border-slate-800 transition-all duration-200",
          sidebarWidth,
          "fixed lg:relative inset-y-0 left-0 z-30",
          mobileOpen ? "translate-x-0 w-56" : "-translate-x-full lg:translate-x-0",
        )}
      >
        {/* Brand */}
        <div className="h-14 flex items-center gap-3 px-3 border-b border-slate-800 flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
            <Building2 size={16} className="text-white" />
          </div>
          {!collapsed && (
            <span className="text-white font-semibold text-sm truncate">
              Reality Intelligence
            </span>
          )}
        </div>

        {/* Nav content */}
        <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
          {/* Global nav */}
          {TOP_NAV.map((item) => (
            <SidebarLink key={item.to} {...item} collapsed={collapsed} />
          ))}

          {/* Project-specific nav */}
          {projectId && (
            <>
              <div className={clsx("pt-3 pb-1", !collapsed && "px-3")}>
                {!collapsed && (
                  <p className="text-xs font-medium text-slate-600 uppercase tracking-wider">
                    Current Project
                  </p>
                )}
              </div>
              {PROJECT_NAV(projectId).map((item) => (
                <SidebarLink key={item.to} {...item} collapsed={collapsed} />
              ))}
            </>
          )}
        </nav>

        {/* Bottom */}
        <div className="p-2 border-t border-slate-800 space-y-1">
          <SidebarLink to="/settings" icon={Settings} label="Settings" collapsed={collapsed} />

          {/* Collapse toggle (desktop only) */}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="hidden lg:flex w-full items-center gap-3 px-3 py-2 rounded-lg
                       text-slate-500 hover:text-white hover:bg-slate-700/60 text-sm transition-colors"
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar onToggleMobile={() => setMobileOpen((v) => !v)} />

        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
