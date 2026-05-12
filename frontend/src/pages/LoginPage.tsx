/**
 * LoginPage.tsx – Authentication entry point
 *
 * Features:
 *  - Email + password form with validation
 *  - "Remember me" checkbox
 *  - Error messaging
 *  - Animated brand panel
 *  - Redirects to dashboard on success
 */

import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import {
  Building2, Eye, EyeOff, Loader2, Layers, BarChart3, Box,
} from "lucide-react";

export default function LoginPage() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const { login, isLoading, error, clearError } = useAuthStore();

  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [showPwd,  setShowPwd]  = useState(false);

  const from = (location.state as any)?.from?.pathname ?? "/dashboard";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    try {
      await login(email, password);
      navigate(from, { replace: true });
    } catch {
      // Error is already set in the store
    }
  };

  return (
    <div className="min-h-screen flex bg-slate-950">
      {/* ── Left brand panel ─────────────────────────────────────────── */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12
                      bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 relative overflow-hidden">
        {/* Background grid */}
        <div className="absolute inset-0 opacity-10"
             style={{
               backgroundImage: "linear-gradient(rgba(148,163,184,.3) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,.3) 1px, transparent 1px)",
               backgroundSize: "40px 40px",
             }} />

        {/* Floating orbs */}
        <div className="absolute top-20 right-20 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-10 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />

        {/* Logo */}
        <div className="relative flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center">
            <Building2 size={22} className="text-white" />
          </div>
          <span className="text-white font-bold text-xl tracking-tight">
            Reality Intelligence
          </span>
        </div>

        {/* Feature list */}
        <div className="relative space-y-6">
          <h1 className="text-4xl font-bold text-white leading-tight">
            AI-Powered<br />
            <span className="text-blue-400">Construction Intelligence</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-md">
            Process drone footage, CCTV streams, and mobile walkthroughs to
            generate real-time 3D construction progress intelligence.
          </p>

          <div className="space-y-4 pt-4">
            {[
              { icon: Box,      label: "3D Reconstruction",      sub: "SfM + MVS dense point clouds" },
              { icon: BarChart3, label: "Progress Analytics",    sub: "Real-time completion tracking" },
              { icon: Layers,    label: "BIM Comparison",        sub: "Planned vs actual deviation" },
            ].map(({ icon: Icon, label, sub }) => (
              <div key={label} className="flex items-center gap-4">
                <div className="w-9 h-9 rounded-lg bg-blue-500/20 border border-blue-500/30
                                flex items-center justify-center flex-shrink-0">
                  <Icon size={16} className="text-blue-400" />
                </div>
                <div>
                  <p className="text-white text-sm font-medium">{label}</p>
                  <p className="text-slate-500 text-xs">{sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <p className="relative text-slate-600 text-xs">
          © 2024 Reality Intelligence Platform · Enterprise-grade AI for construction
        </p>
      </div>

      {/* ── Right login form ──────────────────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-sm space-y-8">
          {/* Mobile logo */}
          <div className="flex lg:hidden items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center">
              <Building2 size={20} className="text-white" />
            </div>
            <span className="text-white font-bold text-lg">Reality Intelligence</span>
          </div>

          <div>
            <h2 className="text-2xl font-bold text-white">Sign in</h2>
            <p className="text-slate-400 text-sm mt-1">
              Welcome back. Enter your credentials to continue.
            </p>
          </div>

          {/* Error banner */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3
                            text-red-400 text-sm flex items-center gap-2">
              <span className="text-red-500">●</span>
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Email address
              </label>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5
                           text-white placeholder-slate-500 text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           transition-colors"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPwd ? "text" : "password"}
                  required
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 pr-10
                             text-white placeholder-slate-500 text-sm
                             focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                             transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPwd((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  tabIndex={-1}
                >
                  {showPwd ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800
                         text-white font-medium py-2.5 rounded-lg text-sm
                         flex items-center justify-center gap-2
                         transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {isLoading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Signing in…
                </>
              ) : (
                "Sign in"
              )}
            </button>
          </form>

          {/* Demo credentials hint */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3 text-xs text-slate-400 space-y-1">
            <p className="font-medium text-slate-300">Demo credentials</p>
            <p>Email: <span className="text-blue-400 font-mono">demo@reality-intelligence.io</span></p>
            <p>Password: <span className="text-blue-400 font-mono">Demo2024!</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}
