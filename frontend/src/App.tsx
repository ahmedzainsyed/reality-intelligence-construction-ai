// =============================================================================
// Reality Intelligence Platform – Frontend Dashboard
// src/App.tsx  (root application shell)
// =============================================================================

import React, { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { Toaster } from "react-hot-toast";

import { useAuthStore } from "./store/authStore";
import { AppShell } from "./components/common/AppShell";
import { LoadingScreen } from "./components/common/LoadingScreen";
import { ErrorBoundary } from "./components/common/ErrorBoundary";

// Lazy-loaded pages for code-splitting
const LoginPage          = lazy(() => import("./pages/LoginPage"));
const DashboardPage      = lazy(() => import("./pages/DashboardPage"));
const ProjectsPage       = lazy(() => import("./pages/ProjectsPage"));
const ProjectDetailPage  = lazy(() => import("./pages/ProjectDetailPage"));
const SiteViewerPage     = lazy(() => import("./pages/SiteViewerPage"));
const ProgressPage       = lazy(() => import("./pages/ProgressPage"));
const AnalyticsPage      = lazy(() => import("./pages/AnalyticsPage"));
const TimelinePage       = lazy(() => import("./pages/TimelinePage"));
const BIMComparisonPage  = lazy(() => import("./pages/BIMComparisonPage"));
const UploadsPage        = lazy(() => import("./pages/UploadsPage"));
const AlertsPage         = lazy(() => import("./pages/AlertsPage"));
const SettingsPage       = lazy(() => import("./pages/SettingsPage"));

// React Query global client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2,      // 2 minutes
      gcTime:    1000 * 60 * 10,     // 10 minutes
      retry: (failureCount, error: any) => {
        if (error?.response?.status === 401) return false;
        if (error?.response?.status === 404) return false;
        return failureCount < 2;
      },
      refetchOnWindowFocus: false,
    },
  },
});

// Auth guard HOC
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s) => s.token);
  return token ? <>{children}</> : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Suspense fallback={<LoadingScreen />}>
            <Routes>
              {/* Public */}
              <Route path="/login" element={<LoginPage />} />

              {/* Protected – wrapped in AppShell (sidebar + topbar) */}
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <AppShell />
                  </ProtectedRoute>
                }
              >
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard"              element={<DashboardPage />} />
                <Route path="projects"               element={<ProjectsPage />} />
                <Route path="projects/:id"           element={<ProjectDetailPage />} />
                <Route path="projects/:id/viewer"    element={<SiteViewerPage />} />
                <Route path="projects/:id/progress"  element={<ProgressPage />} />
                <Route path="projects/:id/analytics" element={<AnalyticsPage />} />
                <Route path="projects/:id/timeline"  element={<TimelinePage />} />
                <Route path="projects/:id/bim"       element={<BIMComparisonPage />} />
                <Route path="projects/:id/uploads"   element={<UploadsPage />} />
                <Route path="alerts"                 element={<AlertsPage />} />
                <Route path="settings"               element={<SettingsPage />} />
                <Route path="*"                      element={<Navigate to="/dashboard" replace />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>

        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: { background: "#1e293b", color: "#f1f5f9", borderRadius: "8px" },
          }}
        />
        {process.env.NODE_ENV === "development" && <ReactQueryDevtools />}
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
