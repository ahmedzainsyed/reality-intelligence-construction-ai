/**
 * Auth Store (Zustand)
 *
 * Manages authentication state:
 *  - JWT token storage
 *  - User profile
 *  - Login / logout actions
 *  - Organisation context
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { tokenStorage, authApi } from "../services/api";

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: "admin" | "project_manager" | "site_engineer" | "viewer";
  organization_id: string;
  organization_name?: string;
  avatar_url?: string;
}

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: UserProfile | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  login:         (username: string, password: string) => Promise<void>;
  logout:        () => void;
  setUser:       (user: UserProfile) => void;
  fetchProfile:  () => Promise<void>;
  clearError:    () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token:        null,
      refreshToken: null,
      user:         null,
      isLoading:    false,
      error:        null,

      login: async (username, password) => {
        set({ isLoading: true, error: null });
        try {
          const { data } = await authApi.login(username, password);
          tokenStorage.set(data.access_token);
          if (data.refresh_token) tokenStorage.setRefresh(data.refresh_token);

          set({
            token:        data.access_token,
            refreshToken: data.refresh_token ?? null,
            isLoading:    false,
            error:        null,
          });

          // Fetch user profile immediately after login
          await get().fetchProfile();
        } catch (err: any) {
          const message =
            err?.response?.data?.message ?? "Invalid credentials. Please try again.";
          set({ isLoading: false, error: message, token: null, user: null });
          throw err;
        }
      },

      logout: () => {
        tokenStorage.removeAll();
        set({ token: null, refreshToken: null, user: null, error: null });
      },

      setUser: (user) => set({ user }),

      fetchProfile: async () => {
        try {
          const { data } = await authApi.me();
          set({ user: data });
        } catch {
          // If profile fetch fails, don't log out (token might be valid)
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name:    "rip-auth",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        token:        state.token,
        refreshToken: state.refreshToken,
        user:         state.user,
      }),
    }
  )
);

// ── Selectors ──────────────────────────────────────────────────────────────────
export const selectIsAuthenticated = (s: AuthState) => !!s.token;
export const selectUserRole        = (s: AuthState) => s.user?.role;
export const selectIsAdmin         = (s: AuthState) => s.user?.role === "admin";
export const selectOrgId           = (s: AuthState) => s.user?.organization_id;
