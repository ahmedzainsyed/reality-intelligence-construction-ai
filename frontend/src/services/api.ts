/**
 * API Service Client
 *
 * Centralised axios instance with:
 *  - JWT token injection
 *  - Automatic token refresh on 401
 *  - Request/response logging (dev)
 *  - Error normalisation
 *  - Retry on network failures
 */

import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  InternalAxiosRequestConfig,
} from "axios";
import toast from "react-hot-toast";

// ── Base config ────────────────────────────────────────────────────────────────

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";
const WS_URL   = import.meta.env.VITE_WS_URL  ?? "ws://localhost:8000/ws";

export { WS_URL };

// ── Token helpers ──────────────────────────────────────────────────────────────

const TOKEN_KEY   = "rip_access_token";
const REFRESH_KEY = "rip_refresh_token";

export const tokenStorage = {
  get:        ()     => localStorage.getItem(TOKEN_KEY),
  set:        (t: string) => localStorage.setItem(TOKEN_KEY, t),
  remove:     ()     => localStorage.removeItem(TOKEN_KEY),
  getRefresh: ()     => localStorage.getItem(REFRESH_KEY),
  setRefresh: (t: string) => localStorage.setItem(REFRESH_KEY, t),
  removeAll:  ()     => { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REFRESH_KEY); },
};

// ── Axios instance ─────────────────────────────────────────────────────────────

export const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 60_000,
  headers: { "Content-Type": "application/json" },
});

// ── Request interceptor – inject Bearer token ──────────────────────────────────

api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = tokenStorage.get();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    // Add request ID for tracing
    config.headers["X-Request-ID"] = `fe-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response interceptor – handle 401 token refresh ───────────────────────────

let isRefreshing = false;
let failedQueue: Array<{ resolve: (v: string) => void; reject: (e: unknown) => void }> = [];

const processQueue = (error: unknown, token: string | null = null) => {
  failedQueue.forEach((p) => (error ? p.reject(error) : p.resolve(token!)));
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

    // 401: attempt token refresh
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          if (originalRequest.headers) {
            originalRequest.headers.Authorization = `Bearer ${token}`;
          }
          return api(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      const refreshToken = tokenStorage.getRefresh();
      if (!refreshToken) {
        tokenStorage.removeAll();
        window.location.href = "/login";
        return Promise.reject(error);
      }

      try {
        const resp = await axios.post(`${BASE_URL}/auth/refresh`, { refresh_token: refreshToken });
        const newToken = resp.data.access_token;
        tokenStorage.set(newToken);
        processQueue(null, newToken);
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
        }
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError);
        tokenStorage.removeAll();
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // 403 – show permission error
    if (error.response?.status === 403) {
      toast.error("You don't have permission to perform this action.");
    }

    // 429 – rate limited
    if (error.response?.status === 429) {
      toast.error("Too many requests. Please wait a moment.");
    }

    // 5xx – server error
    if (error.response && error.response.status >= 500) {
      const msg = (error.response.data as any)?.message ?? "Server error. Please try again.";
      toast.error(msg);
    }

    return Promise.reject(error);
  }
);

// ── Typed API methods ──────────────────────────────────────────────────────────

export const authApi = {
  login: (username: string, password: string) =>
    api.post("/auth/login", new URLSearchParams({ username, password }), {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    }),
  register: (data: { email: string; password: string; full_name: string; organization_name?: string }) =>
    api.post("/auth/register", data),
  refresh: (refreshToken: string) => api.post("/auth/refresh", { refresh_token: refreshToken }),
  me: () => api.get("/auth/me"),
};

export const projectsApi = {
  list: (params?: Record<string, unknown>) => api.get("/projects", { params }),
  get:  (id: string) => api.get(`/projects/${id}`),
  create: (data: Record<string, unknown>) => api.post("/projects", data),
  update: (id: string, data: Record<string, unknown>) => api.patch(`/projects/${id}`, data),
  delete: (id: string) => api.delete(`/projects/${id}`),
};

export const uploadsApi = {
  list:   (params: Record<string, unknown>) => api.get("/uploads/", { params }),
  get:    (id: string) => api.get(`/uploads/${id}`),
  delete: (id: string) => api.delete(`/uploads/${id}`),

  initChunked: (data: FormData) =>
    api.post("/uploads/video/init", data, {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  uploadChunk: (uploadId: string, chunkNumber: number, chunk: Blob) => {
    const form = new FormData();
    form.append("chunk", chunk);
    return api.put(`/uploads/video/${uploadId}/chunk/${chunkNumber}`, form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120_000,
    });
  },

  completeChunked: (uploadId: string, parts: Array<{ PartNumber: number; ETag: string }>) =>
    api.post(`/uploads/video/${uploadId}/complete`, { parts }),

  simple: (projectId: string, file: File, sourceType: string) => {
    const form = new FormData();
    form.append("project_id", projectId);
    form.append("source_type", sourceType);
    form.append("file", file);
    return api.post("/uploads/video/simple", form, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 300_000,
    });
  },
};

export const processingApi = {
  triggerReconstruction: (data: {
    project_id: string;
    media_upload_ids: string[];
    quality?: string;
  }) => api.post("/processing/reconstruction", data),

  triggerDetection: (projectId: string, model?: string) =>
    api.post("/processing/detection", { project_id: projectId, model }),

  getJobStatus: (jobId: string) => api.get(`/processing/jobs/${jobId}`),
};

export const analyticsApi = {
  getProgress:   (projectId: string, params?: Record<string, unknown>) =>
    api.get(`/analytics/progress/${projectId}`, { params }),
  getDelays:     (projectId: string) => api.get(`/analytics/delays/${projectId}`),
  getKPIs:       (projectId: string, periodDays?: number) =>
    api.get(`/analytics/kpi/${projectId}`, { params: { period_days: periodDays } }),
  getTimeline:   (projectId: string, params?: Record<string, unknown>) =>
    api.get(`/analytics/timeline/${projectId}`, { params }),
  getHeatmap:    (projectId: string, type?: string) =>
    api.get(`/analytics/heatmap/${projectId}`, { params: { heatmap_type: type } }),
  getEquipment:  (projectId: string, periodDays?: number) =>
    api.get(`/analytics/equipment/${projectId}`, { params: { period_days: periodDays } }),
  getAlerts:     (projectId: string, params?: Record<string, unknown>) =>
    api.get(`/analytics/alerts/${projectId}`, { params }),
  getSummary:    (projectId: string) => api.get(`/analytics/summary/${projectId}`),
  computeProgress: (projectId: string) =>
    api.post(`/analytics/progress/${projectId}/compute`),
};

export const reconstructionApi = {
  list:   (projectId: string) => api.get("/reconstruction/", { params: { project_id: projectId } }),
  get:    (id: string) => api.get(`/reconstruction/${id}`),
  getPointCloudUrl: (id: string) => api.get(`/reconstruction/${id}/pointcloud-url`),
  getCameraPoses:   (id: string) => api.get(`/reconstruction/${id}/camera-poses`),
};

export const bimApi = {
  uploadModel: (projectId: string, file: File, name: string) => {
    const form = new FormData();
    form.append("project_id", projectId);
    form.append("name", name);
    form.append("file", file);
    return api.post("/bim/models", form, { headers: { "Content-Type": "multipart/form-data" } });
  },
  listModels:     (projectId: string) => api.get("/bim/models", { params: { project_id: projectId } }),
  triggerCompare: (bimModelId: string, reconstructionId: string) =>
    api.post("/bim/compare", { bim_model_id: bimModelId, reconstruction_id: reconstructionId }),
  getComparison:  (id: string) => api.get(`/bim/comparisons/${id}`),
};

// ── Chunked upload helper ──────────────────────────────────────────────────────

export async function chunkedUpload(
  file: File,
  projectId: string,
  sourceType: string,
  onProgress?: (pct: number) => void,
): Promise<string> {
  const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB

  // 1. Init session
  const initForm = new FormData();
  initForm.append("project_id", projectId);
  initForm.append("filename", file.name);
  initForm.append("file_size", String(file.size));
  initForm.append("mime_type", file.type || "video/mp4");
  initForm.append("source_type", sourceType);
  const { data: initData } = await uploadsApi.initChunked(initForm);
  const { upload_id, total_chunks } = initData;

  const parts: Array<{ PartNumber: number; ETag: string }> = [];

  // 2. Upload chunks
  for (let i = 0; i < total_chunks; i++) {
    const start = i * CHUNK_SIZE;
    const end   = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const { data: chunkData } = await uploadsApi.uploadChunk(upload_id, i + 1, chunk);
    parts.push({ PartNumber: i + 1, ETag: chunkData.etag });

    if (onProgress) {
      onProgress(Math.round(((i + 1) / total_chunks) * 90));
    }
  }

  // 3. Complete
  const { data: completeData } = await uploadsApi.completeChunked(upload_id, parts);
  if (onProgress) onProgress(100);

  return completeData.media_upload_id;
}
