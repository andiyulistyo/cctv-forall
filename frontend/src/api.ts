// API client. In production the SPA is same-origin with the backend, so the
// base is empty. Override with VITE_API_BASE if needed.
const BASE = (import.meta as any).env?.VITE_API_BASE ?? "";

export type SourceType = "youtube" | "rtsp" | "rtmp" | "hls" | "http" | "file";
export const DETECTION_CLASSES = ["person", "car", "motorcycle", "truck", "bus"] as const;
export type DetectionClass = (typeof DETECTION_CLASSES)[number];

export interface Source {
  id: number;
  name: string;
  type: SourceType;
  url: string;
  enabled_classes: string[];
  alpr_enabled: boolean;
  face_enabled: boolean;
  line: { a: number[]; b: number[] } | null;
  direction_labels: { in: string; out: string };
  status: string;
  status_message: string | null;
  created_at: string;
}

export interface SourceListResponse {
  sources: Source[];
  total: number;
  active: number;
}

export interface CountBucket {
  class_name: string;
  direction: string;
  count: number;
}
export interface CountsResponse {
  source_id: number | null;
  buckets: CountBucket[];
  live: Record<string, { in: number; out: number }> | null;
}

export interface Plate {
  id: number;
  source_id: number;
  track_id: number;
  vehicle_class: string;
  plate_text: string;
  confidence: number;
  has_image: boolean;
  timestamp: string;
}

export interface EnrolledFace {
  id: number;
  name: string;
  has_image: boolean;
  created_at: string;
}

export interface Sighting {
  id: number;
  source_id: number;
  name: string | null;
  similarity: number;
  has_image: boolean;
  timestamp: string;
}

function getToken(): string | null {
  return localStorage.getItem("token");
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as any) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";

  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (res.status === 401) {
    localStorage.removeItem("token");
    if (!location.pathname.startsWith("/login")) location.href = "/login";
    throw new ApiError(401, "Unauthorized");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  async login(username: string, password: string) {
    const r = await request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    localStorage.setItem("token", r.access_token);
    return r;
  },
  logout() {
    localStorage.removeItem("token");
  },
  listSources: () => request<SourceListResponse>("/sources"),
  getSource: (id: number) => request<Source>(`/sources/${id}`),
  createSource: (body: Partial<Source>) =>
    request<Source>("/sources", { method: "POST", body: JSON.stringify(body) }),
  updateSource: (id: number, body: Partial<Source>) =>
    request<Source>(`/sources/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteSource: (id: number) => request<void>(`/sources/${id}`, { method: "DELETE" }),
  startSource: (id: number) => request<Source>(`/sources/${id}/start`, { method: "POST" }),
  stopSource: (id: number) => request<Source>(`/sources/${id}/stop`, { method: "POST" }),
  setLine: (id: number, a: number[], b: number[], labels?: { in: string; out: string }) =>
    request<Source>(`/sources/${id}/line`, {
      method: "PUT",
      body: JSON.stringify({ line: { a, b }, direction_labels: labels }),
    }),
  getCounts: (sourceId?: number) =>
    request<CountsResponse>(`/counts${sourceId ? `?source=${sourceId}` : ""}`),
  listPlates: (sourceId?: number, limit = 100) =>
    request<Plate[]>(`/plates?limit=${limit}${sourceId ? `&source=${sourceId}` : ""}`),

  // --- Faces ---
  listFaces: () => request<EnrolledFace[]>("/faces"),
  deleteFace: (id: number) => request<void>(`/faces/${id}`, { method: "DELETE" }),
  async enrollFace(name: string, file: File): Promise<EnrolledFace> {
    const fd = new FormData();
    fd.append("name", name);
    fd.append("image", file);
    const token = getToken();
    const res = await fetch(`${BASE}/faces/enroll`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: fd, // browser sets multipart Content-Type + boundary
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        /* ignore */
      }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as EnrolledFace;
  },
  listSightings: (sourceId?: number, limit = 100) =>
    request<Sighting[]>(`/sightings?limit=${limit}${sourceId ? `&source=${sourceId}` : ""}`),

  // Media URLs (token via query string for <img>/stream tags)
  streamUrl: (id: number) => `${BASE}/streams/${id}?token=${getToken() ?? ""}`,
  plateImageUrl: (id: number) => `${BASE}/plates/${id}/image?token=${getToken() ?? ""}`,
  faceImageUrl: (id: number) => `${BASE}/faces/${id}/image?token=${getToken() ?? ""}`,
  sightingImageUrl: (id: number) => `${BASE}/sightings/${id}/image?token=${getToken() ?? ""}`,

  async snapshotBlobUrl(id: number): Promise<string> {
    const token = getToken();
    const res = await fetch(`${BASE}/sources/${id}/snapshot`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, "snapshot failed");
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },
};

export { ApiError };
