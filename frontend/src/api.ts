// API client. In production the SPA is same-origin with the backend, so the
// origin is empty. Override with VITE_API_BASE if the backend lives elsewhere.
const ORIGIN = (import.meta as any).env?.VITE_API_BASE ?? "";
// Every endpoint sits under /api so that page routes like /plates and /faces
// stay the SPA's -- they used to be claimed by the API, which made those pages
// impossible to open or refresh directly.
const BASE = `${ORIGIN}/api`;

export type SourceType = "youtube" | "rtsp" | "rtmp" | "hls" | "http" | "file";
export const DETECTION_CLASSES = ["person", "car", "motorcycle", "truck", "bus"] as const;
export type DetectionClass = (typeof DETECTION_CLASSES)[number];
/** The classes that can carry a plate -- mirrors models.VEHICLE_CLASSES. */
export const VEHICLE_CLASSES = ["car", "motorcycle", "truck", "bus"] as const;
export const CLASS_LABELS: Record<string, string> = {
  person: "Orang",
  car: "Mobil",
  motorcycle: "Motor",
  truck: "Truk",
  bus: "Bus",
};

export interface Source {
  id: number;
  name: string;
  type: SourceType;
  url: string;
  enabled_classes: string[];
  alpr_enabled: boolean;
  face_enabled: boolean;
  /** Start this source again by itself when the backend starts (after a reboot). */
  auto_start: boolean;
  line: { a: number[]; b: number[] } | null;
  /** Where plates are read. Two opposite corners, normalized 0..1. null = anywhere. */
  alpr_zone: { a: number[]; b: number[] } | null;
  direction_labels: { in: string; out: string };
  status: string;
  status_message: string | null;
  created_at: string;
  stats: SourceStats | null;
}

/** Live throughput of a running worker, for diagnosing a choppy stream. */
export interface SourceStats {
  capture_fps?: number;   // frames arriving from the camera/stream
  processed_fps?: number; // frames the worker got to look at
  detect_fps?: number;    // frames that ran through YOLO
  publish_fps?: number;   // frames sent to the browser
  dropped_fps?: number;   // frames discarded because we were behind
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
  /** What OCR read. A review never overwrites this -- see `corrected_text`. */
  plate_text: string;
  confidence: number;
  has_image: boolean;
  /** A full frame of the moment was kept, with the vehicle boxed. */
  has_frame: boolean;
  /**
   * What a person said the plate really is.
   *
   * `null` -- nobody has looked at this read yet.
   * `""`   -- somebody looked and the plate cannot be read from the crop.
   * a string -- the plate, whether or not OCR happened to agree.
   *
   * Kept apart from `plate_text` on purpose: the pair (prediction, truth) is
   * the whole value of a reviewed read, and merging them would leave the
   * screen looking right while destroying any way to measure the reader.
   */
  corrected_text: string | null;
  reviewed_at: string | null;
  reviewed_by: string | null;
  timestamp: string;
}

/** Where a read stands with a human reviewer. Mirrors api/plates.ReviewFilter. */
export const REVIEW_FILTERS = [
  "pending",
  "reviewed",
  "correct",
  "wrong",
  "illegible",
] as const;
export type ReviewFilter = (typeof REVIEW_FILTERS)[number];

/** What a row's review says, as one word the UI can badge. */
export type ReviewState = "pending" | "correct" | "wrong" | "illegible";

export function reviewState(p: Plate): ReviewState {
  if (p.reviewed_at === null) return "pending";
  if (p.corrected_text === "") return "illegible";
  return p.corrected_text === p.plate_text ? "correct" : "wrong";
}

/** The best answer available for a read: the human's if there is one. */
export function plateAnswer(p: Plate): string {
  return p.corrected_text ?? p.plate_text;
}

export interface PlateListResponse {
  plates: Plate[];
  /** Total matching the filters, ignoring limit/offset -- drives the page count. */
  total: number;
  /** Reads per vehicle class under every filter *except* the class one, so the
   *  class filter can show what each choice would give you. */
  class_counts: Record<string, number>;
}

/** Columns the plate list can be ordered by. "source" sorts on the source name. */
export const PLATE_SORT_KEYS = [
  "timestamp",
  "confidence",
  "plate_text",
  "vehicle_class",
  "source",
] as const;
export type PlateSortKey = (typeof PLATE_SORT_KEYS)[number];

export interface PlateQuery {
  source?: number;
  vehicle_class?: string;
  /** Part of the plate text. Spaces and dashes are ignored by the server. */
  q?: string;
  min_confidence?: number;
  review?: ReviewFilter;
  /** ISO timestamps. Only reads at or after `from`, and at or before `to`. */
  from?: string;
  to?: string;
  sort?: PlateSortKey;
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

/** Drops empty values so an unset filter never reaches the server at all --
 *  `?source=` would be a validation error, not "every source". */
function plateQueryString(query: PlateQuery): string {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  });
  return params.toString();
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
  /** A full frame of the moment was kept, with the face boxed. */
  has_frame: boolean;
  timestamp: string;
}

/** How the reviews have gone, and how much is left. */
export interface ReviewProgress {
  total: number;
  reviewed: number;
  pending: number;
  /** Reviewed, legible, and OCR had it right. */
  correct: number;
  /** Reviewed, legible, and OCR had it wrong -- the error set. */
  wrong: number;
  /** Reviewed and not readable by anyone; excluded from every score below. */
  illegible: number;
}

/** How the reader scores against the labels. */
export interface OcrMetrics {
  reads: number;
  exact: number;
  exact_rate: number;
  blank: number;
  blank_rate: number;
  chars: number;
  errors: number;
  /** Character error rate: edits per character of ground truth. */
  cer: number;
  confusions: { pair: string; count: number }[];
}

export interface SourceMetrics extends OcrMetrics {
  source_id: number;
  name: string;
}

export interface DatasetSummary {
  progress: ReviewProgress;
  /** null until something has been reviewed -- not a row of zeroes, which
   *  would read as "the reader gets nothing right". */
  metrics: OcrMetrics | null;
  per_source: SourceMetrics[];
  /** Reviewed reads left out of the scoring, and why. */
  skipped: { reason: string; count: number }[];
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
  /** Pass null to clear the zone and go back to reading anywhere in the frame. */
  setAlprZone: (id: number, zone: { a: number[]; b: number[] } | null) =>
    request<Source>(`/sources/${id}/alpr-zone`, {
      method: "PUT",
      body: JSON.stringify({ zone }),
    }),
  getCounts: (sourceId?: number) =>
    request<CountsResponse>(`/counts${sourceId ? `?source=${sourceId}` : ""}`),
  listPlates: (query: PlateQuery = {}) =>
    request<PlateListResponse>(`/plates?${plateQueryString(query)}`),
  /** Record what a person says this plate is. "" means "looked, unreadable". */
  reviewPlate: (id: number, plateText: string) =>
    request<Plate>(`/plates/${id}/review`, {
      method: "PUT",
      body: JSON.stringify({ plate_text: plateText }),
    }),
  /** Undo a review, putting the read back in the pending queue. */
  unreviewPlate: (id: number) => request<Plate>(`/plates/${id}/review`, { method: "DELETE" }),

  // --- OCR accuracy / dataset ---
  getDataset: () => request<DatasetSummary>("/dataset"),

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
  /** Navigated to directly, so the browser owns the download and its progress
   *  bar -- hence the token in the query string, as for the images. */
  datasetExportUrl: (evalShare: number) =>
    `${BASE}/dataset/export?token=${getToken() ?? ""}&eval_share=${evalShare}`,
  plateImageUrl: (id: number) => `${BASE}/plates/${id}/image?token=${getToken() ?? ""}`,
  plateFrameUrl: (id: number) => `${BASE}/plates/${id}/frame?token=${getToken() ?? ""}`,
  faceImageUrl: (id: number) => `${BASE}/faces/${id}/image?token=${getToken() ?? ""}`,
  sightingImageUrl: (id: number) => `${BASE}/sightings/${id}/image?token=${getToken() ?? ""}`,
  sightingFrameUrl: (id: number) => `${BASE}/sightings/${id}/frame?token=${getToken() ?? ""}`,

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
