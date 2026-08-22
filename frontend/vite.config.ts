import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `vite dev` the backend runs separately on :8000, so proxy the API to
// it. In production the SPA is served by FastAPI from the same origin.
//
// One entry, because every endpoint now lives under /api. The old list named
// each top-level path, which meant it both shadowed the SPA's own /plates
// route in dev and quietly forgot /faces and /sightings.
const proxy = {
  "/api": { target: "http://localhost:8000", changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
});
