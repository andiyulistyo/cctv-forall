import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `vite dev` the backend runs separately on :8000, so proxy the API
// paths to it. In production the SPA is served by FastAPI from the same origin.
const proxyTargets = ["/auth", "/sources", "/streams", "/counts", "/plates", "/api"];
const proxy = Object.fromEntries(
  proxyTargets.map((p) => [p, { target: "http://localhost:8000", changeOrigin: true }])
);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
});
