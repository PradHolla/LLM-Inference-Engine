import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/ui/",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/labbench": "http://127.0.0.1:8081",
      "/v1": "http://127.0.0.1:8081",
      "/health": "http://127.0.0.1:8081",
    },
  },
  build: { outDir: "../ui", emptyOutDir: true },
});
