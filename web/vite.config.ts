import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: {
      "@core": fileURLToPath(new URL("../core/src", import.meta.url)),
    },
  },
  build: {
    target: "es2020",
    outDir: "dist",
  },
});
