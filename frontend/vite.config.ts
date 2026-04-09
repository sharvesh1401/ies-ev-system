import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// BACKEND_URL is a server-side env var (no VITE_ prefix) used by the Vite proxy.
// When running inside Docker Compose it is set to http://backend:8000 (service name).
// When running locally it falls back to http://localhost:8000.
const BACKEND_PROXY_TARGET = process.env.BACKEND_URL || 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': {
        target: BACKEND_PROXY_TARGET,
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
