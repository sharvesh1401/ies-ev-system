import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ocm': {
        target: 'https://api.openchargemap.io',
        changeOrigin: true,
        secure: true,
        rewrite: (path: string) => path.replace(/^\/ocm/, '/v3'),
      }
    }
  }
})
