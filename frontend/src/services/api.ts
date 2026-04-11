import axios from 'axios';

// Backend API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance — 90s timeout to survive cold-start wakeups on free hosting
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 90000,
});

// ── Cold-start wakeup detection ───────────────────────────────────────────────
// If any backend request takes > 3s, broadcast a "waking up" event so the UI
// can show a friendly banner instead of an unexplained spinner.

let _pendingCount = 0
let _slowTimer: ReturnType<typeof setTimeout> | null = null

function _onRequestStart() {
  _pendingCount++
  if (_pendingCount === 1 && !_slowTimer) {
    _slowTimer = setTimeout(() => {
      window.dispatchEvent(new CustomEvent('backend:waking'))
    }, 3000)
  }
}

function _onRequestEnd() {
  _pendingCount = Math.max(0, _pendingCount - 1)
  if (_pendingCount === 0) {
    if (_slowTimer) { clearTimeout(_slowTimer); _slowTimer = null }
    window.dispatchEvent(new CustomEvent('backend:ready'))
  }
}

api.interceptors.request.use(
  (config) => { _onRequestStart(); return config },
  (error) => { _onRequestEnd(); return Promise.reject(error) }
)

api.interceptors.response.use(
  (response) => { _onRequestEnd(); return response.data },
  (error) => { _onRequestEnd(); return Promise.reject(error) }
)

export default api;
