import { useState, useEffect } from 'react'
import './App.css'
import { apiService, HealthStatus } from './services/api'

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  
  // Chat states
  const [message, setMessage] = useState<string>('')
  const [chatResponse, setChatResponse] = useState<string | null>(null)
  const [chatLoading, setChatLoading] = useState<boolean>(false)
  const [chatError, setChatError] = useState<string | null>(null)

  useEffect(() => {
    checkSystemHealth()
  }, [])

  const checkSystemHealth = async () => {
    try {
      setLoading(true)
      const status = await apiService.checkHealth()
      setHealth(status)
      setError(null)
    } catch (err) {
      setError('Failed to connect to backend. Is Docker running?')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleChat = async () => {
    if (!message.trim()) return

    try {
      setChatLoading(true)
      setChatError(null)
      const result = await apiService.chatWithAI(message)
      setChatResponse(result.response)
    } catch (err: any) {
      setChatError(err.message || 'Failed to get AI response')
    } finally {
      setChatLoading(false)
    }
  }

  const isSystemHealthy = health?.status === 'healthy' && 
                          health?.database === 'connected' && 
                          health?.redis === 'connected'

  return (
    <div className="app-container">
      <header className="header">
        <h1>IES_EV System</h1>
        <p>Phase 0: Infrastructure & Foundation</p>
      </header>

      {/* Status Section */}
      <div className="card">
        <h2>System Status</h2>
        
        {loading ? (
          <div className="loading-spinner" style={{ margin: '2rem auto' }}></div>
        ) : error ? (
          <div className="status-value error" style={{ textAlign: 'center', padding: '1rem' }}>
            {error}
            <button onClick={checkSystemHealth} style={{ marginTop: '1rem', display: 'block', margin: '1rem auto' }}>
              Retry Connection
            </button>
          </div>
        ) : (
          <>
            {isSystemHealthy && (
              <div className="success-banner">
                <span>✅</span> Phase 0 Complete! System Operational
              </div>
            )}
            
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">Backend API</span>
                <span className={`status-value ${health?.status}`}>
                  {health?.status || 'Unknown'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">PostgreSQL</span>
                <span className={`status-value ${health?.database}`}>
                  {health?.database || 'Unknown'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Redis</span>
                <span className={`status-value ${health?.redis}`}>
                  {health?.redis || 'Unknown'}
                </span>
              </div>
            </div>
          </>
        )}
      </div>

      {/* AI Integration Section */}
      <div className="card">
        <h2>DeepSeek AI Integration</h2>
        <p style={{ color: '#94a3b8', marginBottom: '1rem' }}>
          Test the AI integration by sending a prompt below.
        </p>
        
        <div className="chat-interface">
          <div className="chat-input-group">
            <input 
              className="chat-input"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask about EV energy optimization..."
              onKeyPress={(e) => e.key === 'Enter' && handleChat()}
              disabled={chatLoading}
            />
            <button 
              onClick={handleChat} 
              disabled={chatLoading || !message.trim()}
            >
              {chatLoading ? <div className="loading-spinner" style={{ width: 16, height: 16 }}></div> : 'Send'}
            </button>
          </div>

          {chatError && (
            <div style={{ color: '#f87171', marginTop: '0.5rem' }}>
              Error: {chatError}
            </div>
          )}

          {chatResponse && (
            <div className="response-area">
              <strong>AI Response:</strong>
              <p style={{ marginTop: '0.5rem', whiteSpace: 'pre-wrap' }}>{chatResponse}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
