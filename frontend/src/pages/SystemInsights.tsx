import { useState, useEffect, useRef } from 'react'

const logEntries = [
  { time: '14:20:01', level: 'INFO', color: 'text-green-400', msg: 'Batch inference successful. Confidence score: 0.992' },
  { time: '14:20:03', level: 'INFO', color: 'text-green-400', msg: 'Feature vector extraction completed in 2.1ms' },
  { time: '14:20:05', level: 'WARN', color: 'text-yellow-400', msg: 'Physics Fallback triggered for node: AX-409 (Confidence < 0.85)' },
  { time: '14:20:08', level: 'INFO', color: 'text-green-400', msg: 'Re-aligned spatial weights for predicted route segments' },
  { time: '14:20:12', level: 'HYBR', color: 'text-blue-400', msg: 'Hybrid resolution merging physics + ML outputs' },
  { time: '14:20:15', level: 'INFO', color: 'text-green-400', msg: 'Global sync achieved. Drift: 0.0002ms' },
  { time: '14:20:18', level: 'INFO', color: 'text-green-400', msg: 'Awaiting next telemetry burst...' },
]

const diagnostics = [
  { label: 'Inference Temp', sub: 'Node cluster thermal', value: '42.5°C', color: 'text-orange-500' },
  { label: 'Memory Buffer', sub: 'VRAM allocation', value: '4.2 GB', color: 'text-brand-primary' },
  { label: 'Queue Depth', sub: 'Async tasks', value: '0 ms wait', color: 'text-surface-900' },
]

export default function SystemInsights() {
  const [logs, setLogs] = useState(logEntries)
  const terminalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date()
      const ts = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`
      const newLog = {
        time: ts,
        level: 'INFO',
        color: 'text-green-400',
        msg: `Heartbeat OK. Latency: ${(10 + Math.random() * 5).toFixed(1)}ms`,
      }
      setLogs((prev) => [...prev.slice(-20), newLog])
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full overflow-y-auto">

      {/* ───── Hybrid Prediction Engine Hero ───── */}
      <div className="grid grid-cols-12 gap-6 stagger-children">
        <div className="col-span-12 lg:col-span-8 bg-surface-900 rounded-2xl p-8 text-white relative overflow-hidden glow-ice">
          <div className="absolute top-0 right-0 w-64 h-64 bg-brand-primary/10 blur-[100px] -mr-32 -mt-32" />
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-2xl font-headline font-bold mb-1">Hybrid Prediction Engine</h3>
                <p className="text-white/40 text-sm">Real-time inference distribution and weighting</p>
              </div>
              <div className="px-3 py-1 bg-accent-success/20 border border-accent-success/30 rounded-full">
                <span className="text-[10px] font-mono tracking-widest uppercase text-accent-success">Status: Optimal</span>
              </div>
            </div>

            {/* Inference Loop */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 py-6 border-y border-white/10">
              {[
                { icon: 'database', label: 'Feature Extraction' },
                { icon: 'hub', label: 'ML Ensemble' },
                { icon: 'verified_user', label: 'Confidence Eval' },
                { icon: 'restart_alt', label: 'Physics Fallback' },
              ].map((step, i) => (
                <div key={step.label} className="flex items-center gap-4">
                  <div className="flex flex-col items-center gap-3 flex-1 text-center">
                    <div className="w-12 h-12 rounded-lg border border-brand-primary/40 flex items-center justify-center bg-brand-primary/10">
                      <span className="material-symbols-outlined text-brand-primary">{step.icon}</span>
                    </div>
                    <span className="text-xs font-mono text-white/40">{step.label}</span>
                  </div>
                  {i < 3 && <span className="material-symbols-outlined text-white/20 hidden md:block">chevron_right</span>}
                </div>
              ))}
            </div>

            {/* Distribution Bars */}
            <div className="mt-8 space-y-4">
              {[
                { label: 'ML Prediction', pct: 75, color: 'bg-brand-primary' },
                { label: 'Hybrid Logic', pct: 19, color: 'bg-blue-400' },
                { label: 'Physics Engine', pct: 6, color: 'bg-white/50' },
              ].map((b) => (
                <div key={b.label}>
                  <div className="flex justify-between text-xs font-mono uppercase tracking-tighter mb-1">
                    <span className={b.color === 'bg-brand-primary' ? 'text-brand-primary' : b.color === 'bg-blue-400' ? 'text-blue-400' : 'text-white/50'}>{b.label}</span>
                    <span>{b.pct}%</span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${b.color} transition-all duration-1000`} style={{ width: `${b.pct}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Latency & Throughput */}
        <div className="col-span-12 lg:col-span-4 grid grid-rows-2 gap-6">
          <div className="glass-ivory rounded-2xl p-6 flex flex-col justify-between card-hover">
            <div className="flex items-center justify-between">
              <span className="font-bold text-surface-900">Inference Latency</span>
              <span className="material-symbols-outlined text-brand-primary">speed</span>
            </div>
            <div>
              <div className="text-4xl font-headline font-bold text-surface-900">12.4<span className="text-lg font-normal text-surface-800/40 ml-1">ms</span></div>
              <p className="text-xs text-accent-success mt-1 flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">trending_down</span>-0.8ms vs baseline
              </p>
            </div>
            <div className="h-12 w-full flex items-end gap-1">
              {[60, 70, 50, 90, 100].map((h, i) => (
                <div
                  key={i}
                  className={`flex-1 rounded-t-sm transition-all ${i >= 3 ? 'bg-brand-primary' : 'bg-surface-200'}`}
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
          </div>
          <div className="glass-ivory rounded-2xl p-6 flex flex-col justify-between card-hover">
            <div className="flex items-center justify-between">
              <span className="font-bold text-surface-900">Throughput</span>
              <span className="material-symbols-outlined text-brand-primary">dns</span>
            </div>
            <div>
              <div className="text-4xl font-headline font-bold text-surface-900">1.2k<span className="text-lg font-normal text-surface-800/40 ml-1">req/s</span></div>
              <p className="text-xs text-surface-800/40 mt-1">Load distribution: Normal</p>
            </div>
            <div className="flex gap-2">
              {[true, true, false, false].map((active, i) => (
                <div key={i} className={`h-1.5 flex-1 rounded-full ${active ? 'bg-brand-primary' : 'bg-brand-primary/15'}`} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ───── Terminal + Diagnostics ───── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Terminal */}
        <div className="md:col-span-2 bg-surface-900 rounded-2xl overflow-hidden flex flex-col h-[400px] glow-ice">
          <div className="px-4 py-2 bg-surface-800/50 border-b border-white/5 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
              </div>
              <span className="text-[10px] font-mono text-white/30 uppercase ml-4">Terminal: engine_runtime_logs</span>
            </div>
            <span className="text-[10px] font-mono text-accent-success animate-pulse">LIVE FEED</span>
          </div>
          <div ref={terminalRef} className="p-6 font-mono text-xs text-white/70 overflow-y-auto space-y-2 flex-1">
            {logs.map((log, i) => (
              <div key={i} className={`flex gap-4 ${i === logs.length - 1 ? 'opacity-50' : ''}`}>
                <span className="text-brand-primary/50">[{log.time}]</span>
                <span className={log.color}>{log.level}:</span>
                <span>{log.msg}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Diagnostics */}
        <div className="glass-ivory rounded-2xl p-6 flex flex-col h-[400px] card-hover">
          <h4 className="font-headline font-bold text-surface-900 mb-6">Internal Diagnostics</h4>
          <div className="space-y-6 flex-1 overflow-y-auto pr-2">
            {diagnostics.map((d) => (
              <div key={d.label} className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-bold text-surface-900">{d.label}</p>
                  <p className="text-[10px] text-surface-800/40">{d.sub}</p>
                </div>
                <span className={`text-sm font-mono font-bold ${d.color}`}>{d.value}</span>
              </div>
            ))}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold text-surface-900">Model Version</p>
                <p className="text-[10px] text-surface-800/40">v4.2.1-stable</p>
              </div>
              <span className="px-2 py-0.5 bg-ice text-[10px] font-mono rounded text-brand-primary font-bold">LATEST</span>
            </div>
          </div>
          <button className="mt-6 w-full bg-surface-900 text-white py-3 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-surface-800 transition-colors">
            <span className="material-symbols-outlined text-[18px]">terminal</span>
            Run Full Diagnostic
          </button>
        </div>
      </div>

      {/* ───── Neural Pathway ───── */}
      <div className="relative glass-ivory rounded-2xl p-8 overflow-hidden h-48 card-hover">
        <div className="relative z-10 flex flex-col h-full justify-between">
          <div className="flex justify-between items-start">
            <div>
              <h4 className="font-headline font-bold text-surface-900">Neural Pathway Visualization</h4>
              <p className="text-sm text-surface-800/40">Active predictive clusters mapping</p>
            </div>
            <div className="flex gap-2">
              <div className="w-2 h-2 rounded-full bg-brand-primary" />
              <div className="w-2 h-2 rounded-full bg-brand-primary/40" />
              <div className="w-2 h-2 rounded-full bg-brand-primary/20" />
            </div>
          </div>
          <div className="flex items-center justify-around overflow-hidden">
            {[
              { icon: 'share', pulse: true },
              { icon: 'psychology', pulse: false },
              { icon: 'rocket_launch', pulse: true },
            ].map((node, i) => (
              <div key={node.icon} className="flex items-center">
                <div className={`w-24 h-24 rounded-full border border-brand-primary/20 flex items-center justify-center ${node.pulse ? 'animate-pulse' : ''}`}>
                  <div className="w-16 h-16 rounded-full bg-ice/50 flex items-center justify-center">
                    <span className="material-symbols-outlined text-brand-primary">{node.icon}</span>
                  </div>
                </div>
                {i < 2 && <div className="w-16 lg:w-32 border-t border-dashed border-brand-primary/20 mx-4" />}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
