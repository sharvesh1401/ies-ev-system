import { useState, useEffect, useRef } from 'react'
import { useVehicle } from '../contexts/VehicleContext'
import VehicleSelector from '../components/VehicleSelector'

/* ── Live log generator ── */
const LOG_POOL = [
  { level: 'INFO', color: 'text-neon-green', bg: 'bg-neon-green/10', msgs: [
    'Batch inference successful. Confidence: 0.992',
    'Feature vector extraction completed in 2.1ms',
    'Re-aligned spatial weights for predicted route',
    'Global sync achieved. Drift: 0.0002ms',
    'Awaiting next telemetry burst...',
    'Energy model recalibrated successfully',
    'Route optimization complete — 3 waypoints',
  ]},
  { level: 'WARN', color: 'text-accent-warning', bg: 'bg-accent-warning/10', msgs: [
    'Physics Fallback triggered (Confidence < 0.85)',
    'Memory allocation approaching 85% threshold',
    'Network latency spike detected: 45ms',
  ]},
  { level: 'HYBR', color: 'text-neon-blue', bg: 'bg-neon-blue/10', msgs: [
    'Hybrid resolution merging physics + ML outputs',
    'Weighted ensemble: ML=75%, Physics=25%',
    'Cross-validation score: 0.967',
  ]},
]

function generateLog() {
  const now = new Date()
  const ts = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`
  const pool = LOG_POOL[Math.random() < 0.7 ? 0 : Math.random() < 0.5 ? 1 : 2]
  return {
    time: ts,
    level: pool.level,
    color: pool.color,
    bg: pool.bg,
    msg: pool.msgs[Math.floor(Math.random() * pool.msgs.length)],
  }
}

/* ── Latency sparkline data ── */
function useSparkline(count: number, baseVal: number, variance: number, intervalMs: number) {
  const [data, setData] = useState(() => Array.from({ length: count }, () => baseVal + (Math.random() - 0.5) * variance))
  useEffect(() => {
    const id = setInterval(() => {
      setData((prev) => [...prev.slice(1), baseVal + (Math.random() - 0.5) * variance])
    }, intervalMs)
    return () => clearInterval(id)
  }, [count, baseVal, variance, intervalMs])
  return data
}

export default function SystemInsights() {
  const { vehicle } = useVehicle()
  const terminalRef = useRef<HTMLDivElement>(null)
  const [logs, setLogs] = useState(() => Array.from({ length: 12 }, () => generateLog()))

  // Live ML distribution
  const [mlPct, setMlPct] = useState(75)
  const [hybridPct, setHybridPct] = useState(19)
  const [physicsPct, setPhysicsPct] = useState(6)

  // Live metrics
  const latencyData = useSparkline(40, 12, 8, 800)
  const throughputData = useSparkline(30, 1200, 400, 1200)
  const cpuData = useSparkline(25, 42, 20, 1000)
  const memData = useSparkline(25, 4.2, 0.8, 1500)

  const currentLatency = latencyData[latencyData.length - 1]
  const currentThroughput = Math.round(throughputData[throughputData.length - 1])
  const currentCPU = cpuData[cpuData.length - 1]
  const currentMem = memData[memData.length - 1]

  // Log feed
  useEffect(() => {
    const id = setInterval(() => {
      setLogs((prev) => [...prev.slice(-40), generateLog()])
    }, 1500)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight
  }, [logs])

  // Fluctuate ML distribution — adapted per vehicle
  useEffect(() => {
    const id = setInterval(() => {
      // Different vehicles use different model ratios
      let mlBase = 75, physBase = 5
      if (vehicle.id === 'model-t-cargo') {
        mlBase = 55; physBase = 18 // cargo relies more on physics
      } else if (vehicle.id === 'model-s-commuter') {
        mlBase = 62; physBase = 12 // degraded uses more physics fallback
      }
      const ml = mlBase + (Math.random() - 0.5) * 10
      const phys = physBase + Math.random() * 6
      const hyb = 100 - ml - phys
      setMlPct(ml)
      setHybridPct(hyb)
      setPhysicsPct(phys)
    }, 4000)
    return () => clearInterval(id)
  }, [vehicle.id])

  // Sparkline SVG renderer - styled for Blade Runner theme
  function renderSparkline(data: number[], color: string, height = 40, width = 300) {
    const min = Math.min(...data) - 1
    const max = Math.max(...data) + 1
    const points = data.map((v, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - ((v - min) / (max - min)) * height
      return `${x},${y}`
    }).join(' ')
    
    // Convert hex color to rgba for gradient
    const tColor = color === '#00b4d8' ? 'rgba(0,180,216,' : 
                   color === '#00f5a0' ? 'rgba(0,245,160,' :
                   color === '#ff3e6c' ? 'rgba(255,62,108,' :
                   color === '#7b2ff7' ? 'rgba(123,47,247,' : 'rgba(255,255,255,';

    return (
      <svg width={width} height={height} className="w-full drop-shadow-[0_0_8px_currentColor]" style={{ color }} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <defs>
          <linearGradient id={`sg-${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={tColor + '0.4)'} />
            <stop offset="100%" stopColor={tColor + '0)'} />
          </linearGradient>
        </defs>
        <polygon points={`0,${height} ${points} ${width},${height}`} fill={`url(#sg-${color.replace('#','')})`} />
        <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }

  return (
    <div className="p-4 md:p-5 lg:p-6 space-y-3 md:space-y-4 max-w-7xl mx-auto w-full h-full overflow-y-auto no-scrollbar flex flex-col pt-4">

      {/* ═══ Top Section: Profile & Metrics ═══ */}
      <div className="grid grid-cols-12 gap-3 md:gap-4 shrink-0">
        
        {/* Profile Card (Ported from Profile.tsx) */}
        <div className="col-span-12 lg:col-span-3 glass-dark p-4 border border-neon-blue/20 relative overflow-hidden flex flex-col justify-between">
          <div className="mb-3">
            <VehicleSelector />
          </div>
          <div className="space-y-2 mt-auto text-[9px] font-mono text-surface-800/60 uppercase tracking-widest">
             <div className="flex justify-between items-center bg-surface-100/50 p-2 rounded">
                <span>Motor Efficiency</span>
                <span className="text-neon-blue font-bold">{(vehicle.specs.motor_efficiency * 100).toFixed(0)}%</span>
             </div>
             <div className="flex justify-between items-center bg-surface-100/50 p-2 rounded">
                <span>Drag Coefficient</span>
                <span className="text-neon-purple font-bold">{vehicle.specs.drag_coefficient} Cd</span>
             </div>
             <div className="flex justify-between items-center bg-surface-100/50 p-2 rounded">
                <span>Base Weight</span>
                <span className="text-neon-green font-bold">{vehicle.specs.mass_kg} kg</span>
             </div>
             <div className="flex justify-between items-center bg-surface-100/50 p-2 rounded">
                <span>Max Power</span>
                <span className="text-accent-warning font-bold">{vehicle.specs.max_power_kw} kW</span>
             </div>
             <div className="flex justify-between items-center bg-surface-100/50 p-2 rounded">
                <span>Battery Health</span>
                <span className={`font-bold ${vehicle.health.soh_percent >= 90 ? 'text-accent-success' : vehicle.health.soh_percent >= 80 ? 'text-accent-warning' : 'text-neon-red'}`}>{vehicle.health.soh_percent}% SoH</span>
             </div>
          </div>
        </div>

        {/* Hybrid Prediction Engine */}
        <div className="col-span-12 lg:col-span-5 glass-dark px-5 py-4 border border-neon-blue/20 relative overflow-hidden flex flex-col justify-between">
          <div className="absolute top-0 right-0 w-80 h-80 bg-neon-blue/10 rounded-full blur-[60px] -mr-40 -mt-40 pointer-events-none" />
          
          <div className="relative z-10 flex items-center justify-between mb-6">
            <div>
              <h3 className="text-xl font-headline font-bold text-surface-900 tracking-tight glow-neon">Hybrid Prediction Engine</h3>
              <p className="text-[9px] font-mono text-neon-blue/60 mt-0.5 uppercase tracking-widest">Real-time inference distribution</p>
            </div>
            <div className="px-3 py-1 bg-neon-green/10 border border-neon-green/30 text-neon-green  shadow-[0_0_15px_rgba(0,245,160,0.2)]">
              <span className="text-[10px] font-mono tracking-widest uppercase font-bold flex items-center gap-1.5 flex-row-reverse">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse" />
                Optimal
              </span>
            </div>
          </div>

          {/* Pipeline steps */}
          <div className="relative z-10 flex items-center justify-between py-3 border-y border-white/5 mb-4 space-x-2">
            <div className="absolute left-[10%] right-[10%] top-1/2 h-px bg-gradient-to-r from-neon-blue/0 via-neon-blue/40 to-neon-blue/0 -translate-y-1/2 z-0" />
            
            {[
              { icon: 'database', label: 'Feature Extract', color: 'text-neon-blue', border: 'border-neon-blue/40' },
              { icon: 'hub', label: 'ML Ensemble', color: 'text-neon-purple', border: 'border-neon-purple/40' },
              { icon: 'verified_user', label: 'Confidence', color: 'text-neon-green', border: 'border-neon-green/40' },
              { icon: 'sync', label: 'Fallback', color: 'text-surface-800/60', border: 'border-surface-800/40' },
            ].map((step) => (
              <div key={step.label} className="relative z-10 flex flex-col items-center gap-3 bg-brand-bg px-4 py-2">
                <div className={`w-12 h-12 rounded-full border ${step.border} flex items-center justify-center bg-surface-100/80 backdrop-blur shadow-[0_0_15px_currentColor] ${step.color}`}>
                  <span className="material-symbols-outlined text-[22px]">{step.icon}</span>
                </div>
                <span className={`text-[9px] font-mono tracking-widest uppercase ${step.color}`}>{step.label}</span>
              </div>
            ))}
          </div>

          {/* Distribution bars */}
          <div className="relative z-10 space-y-4">
            {[
              { label: 'ML Prediction', pct: mlPct, color: '#00b4d8' },
              { label: 'Hybrid Logic', pct: hybridPct, color: '#7b2ff7' },
              { label: 'Physics Engine', pct: physicsPct, color: '#94a3b8' },
            ].map((b) => (
              <div key={b.label} className="group">
                <div className="flex justify-between text-[10px] font-mono font-bold uppercase tracking-widest mb-1.5">
                  <span style={{ color: b.color }} className="drop-shadow-[0_0_5px_currentColor]">{b.label}</span>
                  <span className="tabular-nums" style={{ color: b.color }}>{b.pct.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 w-full bg-surface-200/50 rounded-full overflow-hidden shadow-inner flex">
                  <div className="h-full rounded-full transition-all duration-1000 shadow-[0_0_10px_currentColor] group-hover:brightness-125" style={{ width: `${b.pct}%`, background: b.color, boxShadow: `0 0 10px ${b.color}` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Live metrics cards */}
        <div className="col-span-12 lg:col-span-4 grid grid-rows-2 gap-3 md:gap-4">
          <div className="glass-dark  p-5 border border-neon-blue/10 relative overflow-hidden flex flex-col justify-between group hover:border-neon-blue/30 transition-colors">
            <div className="absolute -top-10 -right-10 w-24 h-24 bg-neon-blue/10 rounded-full blur-[30px] group-hover:bg-neon-blue/20 transition-all" />
            <div className="flex items-center justify-between mb-4 relative z-10">
              <span className="text-[12px] font-mono text-neon-blue uppercase tracking-widest font-bold drop-shadow-[0_0_5px_#00b4d8]">Inference Latency</span>
              <span className="material-symbols-outlined text-neon-blue/50 text-xl">speed</span>
            </div>
            <div className="text-4xl font-headline font-bold text-surface-900 tabular-nums tracking-tighter relative z-10">
              {currentLatency.toFixed(1)}<span className="text-sm font-mono text-neon-blue/60 ml-1.5 tracking-normal">ms</span>
            </div>
            <div className="mt-4 relative z-10 opacity-80 group-hover:opacity-100 transition-opacity">
              {renderSparkline(latencyData, '#00b4d8', 45)}
            </div>
            <div className="absolute bottom-6 right-6 flex gap-1">
               <span className="w-1 h-3 rounded-full bg-neon-blue/40 animate-pulse" style={{animationDelay: '0ms'}}/>
               <span className="w-1 h-2 rounded-full bg-neon-blue/20 animate-pulse" style={{animationDelay: '150ms'}}/>
               <span className="w-1 h-4 rounded-full bg-neon-blue/60 animate-pulse" style={{animationDelay: '300ms'}}/>
            </div>
          </div>
          <div className="glass-dark  p-5 border border-neon-purple/10 relative overflow-hidden flex flex-col justify-between group hover:border-neon-purple/30 transition-colors">
            <div className="absolute -top-10 -right-10 w-24 h-24 bg-neon-purple/10 rounded-full blur-[30px] group-hover:bg-neon-purple/20 transition-all" />
            <div className="flex items-center justify-between mb-4 relative z-10">
              <span className="text-[12px] font-mono text-neon-purple uppercase tracking-widest font-bold drop-shadow-[0_0_5px_#7b2ff7]">Throughput</span>
              <span className="material-symbols-outlined text-neon-purple/50 text-xl">dns</span>
            </div>
            <div className="text-4xl font-headline font-bold text-surface-900 tabular-nums tracking-tighter relative z-10">
              {currentThroughput}<span className="text-sm font-mono text-neon-purple/60 ml-1.5 tracking-normal">req/s</span>
            </div>
            <div className="mt-4 relative z-10 opacity-80 group-hover:opacity-100 transition-opacity">
              {renderSparkline(throughputData, '#7b2ff7', 45)}
            </div>
            <div className="absolute bottom-6 right-6">
              <span className="text-[9px] font-mono text-neon-purple/40 uppercase tracking-widest  border border-neon-purple/20 px-1 py-0.5">PEAK</span>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ Terminal + Diagnostics ═══ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 shrink-0">
        {/* Terminal */}
        <div className="lg:col-span-2 bg-[#0D1117] rounded-xl overflow-hidden flex flex-col h-full lg:h-[220px] border border-neon-green/10 relative shadow-md">
          <div className="absolute inset-0 bg-[linear-gradient(rgba(0,245,160,0.03)_1px,transparent_1px)] bg-[size:100%_4px] pointer-events-none z-0" />
          <div className="px-6 py-4 bg-[#161B22] border-b border-[#21262D] flex items-center justify-between shrink-0 relative z-10">
            <div className="flex items-center gap-4">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-neon-red shadow-[0_0_8px_#ff3e6c]" />
                <div className="w-3 h-3 rounded-full bg-accent-warning shadow-[0_0_8px_#fbbf24]" />
                <div className="w-3 h-3 rounded-full bg-neon-green shadow-[0_0_8px_#00f5a0]" />
              </div>
              <span className="text-[10px] font-mono font-bold text-neon-green uppercase tracking-widest ml-2 drop-shadow-[0_0_5px_#00f5a0]">sys_engine_logs.sh</span>
            </div>
            <span className="text-[9px] px-2 py-0.5 border border-neon-green/30 bg-neon-green/10 font-mono text-neon-green  shadow-[0_0_10px_rgba(0,245,160,0.2)]">● LIVE</span>
          </div>
          <div ref={terminalRef} className="p-6 font-mono text-xs text-[#E8EDF5]/70 overflow-y-auto space-y-2 flex-1 no-scrollbar relative z-10 custom-scrollbar bg-[#0D1117]">
            {logs.map((log, i) => (
              <div key={i} className={`flex gap-4 ${i === logs.length - 1 ? 'opacity-50' : ''} ${i === logs.length - 2 ? 'opacity-70' : ''}`}>
                <span className="text-neon-green/40 shrink-0 w-[70px]">[{log.time}]</span>
                <span className={`${log.color} ${log.bg} shrink-0 w-[42px] text-center  border border-current px-1 text-[10px] py-0.5 shadow-[0_0_5px_currentColor]`}>{log.level}</span>
                <span className="break-words text-[11px] font-medium leading-relaxed tracking-wide text-surface-900 drop-shadow-sm">{log.msg}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Diagnostics */}
        <div className="glass-dark p-4 flex flex-col h-full lg:h-[220px] border border-neon-blue/10 relative overflow-hidden group hover:border-neon-blue/30 transition-colors">
          <div className="absolute top-0 right-0 w-32 h-full bg-gradient-to-l from-neon-blue/5 to-transparent pointer-events-none" />
          
          <h4 className="font-headline font-bold text-surface-900 tracking-tight text-[16px] mb-4 relative z-10 flex items-center gap-2 shrink-0">
            <span className="material-symbols-outlined text-neon-blue text-[20px] glow-neon">tune</span>
            System Diagnostics
          </h4>

          <div className="space-y-6 flex-1 overflow-y-auto custom-scrollbar pr-2 relative z-10">
            {/* CPU */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-mono text-surface-800/60 uppercase tracking-widest">CPU Usage</span>
                <span className={`text-[10px] font-mono font-bold tabular-nums drop-shadow-[0_0_5px_currentColor] ${currentCPU > 60 ? 'text-accent-warning' : 'text-neon-blue'}`}>
                  {currentCPU.toFixed(1)}%
                </span>
              </div>
              {renderSparkline(cpuData, currentCPU > 60 ? '#fbbf24' : '#00b4d8', 35)}
            </div>

            {/* Memory */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-[10px] font-mono text-surface-800/60 uppercase tracking-widest">Memory (VRAM)</span>
                <span className="text-[10px] font-mono font-bold text-neon-purple tabular-nums drop-shadow-[0_0_5px_#7b2ff7]">{currentMem.toFixed(1)} GB</span>
              </div>
              <div className="w-full bg-surface-200/50 h-1.5 rounded-full overflow-hidden shadow-inner blur-[0.5px]">
                <div className="bg-gradient-to-r from-neon-purple to-[#a371f7] h-full rounded-full transition-all duration-500 shadow-[0_0_10px_#7b2ff7]" style={{ width: `${(currentMem / 8) * 100}%` }} />
              </div>
              <p className="text-[9px] font-mono text-surface-800/30 mt-1.5 tracking-wider">8 GB TOTAL // {((currentMem/8)*100).toFixed(1)}%</p>
            </div>

            {/* Minor Stats */}
            <div className="space-y-4 pt-4 border-t border-white/5">
              <div className="flex justify-between items-center">
                <p className="text-[10px] font-mono text-surface-800/60 uppercase tracking-widest">Queue Depth</p>
                <span className="text-[10px] font-mono font-bold text-neon-green flex items-center gap-1">
                  0 ms
                  <span className="material-symbols-outlined text-[14px]">check_circle</span>
                </span>
              </div>

              <div className="flex justify-between items-center">
                <p className="text-[10px] font-mono text-surface-800/60 uppercase tracking-widest">Model Vers</p>
                <span className="px-1.5 py-0.5 bg-neon-blue/10 border border-neon-blue/30 text-[9px] font-mono  text-neon-blue font-bold shadow-[0_0_8px_rgba(0,180,216,0.3)]">v4.2.1</span>
              </div>

              <div className="flex justify-between items-center">
                <p className="text-[10px] font-mono text-surface-800/60 uppercase tracking-widest">Uptime</p>
                <span className="text-[10px] font-mono font-bold text-surface-900">14d 7h 32m</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ Neural Network Live Visualization ═══ */}
      <div className="glass-dark p-4 border border-neon-blue/10 relative overflow-hidden group flex-1 min-h-[200px] flex flex-col">
        <div className="absolute inset-0 bg-gradient-to-b from-brand-bg to-transparent opacity-50 pointer-events-none z-0" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-[radial-gradient(circle_at_center,rgba(0,180,216,0.05)_0%,transparent_70%)] pointer-events-none z-0" />

        <div className="flex justify-between items-start mb-4 relative z-10">
          <div>
            <h4 className="font-headline font-bold text-surface-900 tracking-tight text-xl glow-neon">Neural Pathway Activity</h4>
            <p className="text-[10px] font-mono text-neon-blue/60 uppercase tracking-widest mt-1">Active predictive cluster mapping</p>
          </div>
          <div className="flex items-center gap-4 text-[9px] font-mono uppercase tracking-widest text-surface-800/40">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2  bg-neon-blue shadow-[0_0_10px_#00b4d8] animate-pulse" /> Active Node</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2  border border-surface-800/40" /> Idle Stack</span>
          </div>
        </div>

        {/* Programmatic Neural Net Visualization - Blade Runner Style */}
        <div className="relative flex-1 w-full overflow-hidden border border-white/5 bg-surface-50/30 backdrop-blur z-10 shadow-inner mt-2">
          <svg className="w-full h-full" viewBox="-20 -30 840 260" preserveAspectRatio="xMidYMid meet">
            <defs>
              <filter id="neon-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Layer 1 — Input nodes */}
            {[20, 52, 84, 116, 148, 180].map((y, i) => (
              <g key={`l1-${i}`}>
                <rect x={76} y={y-4} width={8} height={8} fill="rgba(0,180,216,0.1)" stroke="#00b4d8" strokeWidth="1" filter="url(#neon-glow)" />
                <circle cx={80} cy={y} r={1.5} fill="#00b4d8" />
                {/* Connections to layer 2 */}
                {[40, 80, 120, 160].map((y2, j) => (
                  <path key={j} d={`M86 ${y} C 150 ${y}, 230 ${y2}, 294 ${y2}`} fill="none" stroke="#00b4d8" strokeWidth="1" opacity={Math.random() > 0.4 ? 0.3 : 0.05} />
                ))}
              </g>
            ))}

            {/* Layer 2 — Hidden */}
            {[40, 80, 120, 160].map((y, i) => (
              <g key={`l2-${i}`}>
                <polygon points="300,y-6 306,y 300,y+6 294,y" transform={`translate(0, ${y - (y-6)})`} fill="rgba(123,47,247,0.15)" stroke="#7b2ff7" strokeWidth="1.5" filter="url(#neon-glow)" />
                {[60, 100, 140].map((y2, j) => (
                  <path key={j} d={`M306 ${y} C 360 ${y}, 440 ${y2}, 494 ${y2}`} fill="none" stroke="#7b2ff7" strokeWidth="1" opacity={Math.random() > 0.3 ? 0.4 : 0.1} />
                ))}
              </g>
            ))}

            {/* Layer 3 — Hidden */}
            {[60, 100, 140].map((y, i) => (
              <g key={`l3-${i}`}>
                <polygon points="500,y-6 506,y 500,y+6 494,y" transform={`translate(0, ${y - (y-6)})`} fill="rgba(0,245,160,0.15)" stroke="#00f5a0" strokeWidth="1.5" filter="url(#neon-glow)" />
                {[80, 120].map((y2, j) => (
                  <path key={j} d={`M506 ${y} C 560 ${y}, 640 ${y2}, 694 ${y2}`} fill="none" stroke="#00f5a0" strokeWidth="1.5" opacity={0.3} />
                ))}
              </g>
            ))}

            {/* Layer 4 — Output */}
            {[80, 120].map((y, i) => (
              <g key={`l4-${i}`}>
                <rect x={692} y={y-8} width={16} height={16} rx="2" fill="rgba(0,180,216,0.2)" stroke="#00b4d8" strokeWidth="2" filter="url(#neon-glow)" />
                <rect x={695} y={y-5} width={10} height={10} rx="1" fill="#00b4d8" className={i===0 ? "animate-pulse" : ""} />
                {/* Output energy lines */}
                <line x1={710} y1={y} x2={780} y2={y} stroke="#00b4d8" strokeWidth="2" opacity={0.6} strokeDasharray="4 4" className={i===0 ? "animate-pulse" : ""} />
              </g>
            ))}

            {/* Labels overlay */}
            <g className="font-mono text-[8px] uppercase tracking-widest fill-surface-800/50">
              <text x={80} y={20} textAnchor="middle">Sensor In</text>
              <text x={300} y={20} textAnchor="middle">L1 Spatial</text>
              <text x={500} y={40} textAnchor="middle">L2 Temporal</text>
              <text x={700} y={60} textAnchor="middle">Output Vec</text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  )
}

