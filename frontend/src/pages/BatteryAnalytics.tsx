import { useState, useEffect } from 'react'
import { useVehicle } from '../contexts/VehicleContext'

/* ── Simulated live data generator ── */
function useAnimatedValue(target: number, speed = 0.05) {
  const [value, setValue] = useState(target * 0.8)
  useEffect(() => {
    const id = setInterval(() => {
      setValue((v) => {
        const diff = target - v
        return Math.abs(diff) < 0.01 ? target : v + diff * speed
      })
    }, 30)
    return () => clearInterval(id)
  }, [target, speed])
  return value
}

/* ── Cell grid data (generated per vehicle) ── */
function generateCellData(cellMap: 'balanced' | 'deviation' | 'critical', deviation_mv: number) {
  return Array.from({ length: 96 }, (_, i) => {
    const baseVoltage = 3.6 + Math.random() * 0.08
    const deviationFactor = deviation_mv / 100

    let voltage = baseVoltage
    if (cellMap === 'critical') {
      // Several cells show critical deviation
      if (i === 7 || i === 23 || i === 45 || i === 67 || i === 82) {
        voltage += 0.25 + Math.random() * 0.1
      } else if (i % 8 === 0) {
        voltage += 0.16 + Math.random() * 0.05
      }
    } else if (cellMap === 'deviation') {
      // A few amber cells
      if (i === 23 || i === 67) {
        voltage += 0.18 + Math.random() * 0.05
      } else if (i % 12 === 0) {
        voltage += 0.12 + Math.random() * 0.04
      }
    } else {
      // Balanced — very minor variation
      voltage += deviationFactor * (Math.random() - 0.5) * 2
    }

    return {
      id: i + 1,
      voltage,
      temp: 28 + Math.random() * 6,
    }
  })
}

/* ── SoH historical data points ── */
const SOH_DATA = [100, 99.8, 99.5, 99.1, 98.6, 98.0, 97.5, 97.1, 96.5, 96.0, 95.4, 94.8, 94.2]
const SOH_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

/* ── Charge cycle distribution ── */
const CYCLE_DATA = [12, 18, 22, 28, 35, 40, 32, 25, 30, 38, 42, 28]
const CYCLE_MONTHS = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']

export default function BatteryAnalytics() {
  const { vehicle } = useVehicle()
  const sohAnim = useAnimatedValue(vehicle.health.soh_percent)
  const cyclesAnim = useAnimatedValue(vehicle.health.charge_cycles)
  const [activeTab, setActiveTab] = useState<'6m' | '1y' | 'all'>('1y')

  // Simulated live temperature — seeded from vehicle battery temp
  const [liveTemp, setLiveTemp] = useState(vehicle.battery.temperature_c)
  useEffect(() => {
    setLiveTemp(vehicle.battery.temperature_c)
  }, [vehicle.battery.temperature_c])

  useEffect(() => {
    const id = setInterval(() => {
      setLiveTemp((t) => {
        const base = vehicle.battery.temperature_c
        // Fluctuate around the vehicle's base temp
        const next = t + (Math.random() - 0.5) * 0.4
        // Keep within ±3° of base
        if (next > base + 3) return base + 3
        if (next < base - 3) return base - 3
        return next
      })
    }, 2000)
    return () => clearInterval(id)
  }, [vehicle.battery.temperature_c])

  // Generate cell data based on vehicle health profile
  const cellData = generateCellData(vehicle.health.cell_voltage_map, vehicle.battery.cell_deviation_mv)

  // Temp status helper
  const tempStatus = liveTemp > 50 ? 'HOT' : liveTemp > 35 ? 'WARM' : 'STABLE'
  const tempColor = liveTemp > 50 ? 'text-neon-red border-neon-red/50' : liveTemp > 35 ? 'text-accent-warning border-accent-warning/50' : 'text-neon-blue border-neon-blue/50'
  const tempStrokeColor = liveTemp > 50 ? '#ff3e6c' : liveTemp > 35 ? '#fbbf24' : '#00b4d8'

  // SoH status
  const sohStatus = vehicle.health.soh_percent >= 90 ? 'Nominal' : vehicle.health.soh_percent >= 80 ? 'Aging' : 'Critical'
  const sohStatusColor = vehicle.health.soh_percent >= 90 ? 'text-accent-success' : vehicle.health.soh_percent >= 80 ? 'text-accent-warning' : 'text-neon-red'
  const sohBarColor = vehicle.health.soh_percent >= 90
    ? 'bg-gradient-to-r from-accent-success to-[#17ffae] shadow-[0_0_15px_#00f5a0]'
    : vehicle.health.soh_percent >= 80
    ? 'bg-gradient-to-r from-accent-warning to-[#ffd60a] shadow-[0_0_15px_#fbbf24]'
    : 'bg-gradient-to-r from-neon-red to-[#ff7096] shadow-[0_0_15px_#ff3e6c]'
  const sohBorderColor = vehicle.health.soh_percent >= 90 ? 'border-accent-success/20' : vehicle.health.soh_percent >= 80 ? 'border-accent-warning/20' : 'border-neon-red/20'

  // Service urgency
  const serviceUrgent = vehicle.health.next_service_days <= 30

  // SoH chart SVG
  const sohMin = 90
  const sohMax = 101
  const chartW = 800
  const chartH = 200

  const sohPath = SOH_DATA.map((v, i) => {
    const x = (i / (SOH_DATA.length - 1)) * chartW
    const y = chartH - ((v - sohMin) / (sohMax - sohMin)) * chartH
    return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
  }).join(' ')

  const sohArea = `${sohPath} L ${chartW} ${chartH} L 0 ${chartH} Z`

  // Lifetime bar segments
  const lifetimeYears = vehicle.health.lifetime_years_remaining
  const lifetimeFilled = Math.min(10, Math.round(lifetimeYears))

  return (
    <div className="p-4 md:p-5 lg:p-6 flex flex-col gap-4 max-w-7xl mx-auto w-full h-full overflow-y-auto no-scrollbar">

      {/* ═══ Summary Stats ═══ */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 stagger-children">
        {/* Battery Health */}
        <div className={`glass-dark p-5  card-hover relative overflow-hidden border ${sohBorderColor}`}>
          <div className="absolute -top-10 -right-10 w-32 h-32 bg-accent-success/15 rounded-full blur-[40px]" />
          <p className="text-[10px] text-accent-success font-mono uppercase tracking-widest font-bold mb-3">Battery Health (SoH)</p>
          <div className="flex items-baseline gap-3 relative z-10">
            <span className="text-5xl font-headline font-bold text-surface-900 tabular-nums tracking-tight">{sohAnim.toFixed(1)}%</span>
            <span className={`flex items-center gap-1 ${sohStatusColor} text-xs font-mono tracking-widest uppercase border border-current/30 px-2 py-0.5  shadow-[0_0_10px_currentColor]`}>
              <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
              {sohStatus}
            </span>
          </div>
          <div className="mt-5 w-full bg-surface-200/50 h-2 rounded-full overflow-hidden relative z-10 shadow-inner">
            <div className={`h-full rounded-full transition-all duration-1000 ${sohBarColor}`} style={{ width: `${sohAnim}%` }} />
          </div>
          <p className="text-[10px] text-surface-800/40 mt-3 font-mono relative z-10">MIN THRESHOLD // 80%</p>
        </div>

        {/* Charge Cycles */}
        <div className="glass-dark p-5  card-hover relative overflow-hidden border border-neon-blue/20">
          <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-blue/15 rounded-full blur-[40px]" />
          <p className="text-[10px] text-neon-blue font-mono uppercase tracking-widest font-bold mb-3">Charge Cycles</p>
          <div className="flex items-baseline gap-2 relative z-10">
            <span className="text-5xl font-headline font-bold text-surface-900 tabular-nums tracking-tight glow-neon">{Math.round(cyclesAnim)}</span>
            <span className="text-sm font-mono text-surface-800/40 font-medium">/ 2000 max</span>
          </div>
          <div className="mt-5 w-full bg-surface-200/50 h-2 rounded-full overflow-hidden relative z-10 shadow-inner">
            <div className="h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-neon-blue to-neon-accent shadow-[0_0_15px_#00b4d8]" style={{ width: `${(cyclesAnim / 2000) * 100}%` }} />
          </div>
          <p className="text-[10px] text-surface-800/40 mt-3 font-mono relative z-10">OPTIMIZED PATTERN // ACTIVE</p>
        </div>

        {/* Lifetime Remaining */}
        <div className="glass-dark p-5  card-hover relative overflow-hidden border border-neon-purple/20">
          <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-purple/15 rounded-full blur-[40px]" />
          <p className="text-[10px] text-neon-purple font-mono uppercase tracking-widest font-bold mb-3">Lifetime Remaining</p>
          <div className="flex items-baseline gap-2 relative z-10">
            <span className="text-5xl font-headline font-bold text-surface-900 tracking-tight drop-shadow-[0_0_15px_rgba(123,47,247,0.4)]">{lifetimeYears}</span>
            <span className="text-sm font-mono text-surface-800/40 font-medium">years</span>
          </div>
          <div className="mt-5 flex gap-1.5 relative z-10">
            {Array.from({ length: 10 }, (_, i) => (
              <div key={i} className={`flex-1 h-2  flex items-center justify-center ${
                i < lifetimeFilled ? 'bg-gradient-to-r from-neon-purple to-[#a371f7] shadow-[0_0_8px_rgba(123,47,247,0.6)]' : 'bg-surface-200/50'
              }`} />
            ))}
          </div>
          <p className="text-[10px] text-surface-800/40 mt-3 font-mono relative z-10">DEGRADATION RATE // {vehicle.health.degradation_rate_per_month}%/mo</p>
        </div>
      </div>

      {/* ═══ Degradation Chart (SVG) ═══ */}
      <div className="glass-dark p-5 card-hover border border-neon-blue/10 flex-1 flex flex-col min-h-[260px]">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-xl font-headline font-bold text-surface-900 tracking-tight">State of Health Trend</h3>
            <p className="text-[10px] font-mono text-surface-800/40 mt-1 uppercase tracking-widest">SoH degradation over time</p>
          </div>
          <div className="flex bg-surface-100/50 p-1.5 border border-white/5 backdrop-blur shrink-0">
            {(['6m', '1y', 'all'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                className={`px-4 py-1.5 text-[9px] font-mono font-bold transition-all duration-300 uppercase tracking-widest ${
                  activeTab === t
                    ? 'bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,180,216,0.3)] border border-neon-blue/30'
                    : 'text-surface-800/40 hover:text-surface-900'
                }`}
              >
                {t === '6m' ? '6 Months' : t === '1y' ? '1 Year' : 'All Time'}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-1 flex-col relative w-full pt-4">
          <svg className="w-full h-full" viewBox={`0 0 ${chartW} ${chartH}`} preserveAspectRatio="none">
            <defs>
              <linearGradient id="sohGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: 'rgba(0, 180, 216, 0.3)' }} />
                <stop offset="100%" style={{ stopColor: 'rgba(0, 180, 216, 0)' }} />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            {/* Grid lines */}
            {[92, 94, 96, 98, 100].map((v) => {
              const y = chartH - ((v - sohMin) / (sohMax - sohMin)) * chartH
              return <line key={v} x1={0} y1={y} x2={chartW} y2={y} stroke="rgba(255,255,255,0.03)" strokeWidth="1" strokeDasharray="4 4" />
            })}
            {/* Area */}
            <path d={sohArea} fill="url(#sohGrad)" />
            {/* Line */}
            <path d={sohPath} fill="none" stroke="#00b4d8" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)" />
            {/* Data points */}
            {SOH_DATA.map((v, i) => {
              const x = (i / (SOH_DATA.length - 1)) * chartW
              const y = chartH - ((v - sohMin) / (sohMax - sohMin)) * chartH
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r={i === SOH_DATA.length - 1 ? 6 : 4} fill="#0a0e17" stroke="#00b4d8" strokeWidth="2" filter="url(#glow)" />
                  {i === SOH_DATA.length - 1 && (
                    <>
                      <circle cx={x} cy={y} r={12} fill="none" stroke="#00b4d8" strokeWidth="1" className="animate-ping" opacity="0.5" />
                      <circle cx={x} cy={y} r={16} fill="rgba(0,180,216,0.15)" stroke="none" />
                    </>
                  )}
                </g>
              )
            })}
          </svg>
          {/* Y-axis labels */}
          <div className="absolute top-4 bottom-8 -left-1 flex flex-col justify-between text-[10px] font-mono text-surface-800/40 pointer-events-none">
            {[100, 98, 96, 94, 92].map((v) => <span key={v}>{v}%</span>)}
          </div>
          {/* X-axis labels */}
          <div className="flex justify-between text-[9px] font-mono font-bold text-surface-800/40 uppercase tracking-widest px-1 mt-3">
            {SOH_MONTHS.map((m, i) => <span key={`${m}${i}`}>{m}</span>)}
          </div>
        </div>
      </div>

      {/* ═══ Bento Grid ═══ */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">

        {/* Cell Voltage Heatmap */}
        <div className="lg:col-span-2 glass-dark p-6  card-hover border border-neon-blue/10">
          <div className="flex items-center justify-between mb-6">
            <h4 className="font-headline font-bold text-surface-900 tracking-tight">Cell Voltage Map</h4>
            <span className="text-[10px] font-mono border border-neon-blue/30 bg-neon-blue/10 px-2 py-1  text-neon-blue font-bold uppercase tracking-widest shadow-[0_0_10px_rgba(0,180,216,0.2)]">96 Cells</span>
          </div>
          <div className="grid grid-cols-16 gap-1 bg-surface-100/30 p-2  border border-white/5">
            {cellData.map((cell) => {
              const deviation = Math.abs(cell.voltage - 3.7)
              const isCritical = deviation > 0.2
              const isWarning = deviation > 0.15 && !isCritical
              return (
                <div
                  key={cell.id}
                  className={`h-6  transition-all duration-300 cursor-pointer hover:scale-110 relative group ${
                    isCritical
                      ? 'bg-neon-red border-[0.5px] border-[#ff7096] shadow-[0_0_10px_rgba(255,62,108,0.6)] z-10'
                      : isWarning
                      ? 'bg-accent-warning border-[0.5px] border-[#ffd60a] shadow-[0_0_10px_rgba(255,214,10,0.6)] z-10'
                      : 'bg-accent-success/30 border border-accent-success/20 hover:bg-accent-success/50'
                  }`}
                  title={`Cell ${cell.id}: ${cell.voltage.toFixed(3)}V / ${cell.temp.toFixed(1)}°C`}
                />
              )
            })}
          </div>
          <div className="mt-5 flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-surface-800/40 border-t border-white/5 pt-4">
            <div className="flex items-center gap-5">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2  bg-accent-success/30 border border-accent-success/50" />
                Balanced
              </span>
              <span className="flex items-center gap-1.5 text-accent-warning">
                <span className="w-2 h-2  bg-accent-warning shadow-[0_0_8px_#fbbf24]" />
                Deviation
              </span>
              <span className="flex items-center gap-1.5 text-neon-red">
                <span className="w-2 h-2  bg-neon-red shadow-[0_0_8px_#ff3e6c]" />
                Critical
              </span>
            </div>
            <span className="font-bold text-neon-blue drop-shadow-[0_0_5px_#00b4d8]">Δ {vehicle.battery.cell_deviation_mv}mV</span>
          </div>
        </div>

        {/* Core Temperature Gauge */}
        <div className="glass-dark p-6  card-hover relative overflow-hidden border border-neon-blue/10">
          <div className="absolute -top-10 -right-10 w-40 h-40 bg-neon-blue/15 rounded-full blur-[40px]" />
          <div className="relative z-10 h-full flex flex-col">
            <h4 className="font-headline font-bold text-surface-900 tracking-tight text-center">Core Temp</h4>
            <div className="flex flex-1 items-center justify-center py-4">
              <div className="relative w-36 h-36">
                <svg className="w-full h-full -rotate-90 drop-shadow-[0_0_15px_rgba(0,180,216,0.3)]" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6" />
                  <circle
                    cx="50" cy="50" r="42" fill="none"
                    stroke={tempStrokeColor}
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={`${(liveTemp / 60) * 264} 264`}
                    className="transition-all duration-500"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center pt-1">
                  <span className="text-3xl font-headline font-bold text-surface-900 tabular-nums tracking-tighter glow-neon">{liveTemp.toFixed(1)}°</span>
                  <span className={`text-[9px] font-mono tracking-widest mt-1 border px-2 py-0.5  shadow-[0_0_10px_currentColor] ${tempColor}`}>
                    {tempStatus}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center text-[10px] font-mono mt-auto border-t border-white/5 pt-4 uppercase tracking-widest">
              <span className="text-surface-800/40">Liquid Cooling</span>
              <span className="text-neon-green flex items-center gap-1.5 shadow-[0_0_8px_rgba(0,245,160,0.5)]">
                <span className="w-1.5 h-1.5 bg-neon-green rounded-full animate-pulse" />
                ACTIVE
              </span>
            </div>
          </div>
        </div>

        {/* Charge Cycle Bar Chart */}
        <div className="glass-dark p-6  card-hover flex flex-col border border-neon-blue/10">
          <h4 className="font-headline font-bold text-surface-900 tracking-tight text-center mb-1">Monthly Cycles</h4>
          <p className="text-[9px] font-mono text-surface-800/40 uppercase tracking-widest text-center mb-6">Last 12 months</p>
          <div className="flex-1 flex items-end gap-1.5 min-h-[120px] bg-gradient-to-t from-surface-100/50 to-transparent p-2  border border-white/5">
            {CYCLE_DATA.map((v, i) => {
              const maxV = Math.max(...CYCLE_DATA)
              const isLast = i === CYCLE_DATA.length - 1
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group">
                  <div
                    className={`w-full  relative transition-all duration-500 group-hover:scale-y-105 group-hover:bg-[#90e0ef] ${
                      isLast
                        ? 'bg-gradient-to-t from-neon-blue/80 to-[#90e0ef] shadow-[0_0_10px_#00b4d8]'
                        : 'bg-neon-blue/20 group-hover:bg-neon-blue/40'
                    }`}
                    style={{ height: `${(v / maxV) * 100}%` }}
                  >
                    <div className="opacity-0 group-hover:opacity-100 absolute -top-5 left-1/2 -translate-x-1/2 text-[9px] font-mono text-neon-blue transition-opacity drop-shadow-[0_0_5px_#00b4d8]">{v}</div>
                  </div>
                  <span className={`text-[8px] font-mono uppercase tracking-wider ${isLast ? 'text-neon-blue' : 'text-surface-800/30 group-hover:text-surface-800/60'}`}>
                    {CYCLE_MONTHS[i]}
                  </span>
                </div>
              )
            })}
          </div>
          <button className="mt-5 w-full py-3 bg-neon-blue/10 border border-neon-blue/30  text-[10px] font-mono font-bold text-neon-blue uppercase tracking-widest hover:bg-neon-blue/20 hover:shadow-[0_0_15px_rgba(0,180,216,0.2)] transition-all">
            Full Report
          </button>
        </div>
      </div>

      {/* ═══ Maintenance ═══ */}
      <div className={`glass-dark p-4 shrink-0 flex flex-col md:flex-row items-center gap-6 card-hover relative overflow-hidden ${serviceUrgent ? 'border-2 border-accent-warning/40' : 'border border-neon-blue/10'}`}>
        <div className="absolute top-0 right-0 w-64 h-full bg-gradient-to-l from-neon-blue/5 to-transparent pointer-events-none" />
        <div className={`w-14 h-14  border flex items-center justify-center shrink-0 relative z-10 ${serviceUrgent ? 'bg-accent-warning/10 border-accent-warning/30 text-accent-warning shadow-[inset_0_0_15px_rgba(251,191,36,0.2)]' : 'bg-surface-100/80 border-white/10 text-neon-blue shadow-[inset_0_0_15px_rgba(0,180,216,0.1)]'}`}>
          <span className="material-symbols-outlined text-[28px] drop-shadow-[0_0_8px_currentColor]">{serviceUrgent ? 'warning' : 'verified_user'}</span>
        </div>
        <div className="flex-1 relative z-10">
          <h5 className="font-headline font-bold text-surface-900 tracking-tight mb-2">Maintenance Recommendation</h5>
          <p className="text-[11px] font-mono text-surface-800/50 leading-relaxed uppercase tracking-wider">
            System predicts next deep cycle calibration in <span className={`font-bold px-1  drop-shadow-[0_0_5px_currentColor] ${serviceUrgent ? 'text-accent-warning bg-accent-warning/10' : 'text-neon-blue bg-neon-blue/10'}`}>{vehicle.health.next_service_days} days</span>. {serviceUrgent ? 'Service urgently needed.' : 'No immediate service required.'} Current wear rate: <span className="text-surface-900 font-bold border-b border-surface-800/30">{vehicle.health.degradation_rate_per_month}%/month</span>
          </p>
        </div>
        <button className={`px-8 py-3.5 text-brand-bg  text-[11px] font-mono font-extrabold shadow-[0_0_20px_rgba(0,180,216,0.3)] hover:scale-105 transition-all uppercase tracking-widest shrink-0 relative z-10 ${serviceUrgent ? 'bg-gradient-to-r from-accent-warning to-[#ffd60a]' : 'bg-gradient-to-r from-neon-blue to-neon-green'}`}>
          Schedule
        </button>
      </div>

    </div>
  )
}
