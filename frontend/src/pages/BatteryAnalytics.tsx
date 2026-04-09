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
      if (i === 7 || i === 23 || i === 45 || i === 67 || i === 82) {
        voltage += 0.25 + Math.random() * 0.1
      } else if (i % 8 === 0) {
        voltage += 0.16 + Math.random() * 0.05
      }
    } else if (cellMap === 'deviation') {
      if (i === 23 || i === 67) {
        voltage += 0.18 + Math.random() * 0.05
      } else if (i % 12 === 0) {
        voltage += 0.12 + Math.random() * 0.04
      }
    } else {
      voltage += deviationFactor * (Math.random() - 0.5) * 2
    }

    return { id: i + 1, voltage, temp: 28 + Math.random() * 6 }
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
  const [activeTab, setActiveTab] = useState<'1Y' | 'ALL'>('ALL')

  const [liveTemp, setLiveTemp] = useState(vehicle.battery.temperature_c)
  useEffect(() => {
    setLiveTemp(vehicle.battery.temperature_c)
  }, [vehicle.battery.temperature_c])

  useEffect(() => {
    const id = setInterval(() => {
      setLiveTemp((t) => {
        const base = vehicle.battery.temperature_c
        const next = t + (Math.random() - 0.5) * 0.4
        if (next > base + 3) return base + 3
        if (next < base - 3) return base - 3
        return next
      })
    }, 2000)
    return () => clearInterval(id)
  }, [vehicle.battery.temperature_c])

  const cellData = generateCellData(vehicle.health.cell_voltage_map, vehicle.battery.cell_deviation_mv)

  const tempStrokeColor = liveTemp > 50 ? '#ffb4ab' : liveTemp > 35 ? '#ffba20' : '#afecff'
  const tempStatus = liveTemp > 50 ? 'HOT' : liveTemp > 35 ? 'WARM' : 'STABLE'
  const tempTextColor = liveTemp > 50 ? 'text-error' : liveTemp > 35 ? 'text-tertiary-container' : 'text-primary'

  const serviceUrgent = vehicle.health.next_service_days <= 30
  const lifetimeYears = vehicle.health.lifetime_years_remaining

  /* SoH chart */
  const sohMin = 90
  const sohMax = 101
  const chartW = 1000
  const chartH = 200

  const sohPoints = SOH_DATA.map((v, i) => ({
    x: (i / (SOH_DATA.length - 1)) * chartW,
    y: chartH - ((v - sohMin) / (sohMax - sohMin)) * chartH,
  }))

  const sohPath = sohPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  const sohArea = `${sohPath} L ${chartW} ${chartH} L 0 ${chartH} Z`

  /* Cycle chart */
  const maxCycle = Math.max(...CYCLE_DATA)

  return (
    <div className="pt-6 px-8 pb-8 overflow-y-auto no-scrollbar h-full">

      {/* ══ Hero Stats Row ══ */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

        {/* Battery Health Index */}
        <div className="bg-surface-container p-6 rounded-3xl border border-outline-variant/10 relative overflow-hidden group hover:bg-surface-variant transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <span className="material-symbols-outlined text-6xl">health_and_safety</span>
          </div>
          <p className="font-label text-[10px] uppercase tracking-[0.2em] text-primary/60 mb-2">Battery Health Index</p>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-5xl font-bold text-primary">{sohAnim.toFixed(1)}</span>
            <span className="font-mono text-xl text-primary/40">%</span>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full bg-primary shadow-[0_0_8px_#afecff] transition-all duration-1000"
                style={{ width: `${sohAnim}%` }}
              />
            </div>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className={`text-[10px] font-mono py-1 px-2 rounded ${
              vehicle.health.soh_percent >= 90
                ? 'bg-secondary-container/10 text-secondary-container'
                : vehicle.health.soh_percent >= 80
                ? 'bg-tertiary-container/10 text-tertiary-container'
                : 'bg-error-container/20 text-error'
            }`}>
              {vehicle.health.soh_percent >= 90 ? 'Nominal' : vehicle.health.soh_percent >= 80 ? 'Aging' : 'Critical'}
            </span>
          </div>
        </div>

        {/* Discharge Cycles */}
        <div className="bg-surface-container p-6 rounded-3xl border border-outline-variant/10 relative overflow-hidden group hover:bg-surface-variant transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <span className="material-symbols-outlined text-6xl">sync</span>
          </div>
          <p className="font-label text-[10px] uppercase tracking-[0.2em] text-primary/60 mb-2">Discharge Cycles</p>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-5xl font-bold text-on-surface">{Math.round(cyclesAnim)}</span>
            <span className="font-mono text-xl text-on-surface/40">CYC</span>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full bg-primary shadow-[0_0_8px_#afecff] transition-all duration-1000"
                style={{ width: `${(Math.round(cyclesAnim) / 1000) * 100}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-on-surface/40 shrink-0">Rated: 1000</span>
          </div>
        </div>

        {/* Estimated Core Life */}
        <div className="bg-surface-container p-6 rounded-3xl border border-outline-variant/10 relative overflow-hidden group hover:bg-surface-variant transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <span className="material-symbols-outlined text-6xl">hourglass_empty</span>
          </div>
          <p className="font-label text-[10px] uppercase tracking-[0.2em] text-primary/60 mb-2">Estimated Core Life</p>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-5xl font-bold text-on-surface">{lifetimeYears}</span>
            <span className="font-mono text-xl text-on-surface/40">YRS</span>
          </div>
          <p className="mt-4 text-[10px] font-mono text-on-surface/40">
            Degradation: {vehicle.health.degradation_rate_per_month}%/mo
          </p>
          <p className="text-[10px] font-mono text-on-surface/40">
            Next service in <span className={serviceUrgent ? 'text-error font-bold' : 'text-primary font-bold'}>{vehicle.health.next_service_days} days</span>
          </p>
        </div>
      </div>

      {/* ══ Full Width SoH Trend Chart ══ */}
      <div className="bg-surface-container-low p-8 rounded-[32px] border border-outline-variant/5 mb-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-on-surface">SoH Degradation Trend</h2>
            <p className="text-sm text-on-surface-variant">Cubic spline analysis of capacity retention over time</p>
          </div>
          <div className="flex gap-2">
            {(['1Y', 'ALL'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                className={`px-4 py-2 rounded-lg text-xs font-mono border transition-all duration-200 ${
                  activeTab === t
                    ? 'bg-primary text-on-primary border-primary/20 shadow-[0_0_12px_rgba(175,236,255,0.3)]'
                    : 'bg-surface-container-highest text-on-surface border-white/5 hover:bg-surface-variant'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* SVG Chart */}
        <div className="h-64 relative overflow-hidden">
          <svg className="w-full h-full" viewBox={`0 0 ${chartW} ${chartH}`} preserveAspectRatio="none">
            <defs>
              <linearGradient id="sohGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: '#afecff', stopOpacity: 0.2 }} />
                <stop offset="100%" style={{ stopColor: '#afecff', stopOpacity: 0 }} />
              </linearGradient>
            </defs>
            {/* Grid lines */}
            {[50, 100, 150].map((y) => (
              <line key={y} x1="0" y1={y} x2={chartW} y2={y} stroke="rgba(255,255,255,0.05)" strokeDasharray="4" />
            ))}
            {/* Area fill */}
            <path d={sohArea} fill="url(#sohGrad)" stroke="none" />
            {/* Line */}
            <path d={sohPath} fill="none" stroke="#afecff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            {/* Live dot */}
            <circle
              cx={sohPoints[sohPoints.length - 1].x}
              cy={sohPoints[sohPoints.length - 1].y}
              r="5"
              fill="#0a0e16"
              stroke="#afecff"
              strokeWidth="2"
            />
          </svg>
          {/* Y-axis labels */}
          <div className="absolute top-4 left-0 font-mono text-[10px] text-on-surface/40">100%</div>
          <div className="absolute top-[44%] left-0 font-mono text-[10px] text-on-surface/40">95%</div>
          <div className="absolute bottom-4 left-0 font-mono text-[10px] text-on-surface/40">90%</div>
        </div>

        {/* X-axis */}
        <div className="flex justify-between mt-4 px-2">
          {SOH_MONTHS.filter((_, i) => i % 3 === 0).map((m, i) => (
            <span key={i} className="font-mono text-[10px] text-on-surface/20">{m}</span>
          ))}
        </div>
      </div>

      {/* ══ Bottom Data Cluster ══ */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Cell Voltage Map */}
        <div className="bg-surface-container-low p-6 rounded-[32px] border border-outline-variant/5">
          <div className="flex justify-between items-start mb-6">
            <h3 className="font-label text-xs uppercase tracking-widest text-primary">Cell Voltage Map</h3>
            <span className="font-mono text-xs text-secondary-container">AVG 3.82V</span>
          </div>
          <div className="grid grid-cols-12 gap-1.5">
            {cellData.map((cell) => {
              const deviation = Math.abs(cell.voltage - 3.7)
              const isCritical = deviation > 0.2
              const isWarning = deviation > 0.15 && !isCritical
              return (
                <div
                  key={cell.id}
                  className={`aspect-square rounded-sm transition-all duration-500 cursor-pointer hover:scale-110 ${
                    isCritical
                      ? 'bg-orange-400'
                      : isWarning
                      ? 'bg-tertiary-container/80'
                      : 'bg-primary-container'
                  }`}
                  style={{ opacity: isCritical || isWarning ? 1 : 0.4 + Math.random() * 0.6 }}
                  title={`Cell ${cell.id}: ${cell.voltage.toFixed(3)}V`}
                />
              )
            })}
          </div>
          <div className="mt-4 flex justify-between">
            <div className="flex gap-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-primary-container" />
                <span className="text-[9px] text-on-surface/60 font-mono">NOMINAL</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-orange-400" />
                <span className="text-[9px] text-on-surface/60 font-mono">STRESS</span>
              </div>
            </div>
            <span className="text-[9px] text-on-surface/40 font-mono uppercase">
              Δ {vehicle.battery.cell_deviation_mv}mV
            </span>
          </div>
        </div>

        {/* Core Temperature Gauge */}
        <div className="bg-surface-container-low p-6 rounded-[32px] border border-outline-variant/5 flex flex-col items-center">
          <div className="w-full flex justify-between items-start mb-2">
            <h3 className="font-label text-xs uppercase tracking-widest text-primary">Core Temperature</h3>
          </div>
          <div className="relative w-48 h-48 flex items-center justify-center">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke="rgba(255,255,255,0.05)"
                strokeWidth="6"
              />
              <circle
                cx="50" cy="50" r="42"
                fill="none"
                stroke={tempStrokeColor}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={`${(liveTemp / 80) * 264} 264`}
                className="transition-all duration-500"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className={`font-mono text-4xl font-bold ${tempTextColor}`}>
                {liveTemp.toFixed(1)}°
              </span>
              <span className={`text-[9px] font-mono uppercase tracking-widest mt-1 ${tempTextColor}`}>
                {tempStatus}
              </span>
            </div>
          </div>
          <div className="w-full flex justify-between text-[10px] font-mono text-on-surface/40 mt-4 border-t border-white/5 pt-4">
            <span>Liquid Cooling</span>
            <span className="text-secondary-container flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-secondary-container animate-pulse" />
              ACTIVE
            </span>
          </div>
        </div>

        {/* Monthly Cycles + Maintenance */}
        <div className="bg-surface-container-low p-6 rounded-[32px] border border-outline-variant/5 flex flex-col">
          <h3 className="font-label text-xs uppercase tracking-widest text-primary mb-1">Monthly Cycles</h3>
          <p className="text-[10px] font-mono text-on-surface/40 uppercase tracking-widest mb-4">Last 12 months</p>

          <div className="flex-1 flex items-end gap-1.5 min-h-[100px]">
            {CYCLE_DATA.map((v, i) => {
              const isLast = i === CYCLE_DATA.length - 1
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1 group">
                  <div
                    className={`w-full rounded-t-sm transition-all duration-300 ${
                      isLast ? 'bg-primary' : 'bg-surface-container-highest hover:bg-primary/40'
                    }`}
                    style={{ height: `${(v / maxCycle) * 100}%` }}
                  />
                  <span className={`text-[8px] font-mono uppercase ${isLast ? 'text-primary' : 'text-on-surface/30'}`}>
                    {CYCLE_MONTHS[i]}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Maintenance row */}
          <div className={`mt-6 p-4 rounded-xl border flex items-center gap-3 ${
            serviceUrgent ? 'border-error/30 bg-error-container/10' : 'border-outline-variant/20 bg-surface-container'
          }`}>
            <span className={`material-symbols-outlined text-2xl ${serviceUrgent ? 'text-error' : 'text-primary'}`}>
              {serviceUrgent ? 'warning' : 'verified_user'}
            </span>
            <div>
              <p className="text-xs font-mono text-on-surface font-bold">
                {serviceUrgent ? 'Service Urgent' : 'No Service Needed'}
              </p>
              <p className="text-[10px] font-mono text-on-surface-variant">
                Next check in {vehicle.health.next_service_days} days
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
