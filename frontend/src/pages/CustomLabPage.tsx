import { useVehicle } from '../contexts/VehicleContext'
import AnimatedNumber from '../components/AnimatedNumber'
import CarModel from '../components/CarModel'

const BAR_HEIGHTS = [40, 60, 35, 80, 55, 45, 65, 40, 50, 30, 75, 90, 60, 45, 35, 85, 50, 40, 60, 70]
const BAR_ACCENTS = new Set([4, 11, 16, 19])
const BAR_SECONDARY = new Set([6, 13])

export default function CustomLabPage() {
  const { vehicle, setLabMinimized } = useVehicle()

  const soh = vehicle.battery.soh_percent
  const soc = vehicle.battery.soc_percent
  const isCritical = soc < 20
  const accentColor = isCritical ? '#ef4444' : '#A855F7'
  const accentRgb = isCritical ? '239,68,68' : '168,85,247'
  const accentGradient = isCritical ? 'linear-gradient(to right, #ef4444, #991b1b)' : 'linear-gradient(to right, #A855F7, #7C3AED)'
  const sohColor = soh >= 80 ? accentColor : soh >= 65 ? '#FFB800' : '#ef4444'
  const socColor = soc >= 60 ? '#00FF88' : soc >= 30 ? '#FFB800' : '#ef4444'

  // Circular ring for SoC
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference * (1 - soc / 100)

  // Impact preview
  const rangeDiff = vehicle.range_km - 312
  const rangePercent = Math.round((rangeDiff / 312) * 100)
  const predictedEnergy = ((50 * 1000) / (vehicle.range_km * 1000 / vehicle.battery.capacity_kwh)).toFixed(1)
  const availableEnergy = (vehicle.battery.capacity_kwh * (soh / 100) * (soc / 100)).toFixed(1)

  return (
    <div className="flex flex-col h-full overflow-hidden bg-background">

      {/* ── Main Row ── */}
      <div className="flex flex-1 min-h-0 p-6 gap-6 overflow-hidden">

        {/* ── LEFT — Blueprint View (55%) ── */}
        <section className="w-[55%] relative flex flex-col rounded-2xl bg-surface-container-lowest overflow-hidden">

          {/* Radial glow — shifts with accent */}
          <div
            className="absolute inset-0 pointer-events-none transition-all duration-700"
            style={{ background: `radial-gradient(circle at 50% 55%, ${accentColor}1A 0%, transparent 70%)` }}
          />

          {/* Corner bracket accents */}
          <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 rounded-tl-2xl pointer-events-none transition-colors duration-700" style={{ borderColor: `${accentColor}80` }} />
          <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 rounded-tr-2xl pointer-events-none transition-colors duration-700" style={{ borderColor: `${accentColor}80` }} />
          <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 rounded-bl-2xl pointer-events-none transition-colors duration-700" style={{ borderColor: `${accentColor}80` }} />
          <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 rounded-br-2xl pointer-events-none transition-colors duration-700" style={{ borderColor: `${accentColor}80` }} />

          {/* Header */}
          <div className="absolute top-5 left-6 flex items-center gap-2 z-10">
            <div className="w-2 h-2 rounded-full animate-pulse transition-colors duration-700" style={{ background: accentColor }} />
            <span className="text-[10px] font-mono font-bold uppercase tracking-[0.25em] transition-colors duration-700" style={{ color: accentColor }}>
              {isCritical ? '⚠ Critical Battery — Model R' : 'Research Lab — Model R'}
            </span>
          </div>

          {/* 3D Car Model */}
          <div className="flex-1 relative" style={{ minHeight: 0 }}>
            {/* Telemetry overlays positioned over the 3D model */}
            <div className="absolute top-1/4 left-2 flex flex-col items-start pointer-events-none z-10">
              <span className="font-mono text-[10px] text-primary uppercase drop-shadow mb-1">Aero Winglet</span>
              <div className="w-20 h-px bg-gradient-to-r from-primary to-transparent" />
            </div>
            <div className="absolute bottom-1/3 right-2 flex flex-col items-end pointer-events-none z-10">
              <span className="font-mono text-[10px] text-primary uppercase drop-shadow mb-1">Rear Motor Array</span>
              <div className="w-28 h-px bg-gradient-to-l from-primary to-transparent" />
            </div>

            <CarModel
              batteryKwh={vehicle.battery.capacity_kwh}
              tempC={vehicle.battery.temperature_c}
              glowColor={accentColor}
              modelPath="/models/custom car.glb"
              regenActive={vehicle.realtime.regen_active}
              maxPowerKw={vehicle.specs.max_power_kw}
            />

            {/* Model name overlay */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-center pointer-events-none z-10 transition-colors duration-700">
              <h2 className={`text-4xl font-black italic tracking-tighter drop-shadow-lg transition-colors duration-700 ${isCritical ? 'text-red-300' : 'text-on-surface'}`}>MODEL R</h2>
              <p className="font-label text-[10px] uppercase tracking-[0.3em] mt-1 transition-colors duration-700" style={{ color: accentColor }}>
                {vehicle.subtitle}
              </p>
            </div>
          </div>

          {/* Bottom badge row */}
          <div className="shrink-0 flex items-center gap-2 px-5 pb-5 pt-2">
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
              style={{ background: `${sohColor}18`, border: `1px solid ${sohColor}40` }}
            >
              <span className="text-[9px] font-mono font-bold uppercase" style={{ color: sohColor }}>SoH {soh}%</span>
            </div>
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
              style={{ background: `${socColor}18`, border: `1px solid ${socColor}40` }}
            >
              <span className="text-[9px] font-mono font-bold uppercase" style={{ color: socColor }}>SoC {soc}%</span>
            </div>
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
              style={{
                background: vehicle.realtime.regen_active ? 'rgba(0,255,136,0.10)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${vehicle.realtime.regen_active ? 'rgba(0,255,136,0.30)' : 'rgba(255,255,255,0.10)'}`,
              }}
            >
              <div className={`w-1.5 h-1.5 rounded-full ${vehicle.realtime.regen_active ? 'bg-green-400 animate-pulse' : 'bg-slate-600'}`} />
              <span className={`text-[9px] font-mono font-bold uppercase ${vehicle.realtime.regen_active ? 'text-green-400' : 'text-slate-500'}`}>
                Regen {vehicle.realtime.regen_active ? 'ON' : 'OFF'}
              </span>
            </div>
            {/* Single active dot — lab indicator */}
            <div className="ml-auto flex gap-2">
              <span className="h-1 w-10 rounded-full transition-all duration-700" style={{ background: accentColor, boxShadow: `0 0 8px ${accentColor}` }} />
            </div>
          </div>
        </section>

        {/* ── RIGHT — Parameter Impact Cards (45%) ── */}
        <section className="w-[45%] flex flex-col gap-5 overflow-y-auto no-scrollbar">

          {/* Card 1 — Battery Status */}
          <div className="bg-surface-container rounded-2xl p-6 flex items-center justify-between group hover:bg-surface-variant transition-all duration-300">
            <div className="space-y-1">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Battery Status</h3>
              <div className="flex items-baseline gap-2">
                <AnimatedNumber
                  value={soc}
                  duration={2000}
                  className="font-mono text-5xl font-bold text-on-surface"
                />
                <span className="font-mono text-xl transition-colors duration-700" style={{ color: accentColor }}>%</span>
              </div>
              <div className="flex flex-col gap-1 mt-2">
                <div className="flex justify-between text-xs font-mono text-on-surface-variant">
                  <span>Health (SoH)</span>
                  <span style={{ color: soh >= 80 ? accentColor : soh >= 65 ? '#FFB800' : '#ef4444' }}>
                    {soh}%
                  </span>
                </div>
                <div className="flex justify-between text-xs font-mono text-on-surface-variant">
                  <span>Capacity</span>
                  <span className="text-on-surface">{vehicle.battery.capacity_kwh} kWh</span>
                </div>
                <div className="flex justify-between text-xs font-mono text-on-surface-variant">
                  <span>Temperature</span>
                  <span className="text-on-surface">{vehicle.battery.temperature_c}°C</span>
                </div>
              </div>
            </div>

            {/* Circular ring — purple */}
            <div className="relative w-24 h-24 shrink-0">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 96 96">
                <circle
                  cx="48" cy="48" r={radius}
                  fill="transparent"
                  stroke={`rgba(${accentRgb}, 0.15)`}
                  strokeWidth="8"
                  className="transition-colors duration-700"
                />
                <circle
                  cx="48" cy="48" r={radius}
                  fill="transparent"
                  stroke={accentColor}
                  strokeWidth="8"
                  strokeDasharray={`${circumference}`}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="round"
                  className="transition-all duration-700"
                  style={{ filter: `drop-shadow(0 0 6px rgba(${accentRgb}, 0.6))` }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span
                  className="material-symbols-outlined text-3xl transition-colors duration-700"
                  style={{ color: accentColor, fontVariationSettings: "'FILL' 1" }}
                >
                  {isCritical ? 'warning' : 'science'}
                </span>
              </div>
            </div>
          </div>

          {/* Card 2 — Energy Profile */}
          <div className="bg-surface-container rounded-2xl p-6 flex flex-col gap-4 hover:bg-surface-variant transition-all duration-300">
            <div className="flex justify-between items-center">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Energy Profile</h3>
              <div className="flex items-baseline gap-1.5">
                <AnimatedNumber
                  value={vehicle.range_km}
                  duration={1500}
                  className="font-mono text-lg font-bold text-on-surface"
                />
                <span className="font-mono text-xs text-on-surface-variant">km</span>
              </div>
            </div>
            <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, (soc / 100) * 100)}%`,
                  background: accentGradient,
                  boxShadow: `0 0 8px rgba(${accentRgb}, 0.5)`,
                }}
              />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-on-surface-variant uppercase tracking-tighter">
              <span>0 km</span>
              <span className="transition-colors duration-700" style={{ color: accentColor }}>{availableEnergy} kWh avail.</span>
              <span>{vehicle.battery.capacity_kwh} kWh</span>
            </div>
          </div>

          {/* Card 3 — Vehicle Config (click to open editor) */}
          <div
            className="bg-surface-container rounded-2xl p-6 flex flex-col gap-4 hover:bg-surface-variant transition-all duration-300 cursor-pointer group"
            onClick={() => setLabMinimized(false)}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Vehicle Config</h3>
              <button
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-[11px] font-bold uppercase tracking-widest transition-all duration-300 hover:scale-[1.03] shadow-sm transform-gpu pointer-events-auto"
                style={{ 
                  color: isCritical ? '#fff' : accentColor, 
                  background: isCritical ? accentColor : `rgba(${accentRgb}, 0.12)`, 
                  border: `1px solid ${isCritical ? accentColor : `rgba(${accentRgb}, 0.3)`}`,
                  boxShadow: isCritical ? `0 0 16px rgba(${accentRgb}, 0.4)` : 'none'
                }}
              >
                Configure
                <span className="material-symbols-outlined text-[14px]">tune</span>
              </button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: 'Mass', value: `${vehicle.specs.mass_kg}`, unit: 'kg' },
                { label: 'Drag Cd', value: vehicle.specs.drag_coefficient.toFixed(2), unit: '' },
                { label: 'Motor Eff', value: `${Math.round(vehicle.specs.motor_efficiency * 100)}`, unit: '%' },
                { label: 'Regen Eff', value: `${Math.round(vehicle.specs.regen_efficiency * 100)}`, unit: '%' },
                { label: 'HVAC', value: vehicle.realtime.cabin_hvac_kw.toFixed(1), unit: 'kW' },
                { label: 'Power Draw', value: vehicle.realtime.power_draw_kw.toFixed(1), unit: 'kW' },
              ].map(({ label, value, unit }) => (
                <div key={label} className="bg-white/5 rounded-xl p-3 text-center border border-white/5">
                  <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">{label}</p>
                  <p className="text-xs font-mono font-bold text-white">
                    {value}<span className="text-slate-500 text-[9px]">{unit}</span>
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Card 4 — Prediction Impact */}
          <div className="bg-surface-container rounded-2xl p-6 flex flex-col gap-4 hover:bg-surface-variant transition-all duration-300">
            <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Prediction Impact</h3>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Est. Range</span>
                <span className="text-white font-mono font-bold">{vehicle.range_km} km</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">vs Model V baseline</span>
                <span className={`font-mono ${rangeDiff >= 0 ? 'text-green-400' : 'text-amber-400'}`}>
                  {rangeDiff >= 0 ? '+' : ''}{rangeDiff} km ({rangePercent >= 0 ? '+' : ''}{rangePercent}%)
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Available energy</span>
                <span className="text-white font-mono">{availableEnergy} kWh</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Predicted 50 km energy</span>
                <span className="text-white font-mono">~{predictedEnergy} kWh</span>
              </div>
              <div className="pt-0.5">
                <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, (vehicle.range_km / 400) * 100)}%`,
                      background: rangeDiff >= 0 ? accentColor : '#FFB800',
                    }}
                  />
                </div>
              </div>
            </div>
            {soc < 20 && (
              <div className="p-2.5 rounded-lg bg-red-400/10 border border-red-400/20">
                <p className="text-[10px] text-red-400 font-mono">⚠ Critical charge — find a charger</p>
              </div>
            )}
            {soc >= 20 && soc < 40 && (
              <div className="p-2.5 rounded-lg bg-amber-400/10 border border-amber-400/20">
                <p className="text-[10px] text-amber-400 font-mono">⚠ Low charge — limited range</p>
              </div>
            )}
            {soh < 80 && (
              <div className="p-2.5 rounded-lg bg-amber-400/10 border border-amber-400/20">
                <p className="text-[10px] text-amber-400 font-mono">⚠ Degraded battery simulated</p>
              </div>
            )}
            {vehicle.specs.mass_kg > 2500 && (
              <div className="p-2.5 rounded-lg bg-orange-400/10 border border-orange-400/20">
                <p className="text-[10px] text-orange-400 font-mono">ℹ Heavy vehicle — high energy use</p>
              </div>
            )}
          </div>

        </section>
      </div>

      {/* ── Bottom — Parameter Impact Chart ── */}
      <section className="px-6 pb-6">
        <div className="bg-surface-container-low rounded-2xl p-6 h-52 flex flex-col">
          <div className="flex justify-between items-end mb-4">
            <div>
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">
                Parameter Impact
              </h3>
              <p className="font-mono text-xl font-bold text-on-surface">
                {vehicle.range_km}{' '}
                <span className="text-sm font-normal transition-colors duration-700" style={{ color: accentColor }}>km range</span>
              </p>
            </div>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full transition-colors duration-700" style={{ background: accentColor }} />
                <span className="font-label text-[10px] uppercase text-on-surface-variant">Better</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full transition-colors duration-700" style={{ background: `rgba(${accentRgb}, 0.3)` }} />
                <span className="font-label text-[10px] uppercase text-on-surface-variant">Baseline</span>
              </div>
            </div>
          </div>

          {/* Bar chart */}
          <div className="flex-1 flex items-end justify-between gap-1.5 px-2">
            {BAR_HEIGHTS.map((h, i) => (
              <div
                key={i}
                className="w-full rounded-t-lg transition-all duration-300 hover:brightness-125"
                style={{
                  height: `${h}%`,
                  background: BAR_ACCENTS.has(i)
                    ? accentColor
                    : BAR_SECONDARY.has(i)
                    ? `rgba(${accentRgb}, 0.5)`
                    : `rgba(${accentRgb}, 0.18)`,
                  boxShadow: BAR_ACCENTS.has(i) ? `0 0 8px rgba(${accentRgb}, 0.4)` : 'none',
                }}
              />
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
