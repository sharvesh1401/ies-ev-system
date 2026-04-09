import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CarModel from '../components/CarModel'
import AnimatedNumber from '../components/AnimatedNumber'
import VehicleSelector from '../components/VehicleSelector'
import { useVehicle } from '../contexts/VehicleContext'
import CustomLabPage from './CustomLabPage'
import useWindowSize from '../hooks/useWindowSize'

const VEHICLE_ORDER = ['model-v-performance', 'model-s-commuter', 'model-t-cargo']

const BAR_HEIGHTS = [40, 60, 35, 80, 55, 45, 65, 40, 50, 30, 75, 90, 60, 45, 35, 85, 50, 40, 60, 70]
const BAR_ACCENTS = new Set([4, 11, 16, 19]) // primary-coloured bars
const BAR_SECONDARY = new Set([6, 13])        // secondary-container bars

export default function Home() {
  const navigate = useNavigate()
  const { vehicle, allVehicles, currentVehicle, isCustomMode, isLabMinimized, setLabMinimized } = useVehicle() as any
  const [activeRange, setActiveRange] = useState<'1H' | '6H' | '24H'>('6H')
  const { isMobile } = useWindowSize()

  // Determine prev/next ghost vehicles for carousel
  const idx = VEHICLE_ORDER.indexOf(currentVehicle)
  const prevVehicle = allVehicles[VEHICLE_ORDER[(idx + VEHICLE_ORDER.length - 1) % VEHICLE_ORDER.length]]
  const nextVehicle = allVehicles[VEHICLE_ORDER[(idx + 1) % VEHICLE_ORDER.length]]

  // SVG circular progress for SoC
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference * (1 - vehicle.battery.soc_percent / 100)

  // Model display name
  const modelNames: Record<string, string> = {
    'model-v-performance': 'MODEL V',
    'model-s-commuter': 'MODEL S',
    'model-t-cargo': 'MODEL T',
    'custom-lab': 'MODEL R',
  }

  if (isCustomMode) return <CustomLabPage />

  return (
    <div className={`flex flex-col h-full bg-background ${isMobile ? 'overflow-y-auto no-scrollbar' : 'overflow-hidden'}`}>
      {/* ── Main Row ── */}
      <div className={`flex gap-6 ${isMobile ? 'flex-col p-3 shrink-0' : 'flex-1 min-h-0 p-6 overflow-hidden'}`}>

        {/* Left Panel – Vehicle Display (55%) */}
        <section className={`relative flex flex-col justify-center items-center rounded-2xl bg-surface-container-lowest overflow-hidden ${isMobile ? 'w-full' : 'w-[55%]'}`}>
          {/* Radial glow */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_#00d9ff10_0%,_transparent_70%)] pointer-events-none" />

          {/* Vehicle Selector */}
          <div className="absolute top-6 left-6 z-20">
            <VehicleSelector />
            <div className="flex items-center gap-2 mt-2">
              <div
                className="text-[10px] font-bold px-2 py-0.5 rounded uppercase font-mono tracking-widest border"
                style={{
                  background: `linear-gradient(135deg, ${vehicle.badgeColor}22, ${vehicle.badgeColor}44)`,
                  borderColor: `${vehicle.badgeColor}55`,
                  color: vehicle.badgeColor,
                }}
              >
                {vehicle.badge}
              </div>
            </div>
            {vehicle.battery.soh_percent < 85 && (
              <div className="mt-2 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-tertiary-container/10 border border-tertiary-container/25 text-tertiary-container text-[10px] font-mono font-bold uppercase tracking-wide max-w-xs">
                ⚠ Battery degradation detected ({vehicle.battery.soh_percent}% SoH)
              </div>
            )}
          </div>

          {/* Carousel area */}
          <div className={`relative w-full flex items-center justify-center ${isMobile ? 'h-[260px]' : 'h-[400px]'}`}>
            {/* Ghost left */}
            <div className={`absolute -left-20 opacity-20 scale-75 blur-sm grayscale pointer-events-none${isMobile ? ' hidden' : ''}`}>
              <img
                src={prevVehicle.carImage}
                alt={prevVehicle.name}
                className="w-[400px] object-contain"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </div>

            {/* Active vehicle */}
            <div className="relative z-10 flex flex-col items-center w-full">
              {/* Telemetry overlays */}
              {!isMobile && (
                <>
                  <div className="absolute top-1/4 -left-4 flex flex-col items-end pointer-events-none z-20">
                    <span className="font-mono text-[10px] text-primary uppercase mb-1 drop-shadow">Aero Winglet</span>
                    <div className="w-24 h-px bg-gradient-to-r from-primary to-transparent" />
                  </div>
                  <div className="absolute bottom-1/3 -right-4 flex flex-col items-start pointer-events-none z-20">
                    <span className="font-mono text-[10px] text-primary uppercase mb-1 drop-shadow">Rear Motor Array</span>
                    <div className="w-32 h-px bg-gradient-to-l from-primary to-transparent" />
                  </div>
                </>
              )}

              {/* 3D Model or Lab SVG */}
              <div
                className="w-full flex items-center justify-center relative"
                style={{ height: isMobile ? '200px' : '320px' }}
                data-vehicle={vehicle.id}
              >
                <CarModel
                  batteryKwh={vehicle.battery.capacity_kwh}
                  tempC={vehicle.battery.temperature_c}
                  glowColor={vehicle.color}
                  modelPath={vehicle.isCustom ? '/models/custom car.glb' : vehicle.modelPath}
                  regenActive={vehicle.realtime.regen_active}
                  maxPowerKw={vehicle.specs.max_power_kw}
                />
              </div>

              {/* Model name */}
              <div className="text-center space-y-1 mt-2">
                <h2 className={`${isMobile ? 'text-2xl' : 'text-4xl'} font-black italic tracking-tighter text-on-surface drop-shadow-lg`}>
                  {modelNames[currentVehicle] ?? 'MODEL V'}
                </h2>
                <p className="font-label text-[10px] uppercase tracking-[0.3em] text-primary">
                  {vehicle.subtitle}
                </p>
              </div>
            </div>

            {/* Ghost right */}
            <div className={`absolute -right-20 opacity-20 scale-75 blur-sm grayscale pointer-events-none${isMobile ? ' hidden' : ''}`}>
              <img
                src={nextVehicle.carImage}
                alt={nextVehicle.name}
                className="w-[400px] object-contain"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </div>
          </div>

          {/* Carousel dots */}
          <div className="absolute bottom-6 flex gap-3">
            {VEHICLE_ORDER.map((id, i) => (
              <span
                key={id}
                className={`h-1 rounded-full transition-all duration-300 ${
                  i === idx
                    ? 'w-12 bg-primary shadow-[0_0_8px_#afecff]'
                    : 'w-8 bg-white/10'
                }`}
              />
            ))}
          </div>
        </section>

        {/* Right Panel – Stacked Cards (45%) */}
        <section className={`flex flex-col gap-5 ${isMobile ? 'w-full' : 'w-[45%] overflow-y-auto no-scrollbar'}`}>

          {/* SoC Card */}
          <div className="bg-surface-container rounded-2xl p-6 flex items-center justify-between group hover:bg-surface-variant transition-all duration-300">
            <div className="space-y-1">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Battery Capacity</h3>
              <div className="flex items-baseline gap-2">
                <AnimatedNumber
                  value={vehicle.battery.soc_percent}
                  duration={2000}
                  className="font-mono text-5xl font-bold text-on-surface"
                />
                <span className="font-mono text-xl text-primary">%</span>
              </div>
              <div className="flex flex-col gap-1 mt-2">
                <div className="flex justify-between text-xs font-mono text-on-surface-variant">
                  <span>Health (SoH)</span>
                  <span className={vehicle.battery.soh_percent < 85 ? 'text-tertiary-container' : 'text-secondary-container'}>
                    {vehicle.battery.soh_percent}%
                  </span>
                </div>
                <div className="flex justify-between text-xs font-mono text-on-surface-variant">
                  <span>Range</span>
                  <span className="text-on-surface">{vehicle.range_km} km</span>
                </div>
              </div>
            </div>
            {/* Circular progress ring */}
            <div className="relative w-24 h-24 shrink-0">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 96 96">
                <circle
                  className="text-surface-container-highest"
                  cx="48" cy="48" r={radius}
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="8"
                />
                <circle
                  className="text-primary group-hover:text-secondary-container transition-colors duration-300"
                  cx="48" cy="48" r={radius}
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="8"
                  strokeDasharray={`${circumference}`}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span
                  className="material-symbols-outlined text-primary text-3xl"
                  style={{ fontVariationSettings: "'FILL' 1" }}
                >
                  bolt
                </span>
              </div>
            </div>
          </div>

          {/* Nearby Charger Card */}
          <div className="bg-surface-container rounded-2xl p-6 flex items-center gap-5 hover:bg-surface-variant transition-all duration-300 relative overflow-hidden">
            <div className="absolute top-2 right-2 p-2 opacity-10">
              <span className="material-symbols-outlined text-6xl">ev_station</span>
            </div>
            <div className="w-14 h-14 bg-surface-container-highest rounded-xl flex items-center justify-center shrink-0">
              <span className="material-symbols-outlined text-secondary-container">near_me</span>
            </div>
            <div className="flex-1 space-y-1 min-w-0">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Nearby Charger</h3>
              <p className="text-lg font-bold text-on-surface truncate">Tesla M2 - Supercharger</p>
              <div className="flex items-center gap-3 font-mono text-[11px]">
                <span className="text-secondary-container">1.2 MILES</span>
                <span className="text-on-surface-variant">•</span>
                <span className="text-on-surface-variant">4 SLOTS AVAIL.</span>
              </div>
            </div>
            <button
              onClick={() => navigate('/route-planner', { state: { destination: 'Tesla Supercharger M2' } })}
              className="w-10 h-10 rounded-full bg-primary-fixed-dim hover:brightness-110 flex items-center justify-center shrink-0 transition-all shadow-[0_0_15px_rgba(0,217,255,0.3)] cursor-pointer"
            >
              <span className="material-symbols-outlined text-on-primary text-[18px]">directions</span>
            </button>
          </div>

          {/* Sentry Mode Card */}
          <div className="bg-surface-container rounded-2xl p-6 flex items-center justify-between hover:bg-surface-variant transition-all duration-300">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="w-12 h-12 bg-surface-container-highest rounded-full flex items-center justify-center">
                  <span className="material-symbols-outlined text-white">security</span>
                </div>
                <span className="absolute top-0 right-0 w-3 h-3 bg-red-500 rounded-full border-2 border-surface-container animate-pulse" />
              </div>
              <div className="space-y-1">
                <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Security Status</h3>
                <p className="text-lg font-bold text-on-surface">Sentry Mode Active</p>
              </div>
            </div>
            <div className="font-mono text-[11px] text-on-surface-variant text-right">
              3 EVENTS<br />LOGGED
            </div>
          </div>

          {/* Climate Control Card */}
          <div className="bg-surface-container rounded-2xl p-6 flex flex-col gap-4 hover:bg-surface-variant transition-all duration-300">
            <div className="flex justify-between items-center">
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant">Climate Control</h3>
              <span className="font-mono text-lg font-bold text-on-surface">
                {(vehicle.realtime.cabin_hvac_kw * 6 + 18).toFixed(1)}°C
              </span>
            </div>
            <div className="h-2 w-full bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary to-secondary-container transition-all duration-500"
                style={{ width: `${Math.min(100, (vehicle.realtime.cabin_hvac_kw / 4) * 100)}%` }}
              />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-on-surface-variant uppercase tracking-tighter">
              <span>Min (16)</span>
              <span>Auto Active • {vehicle.realtime.cabin_hvac_kw} kW</span>
              <span>Max (28)</span>
            </div>
          </div>

        </section>
      </div>

      {/* ── Bottom: Energy Consumption Chart ── */}
      <section className="px-3 pb-3 md:px-6 md:pb-6">
        <div className="bg-surface-container-low rounded-2xl p-6 h-52 flex flex-col">
          <div className="flex justify-between items-end mb-4 px-1">
            <div>
              <h3 className="font-label text-[10px] uppercase tracking-widest text-on-surface-variant mb-1 ml-0.5">
                Energy Consumption
              </h3>
              <p className="font-mono text-xl font-bold text-on-surface">
                {vehicle.realtime.efficiency_wh_per_km.toFixed(1)}{' '}
                <span className="text-secondary text-sm font-normal">Wh/km</span>
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-primary rounded-full" />
                <span className="font-label text-[10px] uppercase text-on-surface-variant">Drive</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-secondary-container rounded-full" />
                <span className="font-label text-[10px] uppercase text-on-surface-variant">Climate</span>
              </div>
              {/* Time range buttons */}
              <div className="flex gap-1.5 ml-4">
                {(['1H', '6H', '24H'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setActiveRange(t)}
                    className={`text-[9px] font-bold px-3 py-1.5 rounded-lg transition-colors ${
                      t === activeRange
                        ? 'bg-primary/20 text-primary border border-primary/30'
                        : 'text-on-surface-variant hover:text-on-surface bg-surface-container-high'
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Bar Chart */}
          <div className="flex-1 flex items-end justify-between gap-1.5 px-2">
            {BAR_HEIGHTS.map((h, i) => (
              <div
                key={i}
                className={`w-full rounded-t-lg transition-all duration-300 hover:brightness-125 ${
                  BAR_ACCENTS.has(i)
                    ? 'bg-primary'
                    : BAR_SECONDARY.has(i)
                    ? 'bg-secondary-container'
                    : 'bg-surface-container-highest hover:bg-primary/40'
                }`}
                style={{ height: `${h}%` }}
              />
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
