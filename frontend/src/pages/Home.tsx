import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import CarModel from '../components/CarModel'
import AnimatedNumber from '../components/AnimatedNumber'

function createMiniPinIcon() {
  return L.divIcon({
    className: '',
    html: `<div style="width:24px;height:24px;background:rgba(10,14,23,0.8);backdrop-filter:blur(4px);border-radius:50%;border:2px solid #00E5CC;display:flex;align-items:center;justify-content:center;box-shadow:0 0 10px rgba(0,229,204,0.5);font-size:12px;color:#00E5CC">⚡</div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 20],
  })
}

export default function Home() {
  const mapRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!mapRef.current) return
    const map = L.map(mapRef.current, {
      center: [52.3469, 4.9179], // FastNed Amstel
      zoom: 14,
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      keyboard: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
    })

    // Dark CARTO tiles for Blade Runner theme
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.marker([52.3469, 4.9179], { icon: createMiniPinIcon() }).addTo(map)

    return () => { map.remove() }
  }, [])

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden p-4 md:p-6 gap-4 md:gap-6 stagger-children pt-4">
      
      {/* ═══ Top Row (3 Columns) ═══ */}
      <div className="flex-[4] flex gap-4 md:gap-6 min-h-0 flex-col md:flex-row">
        
        {/* Column 1: 3D Model Area */}
        <div className="flex-[4] relative glass-dark  overflow-hidden card-hover min-h-[300px] md:min-h-0">
          <div className="absolute inset-0 z-10 pointer-events-none" style={{
            background: 'radial-gradient(circle at center, transparent 40%, rgba(10, 14, 23, 0.4) 100%)'
          }} />
          <div className="absolute top-8 left-8 z-20 max-w-[80%]">
            <h2 className="text-2xl font-headline font-bold text-surface-900 tracking-tight glow-neon inline-block">
              Model V - Performance
            </h2>
            <div className="flex items-center gap-2 mt-2">
              <span className="w-2 h-2 rounded-full bg-accent-success animate-pulse glow-neon" />
              <span className="text-[10px] font-mono text-accent-success uppercase tracking-widest">Aero Mode Active</span>
            </div>
          </div>
          <CarModel />
        </div>

        {/* Column 2: Right Info Column (SoC + Sentry) */}
        <div className="flex-[2] flex flex-col gap-4 md:gap-6 shrink-0 md:min-w-[260px]">
          
          {/* SoC Card */}
          <div className="glass-dark  p-5 card-hover relative overflow-hidden flex-1 flex flex-col">
            <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-blue/20 rounded-full blur-[40px]" />
            <div className="flex justify-between items-start mb-auto relative z-10">
              <span className="text-[10px] font-mono font-bold text-surface-800/60 uppercase tracking-widest">State of Charge</span>
            </div>
            <div className="relative z-10 mt-3">
              <div className="flex items-baseline gap-1">
                <AnimatedNumber value={76} duration={2000} className="text-7xl font-headline font-bold text-white tabular-nums tracking-tighter" />
                <span className="text-3xl font-bold text-neon-blue">%</span>
                <span className="material-symbols-outlined text-neon-blue/80 text-3xl ml-2 drop-shadow-[0_0_8px_rgba(0,180,216,0.6)] animate-pulse-slow">bolt</span>
              </div>
            </div>
            <div className="mt-5 space-y-3 relative z-10">
              <div className="flex justify-between items-baseline border-b border-white/5 pb-3">
                <span className="text-xs text-surface-800/60 font-mono">Health (SoH)</span>
                <span className="text-sm font-bold text-accent-success"><AnimatedNumber value={94} duration={1500} suffix="%" /></span>
              </div>
              <div className="flex justify-between items-baseline border-b border-white/5 pb-3">
                <span className="text-xs text-surface-800/60 font-mono">Est. Range</span>
                <span className="text-sm font-bold text-white tabular-nums"><AnimatedNumber value={312} duration={2000} /> <span className="text-[10px] text-surface-800/40 ml-1">km</span></span>
              </div>
              <div className="flex justify-between items-baseline border-b border-white/5 pb-3">
                <span className="text-xs text-surface-800/60 font-mono">Battery Core</span>
                <span className="text-sm font-bold text-white tabular-nums">29°C</span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-surface-800/60 font-mono">Power Draw</span>
                <span className="text-sm font-bold text-neon-blue tabular-nums">4.3 <span className="text-[10px] text-surface-800/40 ml-1">kW</span></span>
              </div>
            </div>
          </div>

          {/* Sentry Mode */}
          <div className="glass-dark  p-5 flex items-center gap-4 card-hover shadow-md">
            <div className="w-12 h-12  bg-neon-red/10 border border-neon-red/20 flex items-center justify-center text-neon-red shadow-[0_0_15px_rgba(255,62,108,0.2)]">
              <span className="material-symbols-outlined">shield</span>
            </div>
            <div>
              <h4 className="text-sm font-bold text-surface-900">Sentry Mode</h4>
              <p className="text-[10px] text-surface-800/50 mt-0.5">Active • 2 Events Logged</p>
            </div>
          </div>
        </div>

        {/* Column 3: Nearby Charger with Mini-Map */}
        <div className="flex-[3] glass-dark  p-2 overflow-hidden card-hover relative group md:min-w-[280px]">
          <div className="absolute inset-2  overflow-hidden z-0 pointer-events-none">
            {/* The mini map container */}
            <div ref={mapRef} className="w-full h-full opacity-60 mix-blend-screen scale-125 transition-transform duration-1000 group-hover:scale-110" />
            <div className="absolute inset-0 bg-gradient-to-t from-brand-bg via-brand-bg/80 to-transparent" />
          </div>

          <div className="relative z-10 h-full p-6 flex flex-col justify-between">
            <div className="mt-2 ml-1">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-white/5 backdrop-blur-md border border-white/10 rounded-full mb-3">
                <span className="w-1.5 h-1.5 rounded-full bg-neon-green animate-pulse" />
                <span className="text-[9px] font-mono font-bold text-white uppercase tracking-widest">Nearby Charger</span>
              </div>
              <h3 className="text-xl font-bold text-surface-900 flex items-center gap-2 drop-shadow-md">
                Tesla M2
              </h3>
            </div>
            
            <div className="bg-surface-200/50 backdrop-blur-lg  p-5 border border-white/10 group-hover:border-neon-blue/30 transition-colors mb-1 shadow-xl">
              <div className="flex justify-between items-end">
                <div>
                  <p className="text-[10px] font-mono text-neon-blue/90 uppercase tracking-widest mb-1 font-bold shadow-sm">Recommended</p>
                  <p className="text-base font-bold text-white">Tesla Supercharger M2</p>
                  <p className="text-xs text-surface-800/80 mt-0.5 font-medium">250 kW • CCS2</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-white/80 mb-2 font-mono font-bold">1.2 km away</p>
                  <button onClick={() => navigate('/route-planner', { state: { destination: 'Tesla Supercharger M2' } })} className="w-11 h-11 rounded-full bg-neon-blue hover:bg-[#00c5eb] text-brand-bg border border-neon-blue flex items-center justify-center transition-colors shadow-[0_0_15px_rgba(0,180,216,0.4)] cursor-pointer">
                    <span className="material-symbols-outlined text-[20px]">directions</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* ═══ Bottom Row ═══ */}
      <div className="flex-[3] flex min-h-0">
        
        {/* Efficiency Chart */}
        <div className="flex-1 glass-dark  p-7 card-hover flex flex-col relative overflow-hidden">
          <div className="absolute top-0 right-0 w-full h-full bg-gradient-to-t from-transparent via-transparent to-neon-blue/5 pointer-events-none" />
          
          <div className="flex justify-between items-start mb-6 relative z-10 w-full">
            <h3 className="flex items-center gap-2 text-[10px] font-mono font-bold text-surface-800/60 uppercase tracking-widest">
              <span className="material-symbols-outlined text-base text-neon-blue glow-neon">show_chart</span>
              Energy Efficiency
            </h3>
            <div className="flex gap-1.5">
              {['1H', '6H', '24H'].map((t) => (
                <button
                  key={t}
                  className={`text-[9px] font-bold px-3 py-1.5 rounded-lg transition-colors ${
                    t === '6H' ? 'bg-neon-blue/20 text-neon-blue border border-neon-blue/30' : 'text-surface-800/40 hover:text-surface-900 bg-surface-100'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Simulated Bar Chart */}
          <div className="flex-1 flex items-end justify-between gap-3 px-2 pb-2 relative z-10 w-full">
            {[20, 30, 15, 25, 35, 20, 22, 18, 25, 38, 22, 23, 19, 21, 28, 20, 19, 21, 28].map((val, i) => (
              <div key={i} className="flex-1 flex justify-center group h-full items-end">
                <div
                  className="w-full max-w-[40px]  bg-gradient-to-t from-neon-blue/20 to-neon-blue/60 transition-all duration-500 group-hover:from-neon-blue/40 group-hover:to-neon-blue/90 relative"
                  style={{ height: `${val + 10}%` }}
                >
                  <div className="opacity-0 group-hover:opacity-100 absolute -top-7 left-1/2 -translate-x-1/2 text-[10px] font-mono text-neon-blue font-bold transition-opacity bg-surface-200/80 px-1.5 py-0.5 rounded-md">
                    {val * 3}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-8 mt-4 pt-4 border-t border-white/5 relative z-10 w-full">
            <div>
              <p className="text-[10px] font-mono text-surface-800/50 uppercase">Average</p>
              <p className="text-xl font-headline font-bold text-surface-900 mt-1">142 <span className="text-[10px] text-surface-800/50 font-sans tracking-wide">Wh/km</span></p>
            </div>
            <div>
              <p className="text-[10px] font-mono text-neon-purple/70 uppercase">Optimal</p>
              <p className="text-xl font-headline font-bold text-neon-purple mt-1 drop-shadow-[0_0_8px_rgba(123,47,247,0.5)]">118 <span className="text-[10px] text-neon-purple/50 font-sans tracking-wide">Wh/km</span></p>
            </div>
            <div className="ml-auto self-end">
              <p className="text-[9px] font-mono text-surface-800/30 uppercase tracking-widest">Updated: 2 Sec Ago</p>
            </div>
          </div>
        </div>

      </div>
      
    </div>
  )
}

