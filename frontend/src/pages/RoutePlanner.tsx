import { useState, useEffect, useRef } from 'react'
import { useLocation } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* Amsterdam-area charging stations for route overlay */
const ROUTE_STATIONS = [
  { id: 1, name: 'FastNed Amstel', lat: 52.3469, lng: 4.9179, kW: 300, operator: 'FastNed' },
  { id: 2, name: 'Shell Recharge Centrum', lat: 52.3714, lng: 4.8584, kW: 50, operator: 'Shell' },
  { id: 3, name: 'Allego P+R Sloterdijk', lat: 52.3890, lng: 4.8375, kW: 150, operator: 'Allego' },
  { id: 4, name: 'Tesla Supercharger Schiphol', lat: 52.3056, lng: 4.7581, kW: 250, operator: 'Tesla' },
  { id: 5, name: 'IONITY A4 Hoofddorp', lat: 52.3060, lng: 4.6870, kW: 350, operator: 'IONITY' },
  { id: 6, name: 'FastNed A10 West', lat: 52.3760, lng: 4.8100, kW: 300, operator: 'FastNed' },
  { id: 7, name: 'GreenFlux Zuidas', lat: 52.3364, lng: 4.8738, kW: 50, operator: 'GreenFlux' },
  { id: 8, name: 'Allego Centraal Station', lat: 52.3791, lng: 4.9003, kW: 50, operator: 'Allego' },
  { id: 9, name: 'FastNed Zaandam', lat: 52.4418, lng: 4.8263, kW: 300, operator: 'FastNed' },
  { id: 10, name: 'Shell Recharge Bijlmer', lat: 52.3120, lng: 4.9480, kW: 150, operator: 'Shell' },
]

/* Route waypoints for the demo route line */
const ROUTE_POINTS: [number, number][] = [
  [52.3791, 4.9003],   // Amsterdam Centraal
  [52.3676, 4.9041],   // Dam area
  [52.3530, 4.8630],   // Vondelpark
  [52.3364, 4.8738],   // Zuidas
  [52.3056, 4.7581],   // Schiphol
  [52.2700, 4.7400],   // Destination
]

const START = ROUTE_POINTS[0]
const END = ROUTE_POINTS[ROUTE_POINTS.length - 1]

// Blade Runner styled marker icons
function createStationIcon() {
  return L.divIcon({
    className: '',
    html: `<div style="width:24px;height:24px;background:rgba(10,14,23,0.8);backdrop-filter:blur(4px);border-radius:50%;border:2px solid #00b4d8;display:flex;align-items:center;justify-content:center;box-shadow:0 0 10px rgba(0,180,216,0.5), inset 0 0 8px rgba(0,180,216,0.3);font-size:12px;color:#00b4d8;">⚡</div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24],
  })
}

function createEndpointIcon(label: string, color: string) {
  return L.divIcon({
    className: '',
    html: `<div style="width:32px;height:32px;background:rgba(10,14,23,0.8);backdrop-filter:blur(4px);border-radius:50%;border:2px solid ${color};display:flex;align-items:center;justify-content:center;box-shadow:0 0 15px ${color}80, inset 0 0 10px ${color}40;font-size:12px;font-weight:bold;color:${color};">${label}</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  })
}

function LocationAutocomplete({ placeholder, defaultValue, color, onSelect }: any) {
  const [query, setQuery] = useState(defaultValue)
  const [results, setResults] = useState<any[]>([])
  const [isOpen, setIsOpen] = useState(false)
  
  useEffect(() => {
    if (!query || query === defaultValue) {
      setResults([])
      return
    }
    const timer = setTimeout(() => {
      fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`)
        .then(r => r.json())
        .then(data => setResults(data || []))
        .catch(() => {})
    }, 500)
    return () => clearTimeout(timer)
  }, [query])

  return (
    <div className="relative w-full">
      <input
        className={`w-full bg-surface-100/50 border border-white/10 text-surface-900 text-sm px-4 py-3.5 rounded-lg outline-none focus:bg-surface-200/50 transition-all shadow-inner ${
          color === 'blue' ? 'focus:border-neon-blue/50' : 'focus:border-neon-green/50'
        }`}
        value={query}
        placeholder={placeholder}
        aria-label={placeholder}
        aria-expanded={isOpen}
        onChange={(e) => {
          setQuery(e.target.value)
          setIsOpen(true)
        }}
        onBlur={() => setTimeout(() => setIsOpen(false), 200)}
        onFocus={() => { if (results.length > 0) setIsOpen(true) }}
        type="text"
      />
      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-surface-100/90 backdrop-blur-md border border-white/10  shadow-2xl z-[2000] overflow-hidden">
          {results.map((r: any) => (
            <div 
              key={r.place_id} 
              className="px-4 py-3 hover:bg-surface-200/50 cursor-pointer border-b border-white/5 last:border-0"
              onClick={() => {
                const title = r.display_name.split(',')[0]
                setQuery(title)
                setIsOpen(false)
                if (onSelect) onSelect(r)
              }}
            >
              <p className="text-sm font-semibold text-surface-900 truncate">{r.display_name.split(',')[0]}</p>
              <p className="text-[10px] text-surface-800/50 truncate mt-0.5">{r.display_name}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


export default function RoutePlanner() {
  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<L.Map | null>(null)
  const [showStations, setShowStations] = useState(true)
  const location = useLocation()
  const defaultDest = location.state?.destination || 'Hoofddorp Zuid'

  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return

    const map = L.map(mapRef.current, {
      center: [52.34, 4.84],
      zoom: 12,
      zoomControl: false,
    })

    // Blade Runner dark tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'topright' }).addTo(map)

    // Neon route polyline
    L.polyline(ROUTE_POINTS, {
      color: '#00E5CC',
      weight: 4,
      opacity: 0.8,
      className: 'route-line-glow', // Will be styled in css if needed
    }).addTo(map)

    // Start marker (Neon Blue)
    L.marker(START, { icon: createEndpointIcon('A', '#00b4d8') })
      .addTo(map)
      .bindPopup('<div class="bg-surface-900 border border-neon-blue/30 text-white p-2 "><b>Start:</b> Amsterdam Centraal</div>')

    // End marker (Neon Green)
    L.marker(END, { icon: createEndpointIcon('B', '#00f5a0') })
      .addTo(map)
      .bindPopup('<div class="text-white"><b>Destination:</b> Hoofddorp Zuid</div>')

    // Charging station markers
    ROUTE_STATIONS.forEach((s) => {
      L.marker([s.lat, s.lng], { icon: createStationIcon() })
        .addTo(map)
        .bindPopup(`<b>${s.name}</b><br/>${s.kW} kW • ${s.operator}`)
    })

    const bounds = L.latLngBounds(ROUTE_POINTS)
    map.fitBounds(bounds, { padding: [60, 60] })

    leafletMap.current = map

    return () => {
      map.remove()
      leafletMap.current = null
    }
  }, [])

  return (
    <div className="flex-1 flex h-full overflow-hidden relative">

      {/* ═══ Full-width Map ═══ */}
      <div className="flex-1 relative bg-brand-bg">
        <div ref={mapRef} className="h-full w-full z-0 opacity-90" />
        <div className="absolute inset-0 z-[5] pointer-events-none shadow-[inset_0_0_150px_rgba(10,14,23,1)]" />

        {/* ── Top search bar overlay ── */}
        <div className="absolute top-5 left-5 right-5 lg:left-[420px] lg:right-auto lg:w-[400px] z-[1000]">
          <div className="glass-dark  p-1 flex items-center gap-2 border border-neon-blue/20">
            <div className="w-10 h-10 rounded-lg bg-neon-blue/10 flex items-center justify-center shrink-0 ml-1">
              <span className="material-symbols-outlined text-neon-blue">search</span>
            </div>
            <input
              className="flex-1 bg-transparent text-surface-900 text-sm py-3 pr-4 outline-none placeholder:text-surface-800/40 rounded-lg"
              placeholder="Search places, addresses…"
              type="text"
            />
          </div>
        </div>

        {/* ── Map layer toggle ── */}
        <div className="absolute top-5 right-5 z-[1000] flex flex-col gap-2">
          <button
            aria-label="Toggle Charging Stations Layer"
            aria-pressed={showStations}
            onClick={() => setShowStations(!showStations)}
            className={`px-3 py-2 rounded-lg text-xs font-bold font-mono tracking-widest uppercase flex items-center gap-2 transition-all duration-300 ${
              showStations
                ? 'glass-dark text-neon-blue border border-neon-blue/40 shadow-[0_0_15px_rgba(0,180,216,0.2)]'
                : 'bg-surface-200/50 backdrop-blur text-surface-800/40 border border-white/5'
            }`}
          >
            <span className="material-symbols-outlined text-sm" aria-hidden="true">ev_station</span>
            Chargers
          </button>
        </div>

        {/* ══ Energy Prediction Overlay ══ */}
        <div className="absolute bottom-5 right-5 w-[320px] z-[1000]" style={{ animation: 'slideUp 0.5s ease-out' }}>
          <div className="glass-dark  p-6 border border-neon-blue/15 relative overflow-hidden">
            <div className="absolute -top-10 -right-10 w-32 h-32 bg-accent-success/10 rounded-full blur-[30px]" />
            
            <div className="flex items-center justify-between mb-4 relative z-10">
              <h4 className="text-[10px] font-mono font-bold text-surface-800/60 uppercase tracking-widest">Energy Prediction</h4>
              <span className="px-2 py-0.5 bg-accent-success/10 border border-accent-success/20 text-accent-success text-[9px] rounded-full font-mono font-bold uppercase tracking-widest shadow-[0_0_10px_rgba(0,245,160,0.2)]">Optimized</span>
            </div>

            <div className="mb-5 relative z-10">
              <div className="flex justify-between items-end mb-1.5">
                <span className="text-xs font-mono text-surface-800/50">Energy Needed</span>
                <span className="text-xl font-headline font-bold text-neon-blue glow-neon">12.3 <span className="text-[10px] font-sans text-neon-blue/50">kWh</span></span>
              </div>
              <div className="w-full bg-surface-200/50 h-1.5 rounded-full overflow-hidden">
                <div className="bg-gradient-to-r from-neon-blue to-neon-green h-full rounded-full shadow-[0_0_10px_#00f5a0]" style={{ width: '45%' }} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-5 relative z-10">
              <div className="bg-surface-100/50 backdrop-blur p-3  border border-white/5">
                <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">Arrival SoC</p>
                <p className="text-xl font-headline font-bold text-accent-success glow-neon">64%</p>
              </div>
              <div className="bg-surface-100/50 backdrop-blur p-3  border border-white/5">
                <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">Confidence</p>
                <div className="flex items-center gap-1.5">
                  <p className="text-xl font-headline font-bold text-surface-900">87%</p>
                  <span className="material-symbols-outlined text-accent-success text-sm drop-shadow-[0_0_5px_#00f5a0]">verified</span>
                </div>
              </div>
            </div>

            <div className="flex gap-2 relative z-10">
              <button className="flex-1 text-xs font-bold text-surface-900 py-3 rounded-lg bg-surface-100/80 hover:bg-surface-200 transition-colors border border-white/5" aria-label="View Route Details">
                Details
              </button>
              <button className="flex-1 text-xs font-bold text-brand-bg py-3 rounded-lg bg-neon-green border border-neon-green hover:bg-[#00d68b] transition-colors shadow-[0_0_15px_rgba(0,245,160,0.3)] tracking-wide" aria-label="Start Navigation">
                Start Nav
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ Left Panel — Route Config ═══ */}
      <div className="absolute top-0 left-0 bottom-0 w-[420px] z-[1000] flex flex-col pointer-events-none p-5">
        <div className="flex-1 glass-dark  overflow-hidden flex flex-col pointer-events-auto border border-neon-blue/20" style={{ animation: 'slideRight 0.4s ease-out' }}>

          <div className="p-6 pb-4 relative overflow-hidden shrink-0">
            <div className="absolute top-0 right-0 w-32 h-32 bg-neon-blue/10 rounded-full blur-[40px] -mr-10 -mt-10" />
            <h3 className="text-2xl font-headline font-bold text-surface-900 mb-1 relative z-10">Route Control</h3>
            <p className="text-[10px] font-mono text-neon-blue/60 tracking-widest uppercase relative z-10">Neural Navigation Active</p>
          </div>

          <div className="px-6 pb-5 shrink-0">
            <div className="flex gap-4">
              <div className="flex flex-col items-center py-3 shrink-0">
                <div className="w-3.5 h-3.5 rounded-full bg-surface-50 border-[3px] border-neon-blue shadow-[0_0_10px_rgba(0,180,216,0.6)] z-10" />
                <div className="w-[1px] flex-1 bg-gradient-to-b from-neon-blue via-surface-200 to-neon-green my-1" />
                <div className="w-3.5 h-3.5 rounded-full bg-surface-50 border-[3px] border-neon-green shadow-[0_0_10px_rgba(0,245,160,0.6)] z-10" />
              </div>

              <div className="flex-1 space-y-3">
                <LocationAutocomplete placeholder="Start point..." defaultValue="Amsterdam Centraal" color="blue" />
                <LocationAutocomplete placeholder="Destination..." defaultValue={defaultDest} color="green" />
              </div>
            </div>
          </div>

          <div className="p-6 pt-0 flex-1 overflow-y-auto custom-scrollbar">
            <div className="grid grid-cols-3 gap-3 mb-6">
              <div className="text-center p-3 glass-panel  border-t border-neon-blue/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-blue/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-blue/60 uppercase tracking-widest relative z-10">Distance</p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">14.2<span className="text-[10px] ml-0.5 text-surface-800/40">km</span></p>
              </div>
              <div className="text-center p-3 glass-panel  border-t border-neon-purple/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-purple/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-purple/60 uppercase tracking-widest relative z-10">Duration</p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">22<span className="text-[10px] ml-0.5 text-surface-800/40">min</span></p>
              </div>
              <div className="text-center p-3 glass-panel  border-t border-neon-green/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-green/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-green/60 uppercase tracking-widest relative z-10">Energy</p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">2.1<span className="text-[10px] ml-0.5 text-surface-800/40">kWh</span></p>
              </div>
            </div>

            <div className="space-y-4 relative">
              <div className="absolute left-4 top-2 bottom-6 w-px bg-surface-200/30 z-0" />
              {[
                { icon: 'trip_origin', text: 'Amsterdam Centraal', sub: 'Start point', dist: '' },
                { icon: 'turn_right', text: 'Damrak → Rokin', sub: 'Head south on S100', dist: '1.8 km' },
                { icon: 'turn_left', text: 'A10 Ring West', sub: 'Merge onto motorway', dist: '5.4 km' },
                { icon: 'ev_station', text: 'FastNed A10 West', sub: '300 kW available', dist: '7.1 km', highlight: true },
                { icon: 'turn_right', text: 'A4 → Exit Hoofddorp', sub: 'Take exit 3', dist: '12.8 km' },
                { icon: 'flag', text: 'Hoofddorp Zuid', sub: 'Destination', dist: '14.2 km' },
              ].map((step, i) => (
                <div key={i} className="flex items-start gap-4 relative z-10 group">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border border-brand-bg transition-all ${
                    step.highlight
                      ? 'bg-neon-blue/20 text-neon-blue border-[1.5px] border-neon-blue shadow-[0_0_10px_rgba(0,180,216,0.3)]'
                      : 'bg-surface-100 text-surface-800/50 group-hover:bg-surface-200 group-hover:text-surface-900'
                  }`}>
                    <span className="material-symbols-outlined text-[16px]">{step.icon}</span>
                  </div>
                  <div className="flex-1 min-w-0 pt-1">
                    <p className={`text-sm font-semibold tracking-wide ${step.highlight ? 'text-neon-blue' : 'text-surface-900'}`}>{step.text}</p>
                    <p className="text-[10px] font-mono text-surface-800/40 mt-0.5">{step.sub}</p>
                  </div>
                  {step.dist && (
                    <span className="text-[10px] font-mono text-surface-800/30 shrink-0 pt-1.5">{step.dist}</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="p-6 bg-surface-50/50 border-t border-white/5 shrink-0 backdrop-blur-md">
            <button className="w-full bg-gradient-to-r from-neon-blue to-neon-green hover:from-[#00c5eb] hover:to-[#17ffae] text-brand-bg font-extrabold tracking-widest uppercase py-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 shadow-[0_4px_20px_rgba(0,180,216,0.25)] relative overflow-hidden group" aria-label="Start Sequence Vector Calculation">
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" aria-hidden="true" />
              <span className="material-symbols-outlined text-xl drop-shadow-md relative z-10" aria-hidden="true">navigation</span>
              <span className="relative z-10">Start Sequence</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

