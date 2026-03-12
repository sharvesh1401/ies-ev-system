import { useState, useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

interface Station {
  id: number
  name: string
  lat: number
  lng: number
  town: string
  operator: string
  maxKW: number | null
  connectors: number
  isOperational: boolean
}

/* Netherlands center: Amsterdam */
const NL_CENTER: [number, number] = [52.3676, 4.9041]

/* Amsterdam area charging stations */
const STATIONS: Station[] = [
  { id: 1, name: 'FastNed Amstel', lat: 52.3469, lng: 4.9179, town: 'Amsterdam', operator: 'FastNed', maxKW: 300, connectors: 4, isOperational: true },
  { id: 2, name: 'Shell Recharge Jan van Galenstraat', lat: 52.3714, lng: 4.8584, town: 'Amsterdam', operator: 'Shell Recharge', maxKW: 50, connectors: 2, isOperational: true },
  { id: 3, name: 'IONITY A2 Breukelen', lat: 52.1650, lng: 4.9850, town: 'Breukelen', operator: 'IONITY', maxKW: 350, connectors: 6, isOperational: true },
  { id: 4, name: 'Allego P+R Sloterdijk', lat: 52.3890, lng: 4.8375, town: 'Amsterdam', operator: 'Allego', maxKW: 150, connectors: 4, isOperational: true },
  { id: 5, name: 'EVBox Amstelveenseweg', lat: 52.3530, lng: 4.8630, town: 'Amsterdam', operator: 'EVBox', maxKW: 22, connectors: 2, isOperational: true },
  { id: 6, name: 'Tesla Supercharger Schiphol', lat: 52.3056, lng: 4.7581, town: 'Schiphol', operator: 'Tesla', maxKW: 250, connectors: 12, isOperational: true },
  { id: 7, name: 'GreenFlux Zuidas', lat: 52.3364, lng: 4.8738, town: 'Amsterdam', operator: 'GreenFlux', maxKW: 50, connectors: 3, isOperational: true },
  { id: 8, name: 'NewMotion Leidseplein', lat: 52.3640, lng: 4.8812, town: 'Amsterdam', operator: 'NewMotion', maxKW: 22, connectors: 2, isOperational: false },
  { id: 9, name: 'FastNed A10 West', lat: 52.3760, lng: 4.8100, town: 'Amsterdam', operator: 'FastNed', maxKW: 300, connectors: 4, isOperational: true },
  { id: 10, name: 'Allego Centraal Station', lat: 52.3791, lng: 4.9003, town: 'Amsterdam', operator: 'Allego', maxKW: 50, connectors: 4, isOperational: true },
  { id: 11, name: 'Shell Recharge Bijlmer', lat: 52.3120, lng: 4.9480, town: 'Amsterdam', operator: 'Shell Recharge', maxKW: 150, connectors: 4, isOperational: true },
  { id: 12, name: 'EVBox Haarlem Centrum', lat: 52.3810, lng: 4.6360, town: 'Haarlem', operator: 'EVBox', maxKW: 22, connectors: 2, isOperational: true },
  { id: 13, name: 'IONITY A4 Hoofddorp', lat: 52.3060, lng: 4.6870, town: 'Hoofddorp', operator: 'IONITY', maxKW: 350, connectors: 8, isOperational: true },
  { id: 14, name: 'FastNed Zaandam', lat: 52.4418, lng: 4.8263, town: 'Zaandam', operator: 'FastNed', maxKW: 300, connectors: 4, isOperational: true },
  { id: 15, name: 'Allego Diemen', lat: 52.3410, lng: 4.9620, town: 'Diemen', operator: 'Allego', maxKW: 50, connectors: 3, isOperational: true },
  { id: 16, name: 'GreenFlux Amstelveen', lat: 52.3000, lng: 4.8600, town: 'Amstelveen', operator: 'GreenFlux', maxKW: 50, connectors: 2, isOperational: true },
  { id: 17, name: 'Tesla Supercharger Weesp', lat: 52.3080, lng: 5.0420, town: 'Weesp', operator: 'Tesla', maxKW: 250, connectors: 8, isOperational: true },
  { id: 18, name: 'Shell Recharge Muiden', lat: 52.3340, lng: 5.0710, town: 'Muiden', operator: 'Shell Recharge', maxKW: 50, connectors: 2, isOperational: true },
  { id: 19, name: 'FastNed Almere Poort', lat: 52.3510, lng: 5.1280, town: 'Almere', operator: 'FastNed', maxKW: 300, connectors: 4, isOperational: true },
  { id: 20, name: 'EVBox Hilversum', lat: 52.2230, lng: 5.1720, town: 'Hilversum', operator: 'EVBox', maxKW: 22, connectors: 2, isOperational: true },
]

function createStationIcon() {
  return L.divIcon({
    className: '',
    html: `<div style="
      width:30px;height:30px;background:#5aa9e6;border-radius:50%;border:3px solid #fff;
      display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,0.2);
      font-size:13px;color:#fff;
    ">⚡</div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 30],
    popupAnchor: [0, -30],
  })
}

function createActiveIcon() {
  return L.divIcon({
    className: '',
    html: `<div style="
      width:36px;height:36px;background:#3d7bc9;border-radius:50%;border:4px solid #fff;
      display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(90,169,230,0.5);
      font-size:15px;color:#fff;
    ">⚡</div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 36],
    popupAnchor: [0, -36],
  })
}

export default function ChargingStations() {
  const [selected, setSelected] = useState<Station>(STATIONS[0])
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const markersRef = useRef<Map<number, L.Marker>>(new Map())

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return

    const map = L.map(mapContainerRef.current, {
      center: NL_CENTER,
      zoom: 12,
      zoomControl: true,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map)

    // Add markers
    STATIONS.forEach((s) => {
      const marker = L.marker([s.lat, s.lng], { icon: createStationIcon() })
        .addTo(map)
        .bindPopup(`<b>${s.name}</b><br/>${s.maxKW ? s.maxKW + ' kW • ' : ''}${s.operator}`)

      marker.on('click', () => setSelected(s))
      markersRef.current.set(s.id, marker)
    })

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // Update marker styles when selection changes
  useEffect(() => {
    markersRef.current.forEach((marker, id) => {
      marker.setIcon(id === selected.id ? createActiveIcon() : createStationIcon())
    })
    if (mapRef.current) {
      mapRef.current.flyTo([selected.lat, selected.lng], 13, { duration: 0.8 })
    }
  }, [selected])

  const getStatusColor = (s: Station) =>
    s.isOperational ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'

  const getStatusLabel = (s: Station) =>
    s.isOperational ? 'AVAILABLE' : 'OFFLINE'

  return (
    <div className="flex-1 relative flex h-full overflow-hidden">

      {/* ───── Map Area ───── */}
      <div className="flex-1 relative overflow-hidden">
        <div ref={mapContainerRef} className="h-full w-full" />

        {/* Selected Station Card */}
        <div className="absolute bottom-6 left-6 right-6 lg:right-auto lg:w-[420px] z-[1000]" style={{ animation: 'slideUp 0.4s ease-out' }}>
          <div className="glass-ivory rounded-2xl shadow-2xl p-5">
            <div className="flex justify-between items-start mb-4">
              <div>
                <span className="inline-block px-2 py-0.5 bg-brand-primary/10 text-brand-primary text-[10px] font-bold tracking-wider uppercase rounded-md mb-2">
                  {selected.operator}
                </span>
                <h3 className="text-lg font-bold text-surface-900 leading-tight">{selected.name}</h3>
                <p className="text-sm text-surface-800/50 flex items-center gap-1 mt-1">
                  <span className="material-symbols-outlined text-sm">location_on</span>
                  {selected.town}, Netherlands
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-ice/50 p-3 rounded-xl border border-brand-primary/10">
                <p className="text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-1">Max Power</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-surface-900">{selected.maxKW || '—'}</span>
                  <span className="text-sm font-medium text-surface-800/50">kW</span>
                </div>
              </div>
              <div className="bg-ice/50 p-3 rounded-xl border border-brand-primary/10">
                <p className="text-[10px] font-bold text-surface-800/40 uppercase tracking-widest mb-1">Connectors</p>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-surface-900">{selected.connectors}</span>
                  <span className="text-sm font-medium text-surface-800/50">ports</span>
                </div>
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => window.open(`https://www.google.com/maps/dir/?api=1&destination=${selected.lat},${selected.lng}`, '_blank')}
                className="flex-1 bg-brand-primary text-white font-bold py-3 rounded-xl hover:bg-brand-secondary transition-all duration-300 flex items-center justify-center gap-2 shadow-lg shadow-brand-primary/20"
              >
                <span className="material-symbols-outlined">navigation</span>
                Directions
              </button>
              <button className="w-12 bg-surface-100 text-surface-800/60 rounded-xl flex items-center justify-center hover:bg-surface-200 transition-colors">
                <span className="material-symbols-outlined">share</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ───── Station List ───── */}
      <div className="w-[380px] bg-ivory border-l border-surface-200 flex flex-col shadow-xl z-20 shrink-0">
        <div className="p-6 border-b border-surface-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-surface-900">Nearby Chargers</h3>
            <span className="text-xs font-bold text-brand-primary">{STATIONS.length} found</span>
          </div>
          <div className="relative">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-surface-800/30 text-xl">search</span>
            <input
              className="w-full bg-surface-50 border border-surface-200 rounded-xl pl-11 pr-4 py-2.5 text-sm focus:ring-brand-primary focus:border-brand-primary outline-none text-surface-900"
              placeholder="Search station name…"
              type="text"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {STATIONS.map((s) => (
            <div
              key={s.id}
              onClick={() => setSelected(s)}
              className={`p-4 rounded-2xl cursor-pointer transition-all duration-300 ${
                selected.id === s.id
                  ? 'border-2 border-brand-primary bg-brand-primary/5 ring-4 ring-brand-primary/5 shadow-md'
                  : 'border border-surface-200 bg-white hover:border-brand-primary/20 hover:shadow-sm'
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <h4 className="font-bold text-surface-900 leading-tight text-sm">{s.name}</h4>
                <span className={`text-[10px] px-2 py-0.5 rounded-md font-bold shrink-0 ml-2 ${getStatusColor(s)}`}>
                  {getStatusLabel(s)}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-surface-800/50">
                {s.maxKW && (
                  <div className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm">bolt</span>
                    {s.maxKW} kW
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm">location_on</span>
                  {s.town}
                </div>
              </div>
              <p className="text-[10px] text-surface-800/30 mt-2 font-medium">{s.operator}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
