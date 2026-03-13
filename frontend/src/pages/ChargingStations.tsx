import { useState, useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* Amsterdam-area demonstration fallback data */
const DEMO_STATIONS = [
  { ID: 1, AddressInfo: { Title: 'FastNed Amstel', AddressLine1: 'Julianaplein 1', Town: 'Amsterdam', Latitude: 52.3469, Longitude: 4.9179 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 2, AddressInfo: { Title: 'Shell Recharge Centrum', AddressLine1: 'Waterlooplein 2', Town: 'Amsterdam', Latitude: 52.3714, Longitude: 4.8584 }, Connections: [{ PowerKW: 50, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 3, AddressInfo: { Title: 'Allego P+R Sloterdijk', AddressLine1: 'Piarcoplein 1', Town: 'Amsterdam', Latitude: 52.3890, Longitude: 4.8375 }, Connections: [{ PowerKW: 150, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 4, AddressInfo: { Title: 'Tesla Supercharger Schiphol', AddressLine1: 'Schiphol Boulevard 1', Town: 'Schiphol', Latitude: 52.3056, Longitude: 4.7581 }, Connections: [{ PowerKW: 250, LevelID: 3, ConnectionTypeID: 27 }] },
  { ID: 5, AddressInfo: { Title: 'IONITY A4 Hoofddorp', AddressLine1: 'Rijksweg A4', Town: 'Hoofddorp', Latitude: 52.3060, Longitude: 4.6870 }, Connections: [{ PowerKW: 350, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 6, AddressInfo: { Title: 'FastNed A10 West', AddressLine1: 'Rijksweg A10', Town: 'Amsterdam', Latitude: 52.3760, Longitude: 4.8100 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 7, AddressInfo: { Title: 'GreenFlux Zuidas', AddressLine1: 'Gustav Mahlerlaan', Town: 'Amsterdam', Latitude: 52.3364, Longitude: 4.8738 }, Connections: [{ PowerKW: 50, LevelID: 2, ConnectionTypeID: 25 }] },
  { ID: 8, AddressInfo: { Title: 'Allego Centraal Station', AddressLine1: 'Stationsplein 1', Town: 'Amsterdam', Latitude: 52.3791, Longitude: 4.9003 }, Connections: [{ PowerKW: 50, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 9, AddressInfo: { Title: 'FastNed Zaandam', AddressLine1: 'Rijksweg A8', Town: 'Zaandam', Latitude: 52.4418, Longitude: 4.8263 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }] },
  { ID: 10, AddressInfo: { Title: 'Shell Recharge Bijlmer', AddressLine1: 'Arena Boulevard', Town: 'Amsterdam', Latitude: 52.3120, Longitude: 4.9480 }, Connections: [{ PowerKW: 150, LevelID: 3, ConnectionTypeID: 33 }] },
]

export default function ChargingStations() {
  const [stations, setStations] = useState<any[]>(DEMO_STATIONS)
  const [loading, setLoading] = useState(false)
  const [selectedStation, setSelectedStation] = useState<any | null>(null)
  const [usingDemoData, setUsingDemoData] = useState(true)

  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<L.Map | null>(null)
  const markersRef = useRef<{ [id: string]: L.Marker }>({})

  // Initialize Map & Fetch Data
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return

    const map = L.map(mapRef.current, {
      center: [52.3676, 4.9041], // Default Amsterdam center
      zoom: 12,
      zoomControl: false,
    })

    // Blade Runner Dark CARTO tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'bottomright' }).addTo(map)
    leafletMap.current = map

    const fetchStations = async (lat: number, lng: number) => {
      setLoading(true)
      try {
        const apiKey = import.meta.env.VITE_OPENCHARGE_API_KEY;
        const response = await fetch(`/ocm/poi/?output=json&maxresults=50&compact=true&verbose=false&latitude=${lat}&longitude=${lng}&distance=30`, {
          headers: {
            'X-API-Key': apiKey || ''
          }
        })
        if (!response.ok) {
          if (response.status === 403) throw new Error('API Key Required')
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        const data = await response.json()
        if (data && data.length > 0) {
          setStations(data)
          setUsingDemoData(false)
        }
      } catch (err: any) {
        if (err.message.includes('API Key Required')) {
          setUsingDemoData(true)
          setStations(DEMO_STATIONS)
        } else {
          console.error(err.message)
        }
      } finally {
        setLoading(false)
      }
    }

    // Initial fetch
    fetchStations(52.3676, 4.9041)

    // Listen for map panning to refetch
    let timeout: ReturnType<typeof setTimeout>
    map.on('moveend', () => {
      clearTimeout(timeout)
      // Debounce fetch to stop API spam while user scrolls quickly
      timeout = setTimeout(() => {
        const center = map.getCenter()
        fetchStations(center.lat, center.lng)
      }, 750)
    })

    return () => {
      map.remove()
      leafletMap.current = null
      markersRef.current = {}
      clearTimeout(timeout)
    }
  }, [])

  // Update Markers
  useEffect(() => {
    const map = leafletMap.current
    if (!map) return

    // Clear old markers
    Object.values(markersRef.current).forEach((marker) => marker.remove())
    markersRef.current = {}

    stations.forEach((station) => {
      const lat = station.AddressInfo.Latitude
      const lng = station.AddressInfo.Longitude
      if (!lat || !lng) return

      const maxPower = Math.max(...(station.Connections?.map((c: any) => c.PowerKW || 0) || [0]))
      const isFast = maxPower >= 50

      const pinColor = isFast ? '#00E5CC' : '#00E676'
      const customIcon = L.divIcon({
        className: '',
        html: `<div style="width:28px;height:28px;background:rgba(10,14,23,0.8);backdrop-filter:blur(4px);border-radius:50%;border:2px solid ${pinColor};display:flex;align-items:center;justify-content:center;box-shadow:0 0 15px ${pinColor}80, inset 0 0 10px ${pinColor}40;font-size:14px;color:${pinColor}; transition: transform 0.3s; cursor: pointer;" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">⚡</div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 28],
      })

      const marker = L.marker([lat, lng], { icon: customIcon })
        .addTo(map)
        .on('click', () => {
          setSelectedStation(station)
          map.flyTo([lat, lng], 14, { animate: true, duration: 1 })
        })

      markersRef.current[station.ID] = marker
    })
  }, [stations])

  return (
    <div className="flex-1 flex h-full overflow-hidden p-6 gap-6 relative">

      {/* ═══ Main Map Container ═══ */}
      <div className="flex-1 glass-dark  overflow-hidden relative shadow-[0_0_30px_rgba(0,180,216,0.1)] border border-neon-blue/20">
        <div ref={mapRef} className="h-full w-full z-0 opacity-90" />
        <div className="absolute inset-0 z-10 pointer-events-none shadow-[inset_0_0_100px_rgba(10,14,23,0.8)]" />
        
        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-brand-bg/60 backdrop-blur-sm flex items-center justify-center z-[1000]">
            <div className="w-12 h-12 rounded-full border-2 border-neon-blue/30 border-t-neon-blue animate-spin shadow-[0_0_15px_#00b4d8]" />
          </div>
        )}
      </div>

      {/* ═══ Left Panel (Floating) ═══ */}
      <div className="w-full md:w-[380px] flex flex-col gap-6 shrink-0 relative z-[1000] pointer-events-none h-full" style={{ animation: 'slideRight 0.4s ease-out' }}>
        
        {/* Header Search & API Status */}
        <div className="glass-dark  p-5 md:p-6 pointer-events-auto border border-neon-blue/20 shadow-[0_4px_30px_rgba(0,180,216,0.15)] shrink-0">
          <h2 className="text-2xl font-headline font-bold text-surface-900 tracking-tight glow-neon mb-1">Charging Network</h2>
          <p className="text-[10px] font-mono text-neon-blue/60 uppercase tracking-widest mb-5">Global Grid Access</p>

          <div className="relative mb-5">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-neon-blue text-[20px]">search</span>
            <input
              type="text"
              placeholder="Search area (e.g. Amsterdam)..."
              className="w-full bg-surface-100/50 border border-white/10 text-surface-900 text-sm pl-12 pr-4 py-3.5 outline-none focus:border-neon-blue/50 focus:bg-surface-200/50 transition-all shadow-inner"
            />
          </div>

          <div className={`p-4 border flex gap-3.5 items-start ${
            usingDemoData 
              ? 'bg-accent-warning/5 border-accent-warning/20' 
              : 'bg-accent-success/5 border-accent-success/20'
          }`}>
            <span className={`material-symbols-outlined text-[22px] shrink-0 ${usingDemoData ? 'text-accent-warning' : 'text-accent-success'}`}>
              {usingDemoData ? 'api' : 'verified_user'}
            </span>
            <div className="flex-1 min-w-0">
              <p className={`text-[10px] font-mono font-bold uppercase tracking-widest mb-1 ${
                usingDemoData ? 'text-accent-warning' : 'text-accent-success'
              }`}>{usingDemoData ? 'Demo Mode Active' : 'Live Data Active'}</p>
              <p className="text-xs text-surface-800/60 leading-relaxed">
                {usingDemoData 
                  ? 'Showing fallback data. To use live data, set your OpenChargeMap API key in the backend configuration.'
                  : 'Successfully connected to OpenChargeMap live API.'}
              </p>
            </div>
          </div>
        </div>

        {/* Selected Station Details Panel */}
        {selectedStation ? (
          <div className="glass-dark  p-5 md:p-6 border border-neon-blue/20 shadow-[0_4px_30px_rgba(0,180,216,0.15)] flex-1 overflow-y-auto custom-scrollbar pointer-events-auto relative min-h-0">
            <button 
              onClick={() => setSelectedStation(null)}
              className="absolute top-6 right-6 w-8 h-8 rounded-full bg-surface-100 border border-white/5 flex items-center justify-center text-surface-800/50 hover:text-neon-red hover:border-neon-red/30 transition-all"
            >
              <span className="material-symbols-outlined text-[18px]">close</span>
            </button>

            <div className="mb-6">
              <span className="inline-block px-2 py-0.5 border border-neon-green/30 text-[9px] font-mono font-bold text-neon-green  shadow-[0_0_10px_rgba(0,245,160,0.2)] mb-3 uppercase tracking-widest">
                Station Active
              </span>
              <h3 className="text-xl font-headline font-bold text-surface-900 leading-tight mb-2 pr-10">
                {selectedStation.AddressInfo.Title}
              </h3>
              <p className="text-xs text-surface-800/50 font-mono">
                {selectedStation.AddressInfo.AddressLine1}, {selectedStation.AddressInfo.Town}
              </p>
            </div>

            <h4 className="text-[10px] font-mono text-neon-blue/60 uppercase tracking-widest mb-3 border-b border-neon-blue/10 pb-2">Available Connectors</h4>
            <div className="space-y-3 mb-6">
              {selectedStation.Connections?.map((conn: any, i: number) => (
                <div key={i} className="flex items-center justify-between p-3.5 bg-surface-100/50 border border-white/5  group hover:border-neon-blue/20 transition-all">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10  bg-surface-200/50 border border-white/5 flex items-center justify-center text-neon-blue shrink-0 shadow-[inset_0_0_10px_rgba(0,180,216,0.1)]">
                      <span className="material-symbols-outlined">electrical_services</span>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-surface-900 group-hover:text-neon-blue transition-colors">Type {conn.ConnectionTypeID || 'Unknown'}</p>
                      <p className="text-[10px] text-surface-800/40 font-mono mt-0.5">DC Fast Charging</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-headline font-bold text-neon-green glow-neon">{conn.PowerKW || '?'} <span className="text-[10px] font-sans">kW</span></p>
                  </div>
                </div>
              )) || <p className="text-xs text-surface-800/40 italic">No connection details available.</p>}
            </div>

            <div className="flex gap-3">
              <button className="flex-1 bg-gradient-to-r from-neon-blue to-neon-green text-brand-bg font-extrabold text-[11px] font-mono uppercase tracking-widest py-3.5  transition-all shadow-[0_4px_20px_rgba(0,180,216,0.25)] hover:scale-[1.02] flex items-center justify-center gap-2">
                <span className="material-symbols-outlined text-lg">navigation</span>
                Navigate
              </button>
              <button className="w-12 h-12  bg-surface-100 border border-white/5 flex items-center justify-center text-surface-800/50 hover:text-neon-blue hover:border-neon-blue/30 transition-all">
                <span className="material-symbols-outlined">bookmark</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="glass-dark  border border-neon-blue/20 flex-1 overflow-hidden flex flex-col pointer-events-auto min-h-0">
            <div className="p-4 border-b border-white/5 bg-surface-50/50 backdrop-blur shrink-0">
              <p className="text-[10px] font-mono font-bold text-surface-800/50 uppercase tracking-widest text-center mt-1">Nearby Grid Nodes ({stations.length})</p>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
              {stations.map((s) => {
                const maxPower = Math.max(...(s.Connections?.map((c: any) => c.PowerKW || 0) || [0]))
                const isFast = maxPower >= 50
                return (
                  <div
                    key={s.ID}
                    onClick={() => {
                      setSelectedStation(s)
                      const map = leafletMap.current
                      if (map) map.flyTo([s.AddressInfo.Latitude, s.AddressInfo.Longitude], 14, { animate: true, duration: 1 })
                    }}
                    className="p-4 border-b border-white/5 last:border-0 hover:bg-surface-100/30 cursor-pointer group transition-colors flex items-center gap-4"
                  >
                    <div className={`w-10 h-10 rounded-full border border-white/10 flex items-center justify-center shrink-0 shadow-[inset_0_0_10px_rgba(255,255,255,0.05)] transition-colors ${
                      isFast ? 'bg-neon-blue/10 text-neon-blue group-hover:border-neon-blue/50' : 'bg-neon-green/10 text-neon-green group-hover:border-neon-green/50'
                    }`}>
                      <span className="material-symbols-outlined text-sm">ev_station</span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-surface-900 group-hover:text-neon-blue transition-colors truncate">{s.AddressInfo.Title}</p>
                      <p className="text-[10px] font-mono text-surface-800/40 truncate">{s.AddressInfo.AddressLine1}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className={`text-sm font-bold font-headline ${isFast ? 'text-neon-blue' : 'text-neon-green'}`}>{maxPower > 0 ? `${maxPower} kW` : '--'}</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}

