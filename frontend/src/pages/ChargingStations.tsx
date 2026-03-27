import { useState, useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* Amsterdam-area demonstration fallback data */
const DEMO_STATIONS = [
  { ID: 1, AddressInfo: { Title: 'FastNed Amstel', AddressLine1: 'Julianaplein 1', Town: 'Amsterdam', Latitude: 52.3469, Longitude: 4.9179 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 4 },
  { ID: 2, AddressInfo: { Title: 'Shell Recharge Centrum', AddressLine1: 'Waterlooplein 2', Town: 'Amsterdam', Latitude: 52.3714, Longitude: 4.8584 }, Connections: [{ PowerKW: 50, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 6 },
  { ID: 3, AddressInfo: { Title: 'Allego P+R Sloterdijk', AddressLine1: 'Piarcoplein 1', Town: 'Amsterdam', Latitude: 52.3890, Longitude: 4.8375 }, Connections: [{ PowerKW: 150, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 8 },
  { ID: 4, AddressInfo: { Title: 'Tesla Supercharger Schiphol', AddressLine1: 'Schiphol Boulevard 1', Town: 'Schiphol', Latitude: 52.3056, Longitude: 4.7581 }, Connections: [{ PowerKW: 250, LevelID: 3, ConnectionTypeID: 27 }], StatusTypeID: 5, NumberOfPoints: 12 },
  { ID: 5, AddressInfo: { Title: 'IONITY A4 Hoofddorp', AddressLine1: 'Rijksweg A4', Town: 'Hoofddorp', Latitude: 52.3060, Longitude: 4.6870 }, Connections: [{ PowerKW: 350, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 6 },
  { ID: 6, AddressInfo: { Title: 'FastNed A10 West', AddressLine1: 'Rijksweg A10', Town: 'Amsterdam', Latitude: 52.3760, Longitude: 4.8100 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 4 },
  { ID: 7, AddressInfo: { Title: 'GreenFlux Zuidas', AddressLine1: 'Gustav Mahlerlaan', Town: 'Amsterdam', Latitude: 52.3364, Longitude: 4.8738 }, Connections: [{ PowerKW: 50, LevelID: 2, ConnectionTypeID: 25 }], StatusTypeID: 0, NumberOfPoints: 2 },
  { ID: 8, AddressInfo: { Title: 'Allego Centraal Station', AddressLine1: 'Stationsplein 1', Town: 'Amsterdam', Latitude: 52.3791, Longitude: 4.9003 }, Connections: [{ PowerKW: 50, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 4 },
  { ID: 9, AddressInfo: { Title: 'FastNed Zaandam', AddressLine1: 'Rijksweg A8', Town: 'Zaandam', Latitude: 52.4418, Longitude: 4.8263 }, Connections: [{ PowerKW: 300, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 6 },
  { ID: 10, AddressInfo: { Title: 'Shell Recharge Bijlmer', AddressLine1: 'Arena Boulevard', Town: 'Amsterdam', Latitude: 52.3120, Longitude: 4.9480 }, Connections: [{ PowerKW: 150, LevelID: 3, ConnectionTypeID: 33 }], StatusTypeID: 2, NumberOfPoints: 8 },
]

function getStationStatus(station: any) {
  const statusId = station.StatusTypeID ?? station.StatusType?.ID ?? 2
  if (statusId === 0 || statusId === 100) return { label: 'OFFLINE', color: 'text-neon-red', bg: 'bg-neon-red/10 border-neon-red/30' }
  if (statusId === 5 || statusId === 75) return { label: 'BUSY', color: 'text-accent-warning', bg: 'bg-accent-warning/10 border-accent-warning/30' }
  return { label: 'AVAILABLE', color: 'text-neon-green', bg: 'bg-neon-green/10 border-neon-green/30' }
}

export default function ChargingStations() {
  const [stations, setStations] = useState<any[]>(DEMO_STATIONS)
  const [loading, setLoading] = useState(false)
  const [selectedStation, setSelectedStation] = useState<any | null>(null)
  const [usingDemoData, setUsingDemoData] = useState(true)
  const [activeFilter, setActiveFilter] = useState<'all' | 'fast' | 'available'>('all')
  const [searchQuery, setSearchQuery] = useState('')

  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<L.Map | null>(null)
  const markersRef = useRef<{ [id: string]: L.Marker }>({})

  // Initialize Map & Fetch Data
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return

    const map = L.map(mapRef.current, {
      center: [52.3676, 4.9041],
      zoom: 12,
      zoomControl: false,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'bottomright' }).addTo(map)
    leafletMap.current = map

    const fetchStations = async (lat: number, lng: number) => {
      setLoading(true)
      try {
        const apiKey = import.meta.env.VITE_OPENCHARGE_API_KEY;
        if (!apiKey) throw new Error('API Key Required')
        const response = await fetch(`/ocm/poi/?output=json&maxresults=50&compact=true&verbose=false&latitude=${lat}&longitude=${lng}&distance=30`, {
          headers: { 'X-API-Key': apiKey }
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
        console.warn('[OCM]', err.message)
        setUsingDemoData(true)
        setStations(DEMO_STATIONS)
      } finally {
        setLoading(false)
      }
    }

    fetchStations(52.3676, 4.9041)

    let timeout: ReturnType<typeof setTimeout>
    map.on('moveend', () => {
      clearTimeout(timeout)
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

    Object.values(markersRef.current).forEach((marker) => marker.remove())
    markersRef.current = {}

    stations.forEach((station) => {
      const lat = station.AddressInfo?.Latitude
      const lng = station.AddressInfo?.Longitude
      if (!lat || !lng) return

      const _maxPower = Math.max(...(station.Connections?.map((c: any) => c.PowerKW || 0) || [0])); void _maxPower
      const status = getStationStatus(station)
      const pinColor = status.label === 'AVAILABLE' ? '#00E5CC' : status.label === 'BUSY' ? '#FFB300' : '#FF3D00'

      const customIcon = L.divIcon({
        className: '',
        html: `<div style="width:28px;height:28px;background:rgba(10,14,23,0.85);backdrop-filter:blur(4px);border-radius:50%;border:2px solid ${pinColor};display:flex;align-items:center;justify-content:center;box-shadow:0 0 15px ${pinColor}80, inset 0 0 10px ${pinColor}40;font-size:14px;color:${pinColor};transition:transform 0.3s;cursor:pointer;" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">⚡</div>`,
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

  // Filtered stations
  const filteredStations = stations.filter((s) => {
    const maxPower = Math.max(...(s.Connections?.map((c: any) => c.PowerKW || 0) || [0]))
    const status = getStationStatus(s)
    if (activeFilter === 'fast' && maxPower < 50) return false
    if (activeFilter === 'available' && status.label !== 'AVAILABLE') return false
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      return s.AddressInfo?.Title?.toLowerCase().includes(q) || s.AddressInfo?.Town?.toLowerCase().includes(q)
    }
    return true
  })

  return (
    <div className="flex-1 flex h-full overflow-hidden relative">

      {/* ═══ Map (Left — takes most width) ═══ */}
      <div className="flex-1 relative bg-brand-bg">
        <div ref={mapRef} className="h-full w-full z-0" />
        <div className="absolute inset-0 z-[5] pointer-events-none shadow-[inset_0_0_80px_rgba(10,14,23,0.7)]" />

        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-brand-bg/50 backdrop-blur-sm flex items-center justify-center z-[1000]">
            <div className="w-10 h-10 rounded-full border-2 border-neon-blue/30 border-t-neon-blue animate-spin shadow-[0_0_15px_#00b4d8]" />
          </div>
        )}

        {/* Selected Station Popup (Bottom Left) */}
        {selectedStation && (
          <div className="absolute bottom-5 left-5 w-[360px] z-[1000]" style={{ animation: 'slideUp 0.3s ease-out' }}>
            <div className="glass-dark p-5 border border-neon-blue/20 relative overflow-hidden">
              <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-blue/10 rounded-full blur-[30px]" />
              <button 
                onClick={() => setSelectedStation(null)}
                className="absolute top-4 right-4 w-7 h-7 rounded-full bg-surface-100 border border-white/10 flex items-center justify-center text-surface-800/50 hover:text-neon-red hover:border-neon-red/30 transition-all z-10"
                aria-label="Close station details"
              >
                <span className="material-symbols-outlined text-[16px]">close</span>
              </button>

              <div className="flex items-start gap-3 mb-4 relative z-10">
                <div className="w-10 h-10 rounded-full bg-neon-blue/10 border border-neon-blue/30 flex items-center justify-center text-neon-blue shrink-0">
                  <span className="material-symbols-outlined">ev_station</span>
                </div>
                <div className="pr-8">
                  <h3 className="text-base font-headline font-bold text-surface-900 leading-tight">{selectedStation.AddressInfo.Title}</h3>
                  <p className="text-[10px] text-surface-800/50 font-mono mt-0.5">{selectedStation.AddressInfo.AddressLine1}, {selectedStation.AddressInfo.Town}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4 relative z-10">
                {selectedStation.Connections?.slice(0, 2).map((conn: any, i: number) => (
                  <div key={i} className="bg-surface-100/50 border border-white/5 p-3">
                    <p className="text-[9px] font-mono text-surface-800/40 uppercase tracking-widest mb-1">Connector</p>
                    <p className="text-lg font-headline font-bold text-neon-green">{conn.PowerKW || '?'} <span className="text-[10px] font-sans text-surface-800/40">kW</span></p>
                  </div>
                ))}
              </div>

              <div className="flex gap-2 relative z-10">
                <button className="flex-1 bg-gradient-to-r from-neon-blue to-neon-green text-brand-bg font-extrabold text-[10px] font-mono uppercase tracking-widest py-3 transition-all shadow-[0_4px_20px_rgba(0,180,216,0.25)] hover:scale-[1.02] flex items-center justify-center gap-2" aria-label="Navigate to station">
                  <span className="material-symbols-outlined text-base" aria-hidden="true">navigation</span>
                  Navigate
                </button>
                <button className="w-11 h-11 bg-surface-100 border border-white/5 flex items-center justify-center text-surface-800/50 hover:text-neon-blue hover:border-neon-blue/30 transition-all" aria-label="Bookmark station">
                  <span className="material-symbols-outlined text-lg" aria-hidden="true">bookmark</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ═══ Right Panel ═══ */}
      <div className="w-[340px] xl:w-[380px] flex flex-col shrink-0 bg-surface-100/50 backdrop-blur-xl border-l border-neon-blue/10 h-full overflow-hidden">

        {/* Header + Search */}
        <div className="p-5 pb-4 shrink-0 border-b border-white/5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-headline font-bold text-surface-900 tracking-tight">Nearby Chargers</h2>
            <span className="text-[10px] font-mono font-bold text-neon-blue">{filteredStations.length} found</span>
          </div>

          <div className="relative mb-3">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-surface-800/40 text-[18px]" aria-hidden="true">search</span>
            <input
              type="text"
              placeholder="Search address or station"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-surface-200/50 border border-white/10 text-surface-900 text-sm pl-10 pr-4 py-2.5 outline-none focus:border-neon-blue/50 transition-all"
              aria-label="Search charging stations"
            />
          </div>

          {/* Filter pills */}
          <div className="flex gap-2">
            {[
              { key: 'all', icon: 'bolt', label: 'All' },
              { key: 'fast', icon: 'speed', label: 'Fast' },
              { key: 'available', icon: 'check_circle', label: 'Available' },
            ].map((f) => (
              <button
                key={f.key}
                onClick={() => setActiveFilter(f.key as any)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono font-bold uppercase tracking-widest transition-all border ${
                  activeFilter === f.key
                    ? 'bg-neon-blue/15 text-neon-blue border-neon-blue/30 shadow-[0_0_10px_rgba(0,180,216,0.2)]'
                    : 'text-surface-800/40 border-white/5 hover:text-surface-900 hover:border-white/10'
                }`}
                aria-pressed={activeFilter === f.key}
              >
                <span className="material-symbols-outlined text-[14px]" aria-hidden="true">{f.icon}</span>
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* API Status Badge */}
        <div className={`mx-5 mt-3 mb-2 px-3 py-2 border flex gap-2 items-center shrink-0 ${
          usingDemoData
            ? 'bg-accent-warning/5 border-accent-warning/20'
            : 'bg-accent-success/5 border-accent-success/20'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${usingDemoData ? 'bg-accent-warning animate-pulse' : 'bg-accent-success animate-pulse'}`} />
          <span className={`text-[9px] font-mono font-bold uppercase tracking-widest ${
            usingDemoData ? 'text-accent-warning' : 'text-accent-success'
          }`}>{usingDemoData ? 'Demo Mode' : 'Live Data'}</span>
        </div>

        {/* Station List */}
        <div className="flex-1 overflow-y-auto custom-scrollbar px-3 pb-3">
          {filteredStations.map((s) => {
            const maxPower = Math.max(...(s.Connections?.map((c: any) => c.PowerKW || 0) || [0]))
            const status = getStationStatus(s)
            const totalPorts = s.NumberOfPoints || s.Connections?.length || 4
            const freePorts = status.label === 'AVAILABLE' ? totalPorts : status.label === 'BUSY' ? Math.max(1, Math.floor(totalPorts * 0.3)) : 0
            const isSelected = selectedStation?.ID === s.ID

            return (
              <div
                key={s.ID}
                onClick={() => {
                  setSelectedStation(s)
                  const map = leafletMap.current
                  if (map) map.flyTo([s.AddressInfo.Latitude, s.AddressInfo.Longitude], 14, { animate: true, duration: 1 })
                }}
                className={`p-4 mb-2 border cursor-pointer group transition-all ${
                  isSelected
                    ? 'border-neon-blue/40 bg-neon-blue/5 shadow-[0_0_15px_rgba(0,180,216,0.15)]'
                    : 'border-white/5 hover:border-white/10 hover:bg-surface-200/30'
                }`}
              >
                {/* Header: Name + Status */}
                <div className="flex items-start justify-between mb-2">
                  <h4 className={`text-sm font-semibold leading-tight pr-2 transition-colors ${isSelected ? 'text-neon-blue' : 'text-surface-900 group-hover:text-neon-blue'}`}>
                    {s.AddressInfo.Title}
                  </h4>
                  <span className={`text-[8px] font-mono font-bold uppercase tracking-widest px-2 py-0.5 border shrink-0 ${status.bg} ${status.color}`}>
                    {status.label}
                  </span>
                </div>

                {/* Meta: Power + Distance */}
                <div className="flex items-center gap-3 text-[10px] font-mono text-surface-800/50 mb-3">
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[12px] text-neon-blue" aria-hidden="true">bolt</span>
                    {maxPower > 0 ? `${maxPower} kW` : '--'}
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[12px] text-neon-blue" aria-hidden="true">location_on</span>
                    {s.AddressInfo.Town || 'Unknown'}
                  </span>
                </div>

                {/* Port availability bar */}
                <div className="flex gap-1 mb-1.5">
                  {Array.from({ length: Math.min(totalPorts, 8) }, (_, i) => (
                    <div
                      key={i}
                      className={`flex-1 h-1.5 rounded-full transition-all ${
                        i < freePorts
                          ? 'bg-neon-green shadow-[0_0_5px_rgba(0,245,160,0.4)]'
                          : 'bg-surface-200/60'
                      }`}
                    />
                  ))}
                </div>
                <p className="text-[9px] font-mono text-surface-800/30 tracking-wider">
                  {freePorts}/{totalPorts} ports free
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
