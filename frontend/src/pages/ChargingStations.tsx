import { useState, useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import useWindowSize from '../hooks/useWindowSize'
import GeoSearch, { GeoSearchResult } from '../components/GeoSearch'
import { useVehicle } from '../contexts/VehicleContext'

function useIsLightTheme() {
  const [isLight, setIsLight] = useState(() => document.documentElement.classList.contains('light'))
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsLight(document.documentElement.classList.contains('light'))
    })
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])
  return isLight
}

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
  if (statusId === 0 || statusId === 100) return { label: 'OFFLINE', color: 'text-error', bg: 'bg-error-container/20 border-error/30' }
  if (statusId === 5 || statusId === 75) return { label: 'BUSY', color: 'text-tertiary-container', bg: 'bg-tertiary-container/10 border-tertiary-container/30' }
  return { label: 'AVAILABLE', color: 'text-secondary-container', bg: 'bg-secondary-container/10 border-secondary-container/30' }
}

export default function ChargingStations() {
  const { vehicle } = useVehicle()
  const isLight = useIsLightTheme()
  const [stations, setStations] = useState<any[]>(DEMO_STATIONS)
  const [loading, setLoading] = useState(false)
  const [selectedStation, setSelectedStation] = useState<any | null>(null)
  const [usingDemoData, setUsingDemoData] = useState(true)
  const [activeFilter, setActiveFilter] = useState<'all' | 'fast' | 'available'>('all')

  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<L.Map | null>(null)
  const layerControlRef = useRef<L.TileLayer | null>(null)
  const markersRef = useRef<{ [id: string]: L.Marker }>({})
  const { isMobile } = useWindowSize()
  const [showList, setShowList] = useState(!isMobile)

  // Initialize Map & Fetch Data
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return

    const map = L.map(mapRef.current, {
      center: [52.3676, 4.9041],
      zoom: 12,
      zoomControl: false,
    })
    
    layerControlRef.current = L.tileLayer(isLight ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png' : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'bottomright' }).addTo(map)
    leafletMap.current = map

    const fetchStations = async (lat: number, lng: number) => {
      setLoading(true)
      try {
        const baseUrl = import.meta.env.VITE_API_URL || '';
        const response = await fetch(`${baseUrl}/api/external/ocm/poi?output=json&maxresults=50&compact=true&verbose=false&latitude=${lat}&longitude=${lng}&distance=30`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
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

    const timeout = setTimeout(() => {
      map.invalidateSize()
    }, 100)

    map.on('moveend', () => {
      const center = map.getCenter()
      setTimeout(() => {
        fetchStations(center.lat, center.lng)
      }, 750)
    })

    return () => {
      map.remove()
      leafletMap.current = null
      layerControlRef.current = null
      markersRef.current = {}
      clearTimeout(timeout)
    }
  }, []) // Empty dependency array as this should only run once on mount. isLight will be handled by the other effect.

  // ── Update Map Tiles for Theme
  useEffect(() => {
    if (!leafletMap.current) return
    if (layerControlRef.current) {
      layerControlRef.current.remove()
    }
    const tileUrl = isLight
      ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'

    layerControlRef.current = L.tileLayer(tileUrl, { maxZoom: 19 }).addTo(leafletMap.current)
  }, [isLight])

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
    return true
  })

  return (
    <div className={`flex-1 flex h-full overflow-hidden relative ${isMobile ? 'flex-col' : ''}`}>

      {/* ═══ Map (Left — takes most width) ═══ */}
      <div className="flex-1 relative bg-background">
        <div ref={mapRef} className="h-full w-full z-0" />
        <div className="absolute inset-0 z-[5] pointer-events-none shadow-[inset_0_0_80px_rgba(10,14,23,0.7)]" />

        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-surface-container-lowest/50 backdrop-blur-sm flex items-center justify-center z-[1000]">
            <div className="w-10 h-10 rounded-full border-2 border-primary/30 border-t-primary animate-spin shadow-[0_0_15px_rgba(175,236,255,0.4)]" />
          </div>
        )}

        {/* Selected Station Popup (Bottom Left) */}
        {selectedStation && (
          <div className={`absolute z-[1000] ${isMobile ? 'bottom-2 left-2 right-2 w-auto' : 'bottom-5 left-5 w-[360px]'}`} style={{ animation: 'slideUp 0.3s ease-out' }}>
            <div className="bg-surface-container-high p-7 rounded-2xl border border-outline-variant/20 relative overflow-hidden shadow-2xl">
              <button
                onClick={() => setSelectedStation(null)}
                className="absolute top-5 right-5 w-7 h-7 rounded-full bg-surface-container-highest border border-white/10 flex items-center justify-center text-on-surface-variant hover:text-error hover:border-error/30 transition-all z-10"
                aria-label="Close station details"
              >
                <span className="material-symbols-outlined text-[16px]">close</span>
              </button>

              <div className="flex items-start gap-3 mb-4 relative z-10">
                <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center text-primary shrink-0">
                  <span className="material-symbols-outlined">ev_station</span>
                </div>
                <div className="pr-8">
                  <h3 className="text-base font-bold text-on-surface leading-tight">{selectedStation.AddressInfo.Title}</h3>
                  <p className="text-[10px] text-on-surface-variant font-mono mt-0.5">{selectedStation.AddressInfo.AddressLine1}, {selectedStation.AddressInfo.Town}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4 relative z-10">
                {selectedStation.Connections?.slice(0, 2).map((conn: any, i: number) => (
                  <div key={i} className="bg-surface-container-highest rounded-xl p-3 flex flex-col justify-between">
                    <div>
                      <p className="text-[9px] font-mono text-on-surface-variant uppercase tracking-widest mb-1">Connector</p>
                      <p className="text-lg font-bold text-secondary-container">{conn.PowerKW || '?'} <span className="text-[10px] font-sans text-on-surface-variant">kW</span></p>
                    </div>
                    {conn.PowerKW && (
                      <p className="text-[9px] font-mono text-on-surface-variant uppercase tracking-wide mt-1">
                        ~{Math.round(((vehicle.battery.capacity_kwh * 0.8) / conn.PowerKW) * 60)}m to 80% (for {vehicle.name.split(' - ')[0]})
                      </p>
                    )}
                  </div>
                ))}
              </div>

              <div className="flex gap-2 relative z-10">
                <button className="flex-1 bg-primary-fixed-dim text-on-primary font-extrabold text-[10px] font-mono uppercase tracking-widest py-3 rounded-xl transition-all hover:brightness-110 hover:scale-[1.02] flex items-center justify-center gap-2 shadow-[0_4px_20px_rgba(0,217,255,0.25)]" aria-label="Navigate to station">
                  <span className="material-symbols-outlined text-base" aria-hidden="true">navigation</span>
                  Navigate
                </button>
                <button className="w-11 h-11 bg-surface-container-highest rounded-xl border border-white/5 flex items-center justify-center text-on-surface-variant hover:text-primary hover:border-primary/30 transition-all" aria-label="Bookmark station">
                  <span className="material-symbols-outlined text-lg" aria-hidden="true">bookmark</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ═══ Right Panel ═══ */}
      {/* Mobile toggle button */}
      {isMobile && (
        <button
          onClick={() => setShowList(!showList)}
          className="absolute top-3 right-3 z-[1100] w-11 h-11 rounded-full bg-surface-container-high border border-primary/30 flex items-center justify-center text-primary shadow-[0_0_15px_rgba(175,236,255,0.2)] active:scale-95 transition-transform"
          aria-label="Toggle station list"
        >
          <span className="material-symbols-outlined text-xl">{showList ? 'close' : 'list'}</span>
        </button>
      )}
      <div className={`flex flex-col shrink-0 bg-surface-container-low backdrop-blur-xl border-l border-outline-variant/20 overflow-hidden transition-all duration-300 ${
        isMobile
          ? `absolute bottom-0 left-0 right-0 z-[1000] border-l-0 border-t border-outline-variant/20 ${showList ? 'h-[50vh]' : 'h-0'}`
          : 'w-[340px] xl:w-[380px] h-full relative'
      }`}>

        {/* Header + Search */}
        <div className="p-5 pb-4 shrink-0 border-b border-white/5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-on-surface tracking-tight">Nearby Chargers</h2>
            <span className="text-[10px] font-mono font-bold text-primary">{filteredStations.length} found</span>
          </div>

          <div className="relative mb-3">
            <GeoSearch
              variant="solid"
              color="blue"
              placeholder="Search globally for any location..."
              onSelect={(result: GeoSearchResult) => {
                const map = leafletMap.current
                if (map) {
                  map.flyTo([parseFloat(result.lat), parseFloat(result.lon)], 14, { animate: true, duration: 1 })
                }
              }}
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
                className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono font-bold uppercase tracking-widest transition-all border rounded-lg ${
                  activeFilter === f.key
                    ? 'bg-primary/15 text-primary border-primary/30'
                    : 'text-on-surface-variant border-outline-variant/20 hover:text-on-surface hover:border-outline-variant/40'
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
        <div className={`mx-5 mt-3 mb-2 px-3 py-2 rounded-lg border flex gap-2 items-center shrink-0 ${
          usingDemoData
            ? 'bg-tertiary-container/10 border-tertiary-container/20'
            : 'bg-secondary-container/10 border-secondary-container/20'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 animate-pulse ${usingDemoData ? 'bg-tertiary-container' : 'bg-secondary-container'}`} />
          <span className={`text-[9px] font-mono font-bold uppercase tracking-widest ${
            usingDemoData ? 'text-tertiary-container' : 'text-secondary-container'
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
                className={`p-4 mb-2 rounded-xl border cursor-pointer group transition-all ${
                  isSelected
                    ? 'border-primary/40 bg-primary/5'
                    : 'border-outline-variant/10 hover:border-outline-variant/30 hover:bg-surface-container'
                }`}
              >
                {/* Header: Name + Distance */}
                <div className="flex items-start justify-between mb-3 px-1">
                  <h4 className={`text-sm font-semibold leading-tight pr-2 transition-colors ${isSelected ? 'text-primary' : 'text-on-surface group-hover:text-primary'}`}>
                    {s.AddressInfo.Title}
                  </h4>
                  <span className="text-[10px] font-mono font-bold text-on-surface-variant shrink-0">
                    {(Math.random() * 4 + 0.3).toFixed(1)} MI
                  </span>
                </div>

                {/* Availability bar */}
                <div className="mb-2">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-widest font-bold">Availability</span>
                    <span className={`text-[10px] font-mono font-bold ${
                      status.label === 'AVAILABLE' ? 'text-secondary-container' 
                      : status.label === 'BUSY' ? 'text-tertiary-container' 
                      : 'text-error'
                    }`}>
                      {freePorts}/{totalPorts} FREE
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        status.label === 'AVAILABLE' ? 'bg-secondary-container'
                        : status.label === 'BUSY' ? 'bg-tertiary-container'
                        : 'bg-error'
                      }`}
                      style={{ width: `${(freePorts / totalPorts) * 100}%` }}
                    />
                  </div>
                </div>

                {/* Tags: Power + Connector */}
                <div className="flex gap-1.5 mt-2">
                  <span className="text-[9px] font-mono font-bold text-on-surface-variant bg-surface-container-highest px-2 py-1 rounded">
                    {maxPower > 0 ? `${maxPower}kW` : '--'}
                  </span>
                  <span className="text-[9px] font-mono font-bold text-on-surface-variant bg-surface-container-highest px-2 py-1 rounded">
                    {maxPower >= 150 ? 'CCS2' : maxPower >= 50 ? 'Type 2' : 'AC'}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
