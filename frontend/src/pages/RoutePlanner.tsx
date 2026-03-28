import { useState, useEffect, useRef, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import useWindowSize from '../hooks/useWindowSize'
import GeoSearch, { GeoSearchResult } from '../components/GeoSearch'
import {
  orchestrateRoute,
  type LatLng,
  type RouteContext,
  type PipelineStage,
} from '../services/routeService'
import {
  runHybridPrediction,
  selectBestRoute,
  DEFAULT_VEHICLE,
  type PredictionResult,
} from '../services/hybridPredictionEngine'

// ─── Marker Icons ────────────────────────────────────────────────────────────

function createStationIcon(status: string) {
  const pinColor =
    status === 'available' ? '#00E5CC' : status === 'busy' ? '#FFB300' : '#FF3D00'
  return L.divIcon({
    className: '',
    html: `<div style="width:26px;height:26px;background:rgba(10,14,23,0.85);backdrop-filter:blur(4px);border-radius:50%;border:2px solid ${pinColor};display:flex;align-items:center;justify-content:center;box-shadow:0 0 12px ${pinColor}80, inset 0 0 8px ${pinColor}40;font-size:13px;color:${pinColor};cursor:pointer;transition:transform 0.3s" onmouseover="this.style.transform='scale(1.2)'" onmouseout="this.style.transform='scale(1)'">⚡</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 26],
    popupAnchor: [0, -26],
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

// ─── Energy → Color ──────────────────────────────────────────────────────────

function energyToColor(cost: number, maxCost: number): string {
  if (maxCost <= 0) return '#00E5CC'
  const ratio = Math.min(cost / maxCost, 1)
  if (ratio < 0.33) return '#00E676' // green
  if (ratio < 0.66) return '#FFB300' // yellow
  return '#FF3D00' // red
}

function riskToColor(risk: string): string {
  if (risk === 'low') return '#00E676'
  if (risk === 'medium') return '#FFB300'
  return '#FF3D00'
}

// ─── Location Autocomplete ───────────────────────────────────────────────────

function LocationAutocomplete({
  placeholder,
  defaultValue,
  color,
  onSelect,
}: {
  placeholder: string
  defaultValue: string
  color: string
  onSelect?: (result: any) => void
}) {
  const [query, setQuery] = useState(defaultValue)
  const [results, setResults] = useState<any[]>([])
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    if (!query || query === defaultValue) {
      setResults([])
      return
    }
    const timer = setTimeout(() => {
      fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`
      )
        .then((r) => r.json())
        .then((data) => setResults(data || []))
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
        onFocus={() => {
          if (results.length > 0) setIsOpen(true)
        }}
        type="text"
      />
      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-surface-100/90 backdrop-blur-md border border-white/10 shadow-2xl z-[2000] overflow-hidden">
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
              <p className="text-sm font-semibold text-surface-900 truncate">
                {r.display_name.split(',')[0]}
              </p>
              <p className="text-[10px] text-surface-800/50 truncate mt-0.5">
                {r.display_name}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Pipeline Status Stepper (Enhanced — persistent with completion state) ───

const PIPELINE_STEPS: { key: PipelineStage; label: string; sublabel: string; icon: string }[] = [
  { key: 'fetching_route', label: 'Fetching Route', sublabel: 'OpenRouteService API', icon: 'route' },
  { key: 'sampling_elevation', label: 'Sampling Elevation', sublabel: 'Open-Elevation API', icon: 'terrain' },
  { key: 'reading_weather', label: 'Reading Weather', sublabel: 'OpenWeatherMap API', icon: 'cloud' },
  { key: 'checking_chargers', label: 'Checking Chargers', sublabel: 'Open Charge Map API', icon: 'ev_station' },
  { key: 'ml_prediction', label: 'ML Prediction', sublabel: 'Neural energy model', icon: 'psychology' },
  { key: 'physics_validation', label: 'Physics Validation', sublabel: 'Cross-checking result', icon: 'science' },
]

function PipelineStepper({
  stage,
  prediction,
}: {
  stage: PipelineStage
  prediction: PredictionResult | null
}) {
  const isRunning = stage !== 'idle' && stage !== 'complete' && stage !== 'error'
  const isComplete = stage === 'complete'

  if (stage === 'idle') return null

  const activeIdx = isComplete
    ? PIPELINE_STEPS.length
    : PIPELINE_STEPS.findIndex((s) => s.key === stage)

  // Build final step label
  const finalLabel = isComplete
    ? prediction?.method === 'ml_validated'
      ? '✓ ML Validated — Route Selected'
      : prediction?.method === 'physics_fallback'
      ? '⚠ Physics Fallback — Route Selected'
      : '✓ Route Selected'
    : ''

  return (
    <div
      className={`glass-dark p-5 border ${
        isComplete ? 'border-neon-green/30' : stage === 'error' ? 'border-neon-red/30' : 'border-neon-blue/20'
      }`}
      style={{ animation: 'slideUp 0.4s ease-out' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-[9px] font-mono font-bold uppercase tracking-widest text-neon-blue/60">
          {isComplete ? 'Pipeline Complete' : stage === 'error' ? 'Pipeline Error' : 'Computing Route'}
        </p>
        {isRunning && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full border-2 border-neon-blue/30 border-t-neon-blue animate-spin" />
            <span className="text-[9px] font-mono text-neon-blue/50">Processing</span>
          </div>
        )}
        {isComplete && (
          <span className="text-[9px] font-mono text-neon-green/70 flex items-center gap-1">
            <span className="material-symbols-outlined text-[12px]">check_circle</span>
            Done
          </span>
        )}
      </div>

      {/* Steps */}
      <div className="flex items-start gap-0.5">
        {PIPELINE_STEPS.map((step, i) => {
          const isDone = i < activeIdx
          const isActive = i === activeIdx && isRunning
          return (
            <div key={step.key} className="flex items-center gap-0.5 flex-1">
              <div className="flex flex-col items-center gap-1 min-w-[36px]">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 transition-all duration-500 ${
                    isDone
                      ? 'bg-neon-green/15 text-neon-green border border-neon-green/30'
                      : isActive
                      ? 'bg-neon-blue/20 text-neon-blue border border-neon-blue/40 shadow-[0_0_14px_rgba(0,180,216,0.5)] animate-pulse'
                      : 'bg-surface-200/20 text-surface-800/25 border border-white/5'
                  }`}
                >
                  {isDone ? (
                    <span className="material-symbols-outlined text-[14px]">check</span>
                  ) : (
                    <span className="material-symbols-outlined text-[14px]">{step.icon}</span>
                  )}
                </div>
                <span
                  className={`text-[8px] font-mono text-center leading-tight transition-colors ${
                    isDone
                      ? 'text-neon-green/60'
                      : isActive
                      ? 'text-neon-blue'
                      : 'text-surface-800/20'
                  }`}
                >
                  {step.label.split(' ')[0]}
                </span>
              </div>
              {i < PIPELINE_STEPS.length - 1 && (
                <div
                  className={`flex-1 h-[2px] rounded-full transition-all duration-500 mt-[-12px] ${
                    isDone ? 'bg-neon-green/40' : 'bg-surface-200/15'
                  }`}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Active step description */}
      {isRunning && (
        <div className="mt-3 flex items-center justify-center gap-2">
          <span className="text-[10px] font-mono text-surface-900/70">
            {PIPELINE_STEPS[activeIdx]?.label}
          </span>
          <span className="text-[9px] font-mono text-surface-800/40">
            ({PIPELINE_STEPS[activeIdx]?.sublabel})
          </span>
        </div>
      )}

      {/* Completion badge */}
      {isComplete && finalLabel && (
        <div className="mt-3 text-center">
          <span className="text-[11px] font-mono font-bold text-neon-green/80">
            {finalLabel}
          </span>
        </div>
      )}
    </div>
  )
}

// ─── Decision Explanation Card ───────────────────────────────────────────────

function DecisionCard({
  prediction,
  routeCtx,
}: {
  prediction: PredictionResult
  routeCtx: RouteContext
}) {
  const [expanded, setExpanded] = useState(false)
  const route = routeCtx.routes[routeCtx.selectedRouteIndex]

  // Build plain-language method description
  const methodText =
    prediction.method === 'ml_validated'
      ? 'ML neural network prediction, cross-validated by physics engine'
      : prediction.method === 'physics_fallback'
      ? 'Physics engine (ML confidence was too low or diverged)'
      : prediction.method === 'ml'
      ? 'ML neural network prediction (unvalidated)'
      : 'Hybrid ML + physics blend'

  const methodBadge =
    prediction.method === 'ml_validated'
      ? { text: 'ML VALIDATED', color: 'accent-success' }
      : prediction.method === 'physics_fallback'
      ? { text: 'PHYSICS FALLBACK', color: 'accent-warning' }
      : prediction.method === 'ml'
      ? { text: 'ML ONLY', color: 'neon-blue' }
      : { text: 'HYBRID', color: 'neon-purple' }

  return (
    <div
      className="glass-dark p-5 border border-neon-blue/15 relative overflow-hidden"
      style={{ animation: 'slideUp 0.5s ease-out' }}
    >
      <div className="absolute -top-10 -right-10 w-32 h-32 bg-accent-success/10 rounded-full blur-[30px]" />

      {/* ── Header with method badge ── */}
      <div className="flex items-center justify-between mb-4 relative z-10">
        <h4 className="text-[10px] font-mono font-bold text-surface-800/60 uppercase tracking-widest">
          Route Decision
        </h4>
        <span
          className={`px-2.5 py-1 border text-[9px] rounded-full font-mono font-bold uppercase tracking-widest shadow-sm bg-${methodBadge.color}/10 border-${methodBadge.color}/20 text-${methodBadge.color}`}
          style={{
            backgroundColor:
              prediction.method === 'ml_validated'
                ? 'rgba(0,245,160,0.1)'
                : prediction.method === 'physics_fallback'
                ? 'rgba(255,179,0,0.1)'
                : 'rgba(0,180,216,0.1)',
            borderColor:
              prediction.method === 'ml_validated'
                ? 'rgba(0,245,160,0.2)'
                : prediction.method === 'physics_fallback'
                ? 'rgba(255,179,0,0.2)'
                : 'rgba(0,180,216,0.2)',
            color:
              prediction.method === 'ml_validated'
                ? '#00f5a0'
                : prediction.method === 'physics_fallback'
                ? '#FFB300'
                : '#00b4d8',
          }}
        >
          {methodBadge.text}
        </span>
      </div>

      {/* ── Why this route — always visible ── */}
      <div className="mb-4 p-3 bg-surface-100/30 border border-white/5 relative z-10">
        <p className="text-[9px] font-mono text-neon-blue/50 uppercase tracking-widest mb-1.5">
          Why This Route
        </p>
        <p className="text-[11px] text-surface-900/80 leading-relaxed">
          {prediction.route_explanation}
        </p>
        <p className="text-[10px] text-surface-800/50 mt-1.5 font-mono">
          Method: {methodText}
        </p>
      </div>

      {/* ── Energy headline ── */}
      <div className="mb-4 relative z-10">
        <div className="flex justify-between items-end mb-1.5">
          <span className="text-xs font-mono text-surface-800/50">Energy Needed</span>
          <span className="text-2xl font-headline font-bold text-neon-blue glow-neon">
            {prediction.energy_kwh.toFixed(1)}{' '}
            <span className="text-[10px] font-sans text-neon-blue/50">kWh</span>
          </span>
        </div>
        <div className="w-full bg-surface-200/50 h-1.5 rounded-full overflow-hidden">
          <div
            className="bg-gradient-to-r from-neon-blue to-neon-green h-full rounded-full shadow-[0_0_10px_#00f5a0]"
            style={{
              width: `${Math.min(
                100,
                (prediction.energy_kwh / DEFAULT_VEHICLE.battery_capacity_kwh) * 100
              )}%`,
            }}
          />
        </div>
      </div>

      {/* ── Stats grid ── */}
      <div className="grid grid-cols-2 gap-2.5 mb-4 relative z-10">
        <div className="bg-surface-100/50 backdrop-blur p-3 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            Arrival SoC
          </p>
          <p
            className={`text-xl font-headline font-bold ${
              prediction.arrival_soc > 30
                ? 'text-accent-success'
                : prediction.arrival_soc > 15
                ? 'text-accent-warning'
                : 'text-neon-red'
            }`}
          >
            {prediction.arrival_soc.toFixed(0)}%
          </p>
        </div>
        <div className="bg-surface-100/50 backdrop-blur p-3 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            Confidence
          </p>
          <div className="flex items-center gap-1.5">
            <p className="text-xl font-headline font-bold text-surface-900">
              {(prediction.confidence * 100).toFixed(0)}%
            </p>
            {prediction.confidenceLevel === 'HIGH' && (
              <span className="material-symbols-outlined text-accent-success text-sm drop-shadow-[0_0_5px_#00f5a0]">
                verified
              </span>
            )}
          </div>
        </div>
        <div className="bg-surface-100/50 backdrop-blur p-3 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            SoH Impact
          </p>
          <p className="text-lg font-headline font-bold text-surface-900">
            {prediction.soh_impact.toFixed(2)}%
          </p>
        </div>
        <div className="bg-surface-100/50 backdrop-blur p-3 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            Route Cost
          </p>
          <p className="text-lg font-headline font-bold text-neon-purple">
            {(prediction.route_cost * 100).toFixed(0)}
            <span className="text-[10px] text-surface-800/40">/100</span>
          </p>
        </div>
      </div>

      {/* ── Route summary line ── */}
      <div className="mb-3 p-2.5 bg-surface-100/30 border border-white/5 relative z-10">
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-surface-800/50">
            {(route.distance_m / 1000).toFixed(1)} km · {Math.round(route.duration_s / 60)} min
          </span>
          <span className="text-surface-800/50">
            Avg {prediction.avg_speed_kmh.toFixed(0)} km/h
          </span>
        </div>
      </div>

      {/* ── Charger recommendation ── */}
      {prediction.charger_stop && (
        <div className="mb-3 p-3 bg-accent-warning/5 border border-accent-warning/20 relative z-10">
          <div className="flex items-start gap-2">
            <span className="material-symbols-outlined text-accent-warning text-[16px] mt-0.5">
              ev_station
            </span>
            <div>
              <p className="text-[10px] font-mono font-bold text-accent-warning mb-0.5">
                Charger Stop Recommended
              </p>
              <p className="text-[10px] font-mono text-accent-warning/70">
                {prediction.charger_stop}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Weather & Elevation summary ── */}
      <div className="grid grid-cols-2 gap-2 mb-3 relative z-10">
        <div className="p-2.5 bg-surface-100/30 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">cloud</span>
            Weather
          </p>
          <p className="text-[10px] text-surface-900/70 leading-snug">{prediction.weather_impact}</p>
        </div>
        <div className="p-2.5 bg-surface-100/30 border border-white/5">
          <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-1">
            <span className="material-symbols-outlined text-[10px] align-middle mr-0.5">terrain</span>
            Elevation
          </p>
          <p className="text-[10px] text-surface-900/70 leading-snug">{prediction.elevation_impact}</p>
        </div>
      </div>

      {/* ── Warnings ── */}
      {routeCtx.warnings.length > 0 && (
        <div className="mb-3 space-y-1 relative z-10">
          {routeCtx.warnings.map((w, i) => (
            <p key={i} className="text-[9px] font-mono text-accent-warning/70">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}

      {/* ── Expandable energy breakdown ── */}
      {expanded && (
        <div className="mb-3 relative z-10" style={{ animation: 'slideUp 0.3s ease-out' }}>
          <div className="p-3 bg-surface-100/30 border border-white/5">
            <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-2">
              Energy Breakdown
            </p>
            <div className="space-y-1.5">
              {[
                { label: 'Rolling Resistance', val: prediction.energy_breakdown.rolling, color: 'text-surface-900' },
                { label: 'Aerodynamic Drag', val: prediction.energy_breakdown.aero, color: 'text-surface-900' },
                { label: 'Grade (hills)', val: prediction.energy_breakdown.grade, color: 'text-surface-900' },
                { label: 'Auxiliary (HVAC)', val: prediction.energy_breakdown.auxiliary, color: 'text-surface-900' },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-surface-800/50">{item.label}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1 bg-surface-200/30 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-neon-blue/50 rounded-full"
                        style={{
                          width: `${Math.min(100, (item.val / prediction.energy_kwh) * 100)}%`,
                        }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-surface-900 w-14 text-right">
                      {item.val.toFixed(2)} kWh
                    </span>
                  </div>
                </div>
              ))}
              <div className="flex items-center justify-between border-t border-white/5 pt-1.5 mt-1.5">
                <span className="text-[10px] font-mono text-neon-green/70">Regen Recovered</span>
                <span className="text-[10px] font-mono text-neon-green font-bold">
                  −{prediction.energy_breakdown.regen_recovered.toFixed(2)} kWh
                </span>
              </div>
            </div>
          </div>

          {/* Segment risk overview */}
          {prediction.segment_costs.length > 0 && (
            <div className="p-3 bg-surface-100/30 border border-white/5 mt-2">
              <p className="text-[9px] font-mono text-surface-800/50 uppercase tracking-widest mb-2">
                Segment Energy Map
              </p>
              <div className="flex gap-[2px] h-6 rounded-sm overflow-hidden">
                {prediction.segment_costs.map((cost, i) => {
                  const maxCost = Math.max(...prediction.segment_costs.map(Math.abs), 0.01)
                  const color = energyToColor(Math.abs(cost), maxCost)
                  return (
                    <div
                      key={i}
                      className="flex-1 transition-all hover:opacity-70"
                      style={{ backgroundColor: color }}
                      title={`Segment ${i + 1}: ${Math.abs(cost).toFixed(2)} kWh`}
                    />
                  )
                })}
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[8px] font-mono text-neon-green/50">Low energy</span>
                <span className="text-[8px] font-mono text-neon-red/50">High energy</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Action buttons ── */}
      <div className="flex gap-2 relative z-10">
        <button
          className="flex-1 text-xs font-bold text-surface-900 py-3 rounded-lg bg-surface-100/80 hover:bg-surface-200 transition-colors border border-white/5"
          aria-label="Toggle route details"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Less' : 'Breakdown'}
        </button>
        <button
          className="flex-1 text-xs font-bold text-brand-bg py-3 rounded-lg bg-neon-green border border-neon-green hover:bg-[#00d68b] transition-colors shadow-[0_0_15px_rgba(0,245,160,0.3)] tracking-wide"
          aria-label="Start Navigation"
        >
          Start Nav
        </button>
      </div>
    </div>
  )
}

// ─── Maneuver icon mapping ───────────────────────────────────────────────────

function getStepIcon(type: number, idx: number, total: number): string {
  if (idx === 0) return 'trip_origin'
  if (idx === total - 1) return 'flag'
  if (type === 0 || type === 1) return 'turn_left'
  if (type === 2 || type === 3) return 'turn_right'
  if (type === 6) return 'roundabout_right'
  if (type === 4 || type === 5) return 'straight'
  return 'arrow_forward'
}

// ═════════════════════════════════════════════════════════════════════════════
// Main Component
// ═════════════════════════════════════════════════════════════════════════════

export default function RoutePlanner() {
  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<L.Map | null>(null)
  const layersRef = useRef<L.Layer[]>([])
  const location = useLocation()
  const navigate = useNavigate()
  const defaultDest = location.state?.destination || ''
  const { isMobile } = useWindowSize()

  // ── State
  const [showPanel, setShowPanel] = useState(false)
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle')
  const [routeCtx, setRouteCtx] = useState<RouteContext | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [, setAllPredictions] = useState<PredictionResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [originData, setOriginData] = useState<LatLng | null>(null)
  const [destData, setDestData] = useState<LatLng | null>(null)
  const [originName, setOriginName] = useState('Amsterdam Centraal')
  const [destName, setDestName] = useState(defaultDest || 'Rotterdam Centraal')

  // ── Initialize Map
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return

    const map = L.map(mapRef.current, {
      center: [52.34, 4.84],
      zoom: 12,
      zoomControl: false,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
    }).addTo(map)

    L.control.zoom({ position: 'topright' }).addTo(map)
    leafletMap.current = map

    return () => {
      map.remove()
      leafletMap.current = null
    }
  }, [])

  // ── Clear map layers
  const clearLayers = useCallback(() => {
    layersRef.current.forEach((l) => l.remove())
    layersRef.current = []
  }, [])

  // ── Draw route on map
  const drawRoute = useCallback(
    (ctx: RouteContext, pred: PredictionResult, preds: PredictionResult[]) => {
      const map = leafletMap.current
      if (!map) return
      clearLayers()

      const primary = ctx.routes[ctx.selectedRouteIndex]

      // Draw alternate routes (dimmed, dashed, with label)
      ctx.routes.forEach((route, i) => {
        if (i === ctx.selectedRouteIndex) return
        const latlngs = route.geometry.map((c) => [c.lat, c.lng] as [number, number])
        const altPred = preds[i]

        // Dashed dimmed polyline
        const line = L.polyline(latlngs, {
          color: '#4a5568',
          weight: 3,
          opacity: 0.35,
          dashArray: '10 8',
        }).addTo(map)

        // Popup with comparison info
        line.bindPopup(
          `<div class="text-white text-xs" style="min-width:140px">
            <div style="font-weight:bold;margin-bottom:4px;color:#FFB300">Alternative Route ${i + 1}</div>
            <div>${(route.distance_m / 1000).toFixed(1)} km · ${Math.round(route.duration_s / 60)} min</div>
            ${altPred ? `<div>Energy: ${altPred.energy_kwh.toFixed(1)} kWh</div>
            <div>Arrival SoC: ${altPred.arrival_soc.toFixed(0)}%</div>` : ''}
            <div style="color:#888;margin-top:4px;font-size:9px">Click to select this route</div>
          </div>`
        )

        // Click to switch
        line.on('click', () => {
          const newCtx = { ...ctx, selectedRouteIndex: i }
          const newPred = runHybridPrediction(newCtx, DEFAULT_VEHICLE)
          setRouteCtx(newCtx)
          setPrediction(newPred)
          drawRoute(newCtx, newPred, preds)
        })

        layersRef.current.push(line)
      })

      // Draw primary route — color-coded segments by energy + degradation risk
      if (pred.segment_costs.length > 0 && primary.segments.length > 0) {
        const maxCost = Math.max(...pred.segment_costs.map(Math.abs), 0.01)

        primary.segments.forEach((seg, i) => {
          const cost = Math.abs(pred.segment_costs[i] || 0)
          const baseColor = energyToColor(cost, maxCost)
          const risk = seg.riskLevel || (cost / maxCost > 0.66 ? 'high' : cost / maxCost > 0.33 ? 'medium' : 'low')

          const startIdx = Math.round((i / primary.segments.length) * primary.geometry.length)
          const endIdx = Math.round(((i + 1) / primary.segments.length) * primary.geometry.length)
          const segCoords = primary.geometry
            .slice(startIdx, endIdx + 1)
            .map((c) => [c.lat, c.lng] as [number, number])

          if (segCoords.length >= 2) {
            // Glow underline
            const glow = L.polyline(segCoords, {
              color: baseColor,
              weight: 12,
              opacity: 0.15,
            }).addTo(map)
            layersRef.current.push(glow)

            // Main segment line
            const line = L.polyline(segCoords, {
              color: baseColor,
              weight: 5,
              opacity: 0.9,
              lineCap: 'round',
              lineJoin: 'round',
            }).addTo(map)

            line.bindPopup(
              `<div class="text-white text-xs" style="min-width:120px">
                <div style="font-weight:bold;margin-bottom:3px">Segment ${i + 1}</div>
                <div>Energy: ${cost.toFixed(2)} kWh</div>
                <div>Gradient: ${seg.gradient.toFixed(1)}%</div>
                <div>Risk: <span style="color:${riskToColor(risk)}">${risk.toUpperCase()}</span></div>
              </div>`
            )
            layersRef.current.push(line)
          }
        })
      } else {
        // Fallback: single polyline with glow
        const latlngs = primary.geometry.map((c) => [c.lat, c.lng] as [number, number])
        const glow = L.polyline(latlngs, { color: '#00E5CC', weight: 14, opacity: 0.12 }).addTo(map)
        layersRef.current.push(glow)
        const line = L.polyline(latlngs, { color: '#00E5CC', weight: 5, opacity: 0.9 }).addTo(map)
        layersRef.current.push(line)
      }

      // Start/End markers
      const start = primary.geometry[0]
      const end = primary.geometry[primary.geometry.length - 1]

      const startMarker = L.marker([start.lat, start.lng], {
        icon: createEndpointIcon('A', '#00b4d8'),
      })
        .addTo(map)
        .bindPopup(
          `<div class="text-white"><b>Start:</b> ${ctx.originName}</div>`
        )
      layersRef.current.push(startMarker)

      const endMarker = L.marker([end.lat, end.lng], {
        icon: createEndpointIcon('B', '#00f5a0'),
      })
        .addTo(map)
        .bindPopup(
          `<div class="text-white"><b>Destination:</b> ${ctx.destinationName}</div>`
        )
      layersRef.current.push(endMarker)

      // Charging station pins
      ctx.chargers.forEach((ch) => {
        const marker = L.marker([ch.lat, ch.lng], {
          icon: createStationIcon(ch.status),
        })
          .addTo(map)
          .bindPopup(
            `<div class="text-white text-xs">
              <b>${ch.name}</b><br/>
              ${ch.powerKw} kW · ${ch.operator}<br/>
              <span style="color:${ch.status === 'available' ? '#00E5CC' : ch.status === 'busy' ? '#FFB300' : '#FF3D00'}">${ch.status.toUpperCase()}</span>
              · ${ch.numPorts} ports
            </div>`
          )
        layersRef.current.push(marker)
      })

      // Fit bounds
      const allPoints = primary.geometry.map((c) => [c.lat, c.lng] as [number, number])
      if (allPoints.length > 0) {
        map.fitBounds(L.latLngBounds(allPoints), { padding: [60, 60] })
      }
    },
    [clearLayers]
  )

  // ── Execute Route Planning
  const handlePlanRoute = useCallback(async () => {
    let origin = originData
    let dest = destData

    if (!origin) {
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(originName)}&limit=1`
        )
        const data = await res.json()
        if (data[0]) {
          origin = { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) }
        }
      } catch {
        /* will error below */
      }
    }
    if (!dest) {
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(destName)}&limit=1`
        )
        const data = await res.json()
        if (data[0]) {
          dest = { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) }
        }
      } catch {
        /* will error below */
      }
    }

    if (!origin || !dest) {
      setError('Please select both origin and destination')
      return
    }

    setError(null)
    setPrediction(null)
    setRouteCtx(null)
    setAllPredictions([])
    setPipelineStage('fetching_route')

    try {
      // Orchestrate all API calls
      const ctx = await orchestrateRoute(
        origin,
        dest,
        originName,
        destName,
        (stage) => setPipelineStage(stage)
      )

      // Run ML prediction
      setPipelineStage('ml_prediction')
      await new Promise((r) => setTimeout(r, 400)) // visual pause for UI

      // Run physics validation
      setPipelineStage('physics_validation')
      await new Promise((r) => setTimeout(r, 400)) // visual pause for UI

      const { bestIndex, predictions } = selectBestRoute(ctx)

      // Update context with best route
      ctx.selectedRouteIndex = bestIndex
      const bestPred = predictions[bestIndex]

      setRouteCtx(ctx)
      setPrediction(bestPred)
      setAllPredictions(predictions)
      setPipelineStage('complete')

      // Draw on map
      drawRoute(ctx, bestPred, predictions)

      // Hide stepper after 4 seconds to prevent overlap with the decision card
      setTimeout(() => {
        setPipelineStage('idle')
      }, 4000)
    } catch (err: any) {
      console.error('[RoutePlanner]', err)
      setError(err.message || 'Route planning failed')
      setPipelineStage('error')
    }
  }, [originData, destData, originName, destName, drawRoute])

  // ── Build route step list from ORS instructions
  const routeSteps = routeCtx
    ? (() => {
        const route = routeCtx.routes[routeCtx.selectedRouteIndex]
        const instructions = route.instructions.slice(0, 8)
        return instructions.map((inst, i) => ({
          icon: getStepIcon(inst.type, i, instructions.length),
          text: inst.text,
          sub: `${(inst.distance_m / 1000).toFixed(1)} km`,
          dist: `${Math.round(inst.duration_s / 60)} min`,
          highlight: inst.text.toLowerCase().includes('charg'),
        }))
      })()
    : []

  const primaryRoute = routeCtx?.routes[routeCtx.selectedRouteIndex]

  return (
    <div className={`flex-1 flex h-full overflow-hidden relative ${isMobile ? 'flex-col' : ''}`}>
      {/* ═══ Full-width Map ═══ */}
      <div className="flex-1 relative bg-brand-bg">
        <div ref={mapRef} className="h-full w-full z-0 opacity-90" />
        <div className="absolute inset-0 z-[5] pointer-events-none shadow-[inset_0_0_150px_rgba(10,14,23,1)]" />

        {/* ── Top search bar overlay ── */}
        <div className="absolute top-5 left-5 right-5 lg:left-[440px] lg:right-auto lg:w-[400px] z-[1000]">
          <GeoSearch
            variant="glass"
            color="blue"
            placeholder="Search places, addresses…"
            onSelect={(result: GeoSearchResult) => {
              setDestData({ lat: parseFloat(result.lat), lng: parseFloat(result.lon) })
              setDestName(result.display_name.split(',')[0])
              const map = leafletMap.current
              if (map) map.flyTo([parseFloat(result.lat), parseFloat(result.lon)], 14, { animate: true, duration: 1 })
            }}
          />
        </div>

        {/* ── Map layer toggle ── */}
        <div className="absolute top-5 right-5 z-[1000] flex flex-col gap-2">
          <button
            onClick={() => navigate('/charging-stations')}
            aria-label="Navigate to Charging Stations"
            className="px-3 py-2 rounded-lg text-xs font-bold font-mono tracking-widest uppercase flex items-center gap-2 transition-all duration-300 glass-dark text-neon-blue border border-neon-blue/40 shadow-[0_0_15px_rgba(0,180,216,0.2)] hover:bg-neon-blue/10 cursor-pointer"
          >
            <span className="material-symbols-outlined text-sm" aria-hidden="true">
              ev_station
            </span>
            Chargers
          </button>
        </div>

        {/* ── Pipeline Status Stepper (Center Overlay — persistent) ── */}
        {pipelineStage !== 'idle' && (
          <div
            className={`absolute z-[1000] ${
              isMobile ? 'bottom-2 left-2 right-2' : 'top-20 lg:left-[calc(50%+210px)] -translate-x-1/2 w-full max-w-[600px] px-5'
            }`}
          >
            <PipelineStepper stage={pipelineStage} prediction={prediction} />
          </div>
        )}

        {/* ── Error Overlay ── */}
        {error && (
          <div
            className={`absolute z-[1000] ${
              isMobile ? 'bottom-2 left-2 right-2' : 'bottom-5 right-5 w-[360px]'
            }`}
            style={{ animation: 'slideUp 0.3s ease-out' }}
          >
            <div className="glass-dark p-5 border border-neon-red/30">
              <div className="flex items-center gap-3 mb-2">
                <span className="material-symbols-outlined text-neon-red">error</span>
                <h4 className="text-sm font-headline font-bold text-neon-red">Route Error</h4>
              </div>
              <p className="text-xs text-surface-800/70">{error}</p>
              <button
                className="mt-3 w-full text-xs font-bold text-surface-900 py-2.5 rounded-lg bg-surface-100/80 hover:bg-surface-200 transition-colors border border-white/5"
                onClick={() => { setError(null); setPipelineStage('idle') }}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* ══ Decision Card Overlay ══ */}
        {prediction && routeCtx && (
          <div
            className={`absolute z-[1000] ${
              isMobile ? 'bottom-2 left-2 right-2 w-auto' : 'bottom-5 right-5 w-[380px] max-h-[calc(100%-100px)] overflow-y-auto custom-scrollbar'
            }`}
          >
            <DecisionCard prediction={prediction} routeCtx={routeCtx} />
          </div>
        )}
      </div>

      {/* ═══ Left Panel — Route Config ═══ */}
      {isMobile && (
        <button
          onClick={() => setShowPanel(!showPanel)}
          className="absolute top-20 left-3 z-[1100] w-11 h-11 rounded-full glass-dark border border-neon-blue/30 flex items-center justify-center text-neon-blue shadow-[0_0_15px_rgba(0,180,216,0.3)] active:scale-95 transition-transform"
          aria-label="Toggle route panel"
        >
          <span className="material-symbols-outlined text-xl">
            {showPanel ? 'close' : 'route'}
          </span>
        </button>
      )}
      <div
        className={`absolute z-[1000] flex flex-col pointer-events-none ${
          isMobile
            ? `top-0 left-0 right-0 bottom-0 p-3 pt-[84px] ${showPanel ? '' : 'hidden'}`
            : 'top-0 left-0 bottom-0 w-[420px] p-5'
        }`}
      >
        <div
          className={`glass-dark overflow-hidden flex flex-col pointer-events-auto border border-neon-blue/20 ${
            isMobile ? 'max-h-[60vh]' : 'flex-1'
          }`}
          style={{ animation: 'slideRight 0.4s ease-out' }}
        >
          {/* Header */}
          <div className="p-6 pb-4 relative overflow-hidden shrink-0">
            <div className="absolute top-0 right-0 w-32 h-32 bg-neon-blue/10 rounded-full blur-[40px] -mr-10 -mt-10" />
            <h3 className="text-2xl font-headline font-bold text-surface-900 mb-1 relative z-10">
              Route Control
            </h3>
            <p className="text-[10px] font-mono text-neon-blue/60 tracking-widest uppercase relative z-10">
              Neural Navigation Active
            </p>
          </div>

          {/* Origin/Destination inputs */}
          <div className="px-6 pb-5 shrink-0">
            <div className="flex gap-4">
              <div className="flex flex-col items-center py-3 shrink-0">
                <div className="w-3.5 h-3.5 rounded-full bg-surface-50 border-[3px] border-neon-blue shadow-[0_0_10px_rgba(0,180,216,0.6)] z-10" />
                <div className="w-[1px] flex-1 bg-gradient-to-b from-neon-blue via-surface-200 to-neon-green my-1" />
                <div className="w-3.5 h-3.5 rounded-full bg-surface-50 border-[3px] border-neon-green shadow-[0_0_10px_rgba(0,245,160,0.6)] z-10" />
              </div>

              <div className="flex-1 space-y-3">
                <LocationAutocomplete
                  placeholder="Start point..."
                  defaultValue="Amsterdam Centraal"
                  color="blue"
                  onSelect={(r: any) => {
                    setOriginData({ lat: parseFloat(r.lat), lng: parseFloat(r.lon) })
                    setOriginName(r.display_name.split(',')[0])
                  }}
                />
                <LocationAutocomplete
                  placeholder="Destination..."
                  defaultValue={defaultDest || 'Rotterdam Centraal'}
                  color="green"
                  onSelect={(r: any) => {
                    setDestData({ lat: parseFloat(r.lat), lng: parseFloat(r.lon) })
                    setDestName(r.display_name.split(',')[0])
                  }}
                />
              </div>
            </div>
          </div>

          {/* Route stats */}
          <div className="p-6 pt-0 flex-1 overflow-y-auto custom-scrollbar">
            <div className="grid grid-cols-3 gap-3 mb-6">
              <div className="text-center p-3 glass-panel border-t border-neon-blue/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-blue/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-blue/60 uppercase tracking-widest relative z-10">
                  Distance
                </p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">
                  {primaryRoute ? (primaryRoute.distance_m / 1000).toFixed(1) : '—'}
                  <span className="text-[10px] ml-0.5 text-surface-800/40">km</span>
                </p>
              </div>
              <div className="text-center p-3 glass-panel border-t border-neon-purple/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-purple/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-purple/60 uppercase tracking-widest relative z-10">
                  Duration
                </p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">
                  {primaryRoute ? Math.round(primaryRoute.duration_s / 60) : '—'}
                  <span className="text-[10px] ml-0.5 text-surface-800/40">min</span>
                </p>
              </div>
              <div className="text-center p-3 glass-panel border-t border-neon-green/30 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-neon-green/5 to-transparent" />
                <p className="text-[9px] font-mono text-neon-green/60 uppercase tracking-widest relative z-10">
                  Energy
                </p>
                <p className="text-xl font-headline font-bold text-surface-900 mt-1 relative z-10">
                  {prediction ? prediction.energy_kwh.toFixed(1) : '—'}
                  <span className="text-[10px] ml-0.5 text-surface-800/40">kWh</span>
                </p>
              </div>
            </div>

            {/* Route steps */}
            <div className="space-y-4 relative">
              {routeSteps.length > 0 && (
                <div className="absolute left-4 top-2 bottom-6 w-px bg-surface-200/30 z-0" />
              )}
              {routeSteps.length > 0
                ? routeSteps.map((step, i) => (
                    <div key={i} className="flex items-start gap-4 relative z-10 group">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border border-brand-bg transition-all ${
                          step.highlight
                            ? 'bg-neon-blue/20 text-neon-blue border-[1.5px] border-neon-blue shadow-[0_0_10px_rgba(0,180,216,0.3)]'
                            : 'bg-surface-100 text-surface-800/50 group-hover:bg-surface-200 group-hover:text-surface-900'
                        }`}
                      >
                        <span className="material-symbols-outlined text-[16px]">
                          {step.icon}
                        </span>
                      </div>
                      <div className="flex-1 min-w-0 pt-1">
                        <p
                          className={`text-sm font-semibold tracking-wide ${
                            step.highlight ? 'text-neon-blue' : 'text-surface-900'
                          }`}
                        >
                          {step.text}
                        </p>
                        <p className="text-[10px] font-mono text-surface-800/40 mt-0.5">
                          {step.sub}
                        </p>
                      </div>
                      {step.dist && (
                        <span className="text-[10px] font-mono text-surface-800/30 shrink-0 pt-1.5">
                          {step.dist}
                        </span>
                      )}
                    </div>
                  ))
                : (
                    <div className="text-center py-8">
                      <span className="material-symbols-outlined text-3xl text-surface-800/20 mb-2">
                        route
                      </span>
                      <p className="text-xs text-surface-800/30 font-mono">
                        Enter origin & destination, then tap Start Sequence
                      </p>
                    </div>
                  )}
            </div>
          </div>

          {/* Start Sequence Button */}
          <div className="p-6 bg-surface-50/50 border-t border-white/5 shrink-0 backdrop-blur-md">
            <button
              className={`w-full font-extrabold tracking-widest uppercase py-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 relative overflow-hidden group ${
                pipelineStage !== 'idle' && pipelineStage !== 'complete' && pipelineStage !== 'error'
                  ? 'bg-surface-200/50 text-surface-800/40 cursor-wait'
                  : 'bg-gradient-to-r from-neon-blue to-neon-green hover:from-[#00c5eb] hover:to-[#17ffae] text-brand-bg shadow-[0_4px_20px_rgba(0,180,216,0.25)]'
              }`}
              aria-label="Start Sequence Vector Calculation"
              onClick={handlePlanRoute}
              disabled={
                pipelineStage !== 'idle' &&
                pipelineStage !== 'complete' &&
                pipelineStage !== 'error'
              }
            >
              <div
                className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"
                aria-hidden="true"
              />
              {pipelineStage !== 'idle' &&
              pipelineStage !== 'complete' &&
              pipelineStage !== 'error' ? (
                <>
                  <div className="w-5 h-5 rounded-full border-2 border-surface-800/30 border-t-neon-blue animate-spin relative z-10" />
                  <span className="relative z-10 text-sm">Computing...</span>
                </>
              ) : (
                <>
                  <span
                    className="material-symbols-outlined text-xl drop-shadow-md relative z-10"
                    aria-hidden="true"
                  >
                    navigation
                  </span>
                  <span className="relative z-10">Start Sequence</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
