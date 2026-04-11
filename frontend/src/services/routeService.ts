/**
 * Route Orchestrator Service
 * 
 * Unified API layer for: OpenRouteService, Open-Elevation, OpenWeatherMap, Open Charge Map.
 * Orchestrates calls in sequence and merges into a single RouteContext.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface LatLng {
  lat: number
  lng: number
}

export interface RouteCandidate {
  index: number
  geometry: LatLng[]          // decoded polyline coords
  distance_m: number
  duration_s: number
  instructions: RouteInstruction[]
  segments: RouteSegment[]
}

export interface RouteInstruction {
  text: string
  distance_m: number
  duration_s: number
  type: number               // ORS maneuver type
  wayPoints: number[]
}

export interface RouteSegment {
  from: LatLng
  to: LatLng
  distance_m: number
  gradient: number           // percent grade
  elevation_gain: number
  elevation_loss: number
  energyCost?: number        // filled by prediction engine
  riskLevel?: 'low' | 'medium' | 'high'
}

export interface ElevationData {
  gain: number               // total meters gained
  loss: number               // total meters lost
  avgGradient: number        // average gradient %
  maxGradient: number
  profile: { lat: number; lng: number; elevation: number }[]
  isEstimated: boolean
}

export interface WeatherData {
  temperature: number        // Celsius
  windSpeed: number          // m/s
  windDirection: number      // degrees
  humidity: number
  rain: boolean
  description: string
  icon: string
  isEstimated: boolean
}

export interface ChargerInfo {
  id: number
  name: string
  lat: number
  lng: number
  powerKw: number
  operator: string
  status: 'available' | 'busy' | 'offline' | 'unknown'
  numPorts: number
  distance_m?: number        // distance from route
}

export interface RouteContext {
  routes: RouteCandidate[]
  selectedRouteIndex: number
  elevation: ElevationData
  weather: WeatherData
  chargers: ChargerInfo[]
  warnings: string[]
  originName: string
  destinationName: string
}

export type PipelineStage =
  | 'idle'
  | 'fetching_route'
  | 'sampling_elevation'
  | 'reading_weather'
  | 'checking_chargers'
  | 'ml_prediction'
  | 'physics_validation'
  | 'complete'
  | 'error'

// ─── API endpoints use our backend proxy to secure API keys ──────────────────

const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ─── 1. OpenRouteService — Route Geometry ────────────────────────────────────

export async function getRouteGeometry(
  origin: LatLng,
  destination: LatLng
): Promise<RouteCandidate[]> {
  const body = {
    coordinates: [
      [origin.lng, origin.lat],
      [destination.lng, destination.lat],
    ],
    alternative_routes: { target_count: 3, share_factor: 0.6, weight_factor: 1.6 },
    instructions: true,
    geometry: true,
  }

  const res = await fetch(`${baseURL}/api/external/ors/directions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`OpenRouteService proxy error ${res.status}: ${text}`)
  }

  const data = await res.json()
  const features = data.features || []

  return features.map((f: any, idx: number) => {
    const coords: LatLng[] = f.geometry.coordinates.map((c: number[]) => ({
      lat: c[1],
      lng: c[0],
    }))

    const props = f.properties || {}
    const summary = props.summary || {}
    const segments = props.segments || []

    const instructions: RouteInstruction[] = []
    segments.forEach((seg: any) => {
      (seg.steps || []).forEach((step: any) => {
        instructions.push({
          text: step.instruction || '',
          distance_m: step.distance || 0,
          duration_s: step.duration || 0,
          type: step.type || 0,
          wayPoints: step.way_points || [],
        })
      })
    })

    return {
      index: idx,
      geometry: coords,
      distance_m: summary.distance || 0,
      duration_s: summary.duration || 0,
      instructions,
      segments: [],
    } as RouteCandidate
  })
}

// ─── 2. Open-Elevation — Elevation Profile ───────────────────────────────────

function sampleCoords(coords: LatLng[], maxSamples: number = 80): LatLng[] {
  if (coords.length <= maxSamples) return coords
  const step = (coords.length - 1) / (maxSamples - 1)
  const sampled: LatLng[] = []
  for (let i = 0; i < maxSamples; i++) {
    sampled.push(coords[Math.round(i * step)])
  }
  return sampled
}

export async function getElevationProfile(coords: LatLng[]): Promise<ElevationData> {
  const sampled = sampleCoords(coords, 80)

  try {
    const body = {
      locations: sampled.map((c) => ({ latitude: c.lat, longitude: c.lng })),
    }

    const res = await fetch(`${baseURL}/api/external/elevation/lookup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!res.ok) throw new Error(`Elevation API error ${res.status}`)

    const data = await res.json()
    const results = data.results || []

    let gain = 0
    let loss = 0
    let maxGrad = 0
    const profile = results.map((r: any, i: number) => {
      if (i > 0) {
        const delta = r.elevation - results[i - 1].elevation
        if (delta > 0) gain += delta
        else loss += Math.abs(delta)

        // approximate horizontal distance between samples
        const dLat = r.latitude - results[i - 1].latitude
        const dLng = r.longitude - results[i - 1].longitude
        const hDist = Math.sqrt(dLat * dLat + dLng * dLng) * 111320 // rough m
        if (hDist > 0) {
          const grad = (Math.abs(delta) / hDist) * 100
          if (grad > maxGrad) maxGrad = grad
        }
      }
      return { lat: r.latitude, lng: r.longitude, elevation: r.elevation }
    })

    const totalHDist = coords.length > 1
      ? (() => {
          let d = 0
          for (let i = 1; i < profile.length; i++) {
            const dLat = profile[i].lat - profile[i - 1].lat
            const dLng = profile[i].lng - profile[i - 1].lng
            d += Math.sqrt(dLat * dLat + dLng * dLng) * 111320
          }
          return d
        })()
      : 1

    return {
      gain: Math.round(gain),
      loss: Math.round(loss),
      avgGradient: totalHDist > 0 ? Math.round(((gain + loss) / totalHDist) * 100 * 100) / 100 : 0,
      maxGradient: Math.round(maxGrad * 100) / 100,
      profile,
      isEstimated: false,
    }
  } catch (err) {
    console.warn('[Elevation] Fallback to flat profile:', err)
    return {
      gain: 0,
      loss: 0,
      avgGradient: 0,
      maxGradient: 0,
      profile: sampled.map((c) => ({ ...c, elevation: 0 })),
      isEstimated: true,
    }
  }
}

// ─── 3. OpenWeatherMap — Weather ─────────────────────────────────────────────

export async function getWeather(point: LatLng): Promise<WeatherData> {
  try {
    const res = await fetch(
      `${baseURL}/api/external/weather?lat=${point.lat}&lon=${point.lng}&units=metric`
    )
    if (!res.ok) throw new Error(`Weather API error ${res.status}`)

    const d = await res.json()
    return {
      temperature: d.main?.temp ?? 20,
      windSpeed: d.wind?.speed ?? 0,
      windDirection: d.wind?.deg ?? 0,
      humidity: d.main?.humidity ?? 50,
      rain: !!(d.rain || d.weather?.some((w: any) => w.main === 'Rain')),
      description: d.weather?.[0]?.description ?? 'clear',
      icon: d.weather?.[0]?.icon ?? '01d',
      isEstimated: false,
    }
  } catch (err) {
    console.warn('[Weather] Fallback to defaults:', err)
    return {
      temperature: 20,
      windSpeed: 0,
      windDirection: 0,
      humidity: 50,
      rain: false,
      description: 'assumed clear (data unavailable)',
      icon: '01d',
      isEstimated: true,
    }
  }
}

// ─── 4. Open Charge Map — Charging Stations ──────────────────────────────────

export async function getChargingStations(routeCoords: LatLng[]): Promise<ChargerInfo[]> {
  try {
    // Compute bounding box with 5 km buffer
    let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity
    routeCoords.forEach((c) => {
      if (c.lat < minLat) minLat = c.lat
      if (c.lat > maxLat) maxLat = c.lat
      if (c.lng < minLng) minLng = c.lng
      if (c.lng > maxLng) maxLng = c.lng
    })
    const buffer = 0.045 // ~5 km
    minLat -= buffer; maxLat += buffer; minLng -= buffer; maxLng += buffer

    // Center of route
    const centerLat = (minLat + maxLat) / 2
    const centerLng = (minLng + maxLng) / 2
    const distance = Math.max(
      haversineKm({ lat: minLat, lng: minLng }, { lat: maxLat, lng: maxLng }),
      20
    )

    const res = await fetch(
      `${baseURL}/api/external/ocm/poi?output=json&maxresults=30&compact=true&verbose=false&latitude=${centerLat}&longitude=${centerLng}&distance=${Math.ceil(distance / 2)}&distanceunit=KM`
    )

    if (!res.ok) throw new Error(`OCM error ${res.status}`)
    const data = await res.json()

    return (data || [])
      .filter((s: any) => s.AddressInfo?.Latitude && s.AddressInfo?.Longitude)
      .map((s: any) => {
        const maxPower = Math.max(
          ...(s.Connections?.map((c: any) => c.PowerKW || 0) || [0])
        )
        const statusId = s.StatusTypeID ?? s.StatusType?.ID ?? 2
        let status: ChargerInfo['status'] = 'unknown'
        if (statusId === 0 || statusId === 100) status = 'offline'
        else if (statusId === 5 || statusId === 75) status = 'busy'
        else if (statusId === 2 || statusId === 50) status = 'available'

        return {
          id: s.ID,
          name: s.AddressInfo.Title || 'Charging Station',
          lat: s.AddressInfo.Latitude,
          lng: s.AddressInfo.Longitude,
          powerKw: maxPower,
          operator: s.OperatorInfo?.Title || s.AddressInfo?.Town || 'Unknown',
          status,
          numPorts: s.NumberOfPoints || s.Connections?.length || 1,
        }
      })
  } catch (err) {
    console.warn('[OCM] Charger fetch failed:', err)
    return []
  }
}

// ─── 5. Orchestrator ─────────────────────────────────────────────────────────

export async function orchestrateRoute(
  origin: LatLng,
  destination: LatLng,
  originName: string,
  destinationName: string,
  onStageChange: (stage: PipelineStage) => void
): Promise<RouteContext> {
  // Broadcast wakeup detection — same event system as services/api.ts interceptors
  const slowTimer = setTimeout(() => window.dispatchEvent(new CustomEvent('backend:waking')), 3000)
  const done = () => { clearTimeout(slowTimer); window.dispatchEvent(new CustomEvent('backend:ready')) }

  try {
  const warnings: string[] = []

  // Step 1: Route Geometry
  onStageChange('fetching_route')
  const routes = await getRouteGeometry(origin, destination)
  if (routes.length === 0) throw new Error('No route found')

  const primaryRoute = routes[0]

  // Step 2: Elevation
  onStageChange('sampling_elevation')
  const elevation = await getElevationProfile(primaryRoute.geometry)
  if (elevation.isEstimated) warnings.push('Elevation data unavailable — using flat estimate')

  // Step 3: Weather (at origin, destination, midpoint — use midpoint as representative)
  onStageChange('reading_weather')
  const midIdx = Math.floor(primaryRoute.geometry.length / 2)
  const midPoint = primaryRoute.geometry[midIdx] || destination
  const weather = await getWeather(midPoint)
  if (weather.isEstimated) warnings.push('Weather data unavailable — using defaults')

  // Step 4: Charging stations
  onStageChange('checking_chargers')
  const chargers = await getChargingStations(primaryRoute.geometry)
  if (chargers.length === 0) warnings.push('No charging stations found along route')

  // Build segments with elevation data
  const segmentSize = Math.max(1, Math.floor(primaryRoute.geometry.length / 10))
  const segments: RouteSegment[] = []
  for (let i = 0; i < primaryRoute.geometry.length - 1; i += segmentSize) {
    const end = Math.min(i + segmentSize, primaryRoute.geometry.length - 1)
    const from = primaryRoute.geometry[i]
    const to = primaryRoute.geometry[end]

    const fromElev = elevation.profile.find(
      (p) => Math.abs(p.lat - from.lat) < 0.001 && Math.abs(p.lng - from.lng) < 0.001
    )?.elevation || 0
    const toElev = elevation.profile.find(
      (p) => Math.abs(p.lat - to.lat) < 0.001 && Math.abs(p.lng - to.lng) < 0.001
    )?.elevation || 0

    const dist = haversineKm(from, to) * 1000
    const elevDiff = toElev - fromElev
    const gradient = dist > 0 ? (elevDiff / dist) * 100 : 0

    segments.push({
      from,
      to,
      distance_m: dist,
      gradient,
      elevation_gain: Math.max(0, elevDiff),
      elevation_loss: Math.max(0, -elevDiff),
    })
  }

  // Attach segments to primary route
  routes[0].segments = segments

  done()
  return {
    routes,
    selectedRouteIndex: 0,
    elevation,
    weather,
    chargers,
    warnings,
    originName,
    destinationName,
  }
  } catch (err) { done(); throw err }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

function haversineKm(a: LatLng, b: LatLng): number {
  const R = 6371
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLng = ((b.lng - a.lng) * Math.PI) / 180
  const la = (a.lat * Math.PI) / 180
  const lb = (b.lat * Math.PI) / 180
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(la) * Math.cos(lb) * Math.sin(dLng / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x))
}
