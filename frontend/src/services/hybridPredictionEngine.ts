/**
 * Hybrid ML/Physics Prediction Engine (Client-Side)
 *
 * Mirrors the backend HybridPredictor logic:
 * - FastAPI ML path (ONNX/Teacher/Student)
 * - Physics validation: per-segment detailed computation
 * - Decision: ML confidence ≥ 0.75 → accept; else physics fallback
 * - If ML/physics diverge > 10% → choose physics for safety
 * - Route cost = w_energy * energy + w_degradation * degradation + w_time * time
 *
 * Dynamic features:
 * - SoC-aware: uses actual vehicle battery SoC, not hardcoded 85%
 * - SoH-aware: computes effective battery capacity from health %
 * - Driver aggression: eco/normal/sport scales energy consumption
 * - Temperature-dependent battery efficiency
 * - Intelligent multi-factor charger scoring with route-position awareness
 * - Adaptive route cost weights based on battery state
 */

import type { RouteContext, ChargerInfo, LatLng } from './routeService'
import { predictEnergy } from './predictions'

// ─── Types ───────────────────────────────────────────────────────────────────

export interface VehicleProfile {
  mass_kg: number
  drag_coefficient: number
  frontal_area_m2: number
  rolling_resistance: number
  battery_capacity_kwh: number
  motor_efficiency: number
  regen_efficiency: number
  auxiliary_power_kw: number
}

/** Extended options passed alongside vehicle for dynamic behavior */
export interface PredictionOptions {
  initialSoc: number          // 0–100, from vehicle.battery.soc_percent
  sohPercent: number          // 0–100, from vehicle.battery.soh_percent or health
  driverAggression: number    // 0–1 (0=eco, 0.5=normal, 1=aggressive)
  batteryTempC?: number       // battery temperature for efficiency scaling
}

export interface PredictionResult {
  energy_kwh: number
  arrival_soc: number
  soh_impact: number        // % degradation risk
  confidence: number        // 0–1
  confidenceLevel: 'HIGH' | 'MEDIUM' | 'LOW'
  method: 'ml' | 'physics' | 'ml_validated' | 'physics_fallback'
  duration_minutes: number
  avg_speed_kmh: number
  route_explanation: string
  charger_stop: string | null
  energy_breakdown: {
    rolling: number
    aero: number
    grade: number
    auxiliary: number
    regen_recovered: number
  }
  segment_costs: number[]   // per-segment energy for color coding
  route_cost: number        // weighted composite score
  weather_impact: string
  elevation_impact: string
}

// ─── Default Vehicle ─────────────────────────────────────────────────────────

export const DEFAULT_VEHICLE: VehicleProfile = {
  mass_kg: 1500,
  drag_coefficient: 0.28,
  frontal_area_m2: 2.2,
  rolling_resistance: 0.012,
  battery_capacity_kwh: 60,
  motor_efficiency: 0.92,
  regen_efficiency: 0.70,
  auxiliary_power_kw: 0.5,
}

export const DEFAULT_OPTIONS: PredictionOptions = {
  initialSoc: 85,
  sohPercent: 100,
  driverAggression: 0.5,
}

// ─── Constants ───────────────────────────────────────────────────────────────

const G = 9.81           // m/s²
const AIR_DENSITY = 1.225 // kg/m³
const CONFIDENCE_THRESHOLD = 0.75
const DIVERGENCE_THRESHOLD = 0.10  // 10%

// ─── Helper: Effective Battery Capacity ──────────────────────────────────────

function effectiveCapacity(nominalKwh: number, sohPercent: number, batteryTempC?: number): number {
  let cap = nominalKwh * (sohPercent / 100)
  // Cold battery penalty: Li-ion loses ~1% capacity per degree below 20°C (capped at 25%)
  if (batteryTempC !== undefined && batteryTempC < 20) {
    const coldPenalty = Math.min(0.25, (20 - batteryTempC) * 0.01)
    cap *= (1 - coldPenalty)
  }
  return Math.max(cap, 1) // floor at 1 kWh to prevent division by zero
}

// ─── Helper: Driver Aggression Factor ────────────────────────────────────────

function aggressionMultiplier(aggression: number): number {
  // 0 (eco) → 0.85, 0.5 (normal) → 1.0, 1.0 (sport) → 1.20
  return 0.85 + aggression * 0.35
}

// ─── Helper: Adaptive Cost Weights ───────────────────────────────────────────

function adaptiveCostWeights(initialSoc: number, sohPercent: number): { wEnergy: number; wDeg: number; wTime: number } {
  let wEnergy = 0.5
  let wDeg = 0.3
  let wTime = 0.2

  // Low SoC → prioritize energy efficiency over time
  if (initialSoc < 30) { wEnergy = 0.65; wTime = 0.05; wDeg = 0.30 }
  else if (initialSoc < 50) { wEnergy = 0.55; wTime = 0.15; wDeg = 0.30 }

  // Low SoH → prioritize avoiding degradation
  if (sohPercent < 80) { wDeg = 0.45; wEnergy = 0.40; wTime = 0.15 }

  return { wEnergy, wDeg, wTime }
}

// ─── Helper: Haversine ───────────────────────────────────────────────────────

function haversineKm(a: LatLng, b: LatLng): number {
  const R = 6371
  const dLat = ((b.lat - a.lat) * Math.PI) / 180
  const dLng = ((b.lng - a.lng) * Math.PI) / 180
  const la = (a.lat * Math.PI) / 180
  const lb = (b.lat * Math.PI) / 180
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(la) * Math.cos(lb) * Math.sin(dLng / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x))
}

// ─── Helper: Point-to-polyline minimum distance ─────────────────────────────

function minDistToRoute(point: LatLng, routeCoords: LatLng[]): number {
  let minDist = Infinity
  for (const c of routeCoords) {
    const d = haversineKm(point, c)
    if (d < minDist) minDist = d
  }
  return minDist // km
}

// ─── Helper: Position along route (0–1) ─────────────────────────────────────

function positionAlongRoute(point: LatLng, routeCoords: LatLng[]): number {
  let bestIdx = 0
  let bestDist = Infinity
  for (let i = 0; i < routeCoords.length; i++) {
    const d = haversineKm(point, routeCoords[i])
    if (d < bestDist) { bestDist = d; bestIdx = i }
  }
  return routeCoords.length > 1 ? bestIdx / (routeCoords.length - 1) : 0.5
}

// ─── ML Fast Path ────────────────────────────────────────────────────────────

function mlFastPath(
  ctx: RouteContext,
  vehicle: VehicleProfile,
  opts: PredictionOptions
): { energy: number; confidence: number; breakdown: PredictionResult['energy_breakdown'] } {
  const route = ctx.routes[ctx.selectedRouteIndex]
  const dist_m = route.distance_m
  const dist_km = dist_m / 1000
  const duration_s = route.duration_s
  const avgSpeed = duration_s > 0 ? dist_m / duration_s : 13.9 // ~50 km/h default

  // Rolling resistance energy
  const F_roll = vehicle.rolling_resistance * vehicle.mass_kg * G
  const E_roll = (F_roll * dist_m) / (3.6e6 * vehicle.motor_efficiency) // kWh

  // Aerodynamic drag energy
  const F_aero = 0.5 * vehicle.drag_coefficient * vehicle.frontal_area_m2 * AIR_DENSITY * avgSpeed * avgSpeed
  const E_aero = (F_aero * dist_m) / (3.6e6 * vehicle.motor_efficiency) // kWh

  // Grade energy  
  const elev = ctx.elevation
  const E_grade_positive = (vehicle.mass_kg * G * elev.gain) / (3.6e6 * vehicle.motor_efficiency)
  const E_grade_regen = (vehicle.mass_kg * G * elev.loss * vehicle.regen_efficiency) / 3.6e6
  const E_grade = E_grade_positive - E_grade_regen

  // Auxiliary power
  const E_aux = (vehicle.auxiliary_power_kw * duration_s) / 3600

  // Weather adjustments
  let weatherFactor = 1.0
  const weather = ctx.weather
  if (weather.temperature < 10) weatherFactor += 0.08
  else if (weather.temperature < 20) weatherFactor += 0.03
  if (weather.rain) weatherFactor += 0.04
  if (weather.windSpeed > 5) weatherFactor += 0.03

  // Driver aggression scaling
  const aggrFactor = aggressionMultiplier(opts.driverAggression)

  const totalEnergy = Math.max(0.1, (E_roll + E_aero + Math.max(0, E_grade) + E_aux) * weatherFactor * aggrFactor)

  // Confidence scoring
  let confidence = 0.85
  if (elev.isEstimated) confidence -= 0.15
  if (weather.isEstimated) confidence -= 0.10
  if (dist_km > 200) confidence -= 0.10
  if (dist_km < 5) confidence -= 0.05
  if (opts.sohPercent < 80) confidence -= 0.05 // degraded battery adds uncertainty
  confidence = Math.max(0.1, Math.min(1.0, confidence))

  return {
    energy: Math.round(totalEnergy * 100) / 100,
    confidence,
    breakdown: {
      rolling: Math.round(E_roll * aggrFactor * 100) / 100,
      aero: Math.round(E_aero * aggrFactor * 100) / 100,
      grade: Math.round(E_grade * 100) / 100,
      auxiliary: Math.round(E_aux * 100) / 100,
      regen_recovered: Math.round(E_grade_regen * 100) / 100,
    },
  }
}

// ─── Physics Validation ──────────────────────────────────────────────────────

function physicsValidation(
  ctx: RouteContext,
  vehicle: VehicleProfile,
  opts: PredictionOptions
): { energy: number; segmentCosts: number[] } {
  const route = ctx.routes[ctx.selectedRouteIndex]
  const segments = route.segments
  if (segments.length === 0) {
    const ml = mlFastPath(ctx, vehicle, opts)
    return { energy: ml.energy, segmentCosts: [ml.energy] }
  }

  const aggrFactor = aggressionMultiplier(opts.driverAggression)
  const segmentCosts: number[] = []
  let totalEnergy = 0

  for (const seg of segments) {
    const dist = seg.distance_m
    if (dist <= 0) { segmentCosts.push(0); continue }

    const avgSpeed = route.duration_s > 0 ? route.distance_m / route.duration_s : 13.9
    const segDuration = dist / avgSpeed

    // Forces
    const F_roll = vehicle.rolling_resistance * vehicle.mass_kg * G
    const F_aero = 0.5 * vehicle.drag_coefficient * vehicle.frontal_area_m2 * AIR_DENSITY * avgSpeed * avgSpeed
    const gradeRad = Math.atan(seg.gradient / 100)
    const F_grade = vehicle.mass_kg * G * Math.sin(gradeRad)

    const F_total = F_roll + F_aero + F_grade
    let E_seg: number

    if (F_total > 0) {
      E_seg = (F_total * dist) / (3.6e6 * vehicle.motor_efficiency)
    } else {
      E_seg = (F_total * dist * vehicle.regen_efficiency) / 3.6e6
    }

    // Auxiliary + aggression
    E_seg = E_seg * aggrFactor + (vehicle.auxiliary_power_kw * segDuration) / 3600

    segmentCosts.push(Math.round(E_seg * 1000) / 1000)
    totalEnergy += E_seg
  }

  // Weather adjustment
  let weatherFactor = 1.0
  if (ctx.weather.temperature < 10) weatherFactor += 0.08
  else if (ctx.weather.temperature < 20) weatherFactor += 0.03
  if (ctx.weather.rain) weatherFactor += 0.04
  if (ctx.weather.windSpeed > 5) weatherFactor += 0.03

  totalEnergy *= weatherFactor

  return {
    energy: Math.round(Math.max(0.1, totalEnergy) * 100) / 100,
    segmentCosts,
  }
}

// ─── Route Cost Scoring (now adaptive) ───────────────────────────────────────

function computeRouteCost(
  energy: number,
  degradation: number,
  durationMin: number,
  opts: PredictionOptions
): number {
  const { wEnergy, wDeg, wTime } = adaptiveCostWeights(opts.initialSoc, opts.sohPercent)
  const normEnergy = Math.min(energy / 50, 1)
  const normDeg = Math.min(degradation / 5, 1)
  const normTime = Math.min(durationMin / 180, 1)

  return wEnergy * normEnergy + wDeg * normDeg + wTime * normTime
}

// ─── Battery Degradation Risk ────────────────────────────────────────────────

function estimateDegradation(
  energy: number,
  effCapacity: number,
  temperature: number,
  initialSoc: number,
  sohPercent: number
): number {
  const dod = (energy / effCapacity) * 100
  let risk = dod * 0.005

  // Temperature stress
  if (temperature > 35) risk *= 1.5
  else if (temperature < 5) risk *= 1.3

  // Deep discharge stress
  const finalSoc = initialSoc - (energy / effCapacity) * 100
  if (finalSoc < 10) risk *= 2.0
  else if (finalSoc < 20) risk *= 1.5

  // Already-degraded batteries are more vulnerable
  if (sohPercent < 80) risk *= 1.4
  else if (sohPercent < 90) risk *= 1.15

  return Math.round(risk * 100) / 100
}

// ─── Intelligent Charger Scoring ─────────────────────────────────────────────

function scoreCharger(
  charger: ChargerInfo,
  routeCoords: LatLng[],
  optimalStopPosition: number, // 0–1, where along route a stop is most needed
  maxPower: number
): number {
  const detourKm = minDistToRoute({ lat: charger.lat, lng: charger.lng }, routeCoords)
  const chargerPosition = positionAlongRoute({ lat: charger.lat, lng: charger.lng }, routeCoords)

  // Scoring factors
  const detourScore = Math.max(0, 1 - detourKm / 10) // 10 km = max acceptable detour
  const powerScore = maxPower > 0 ? charger.powerKw / maxPower : 0.5
  const availScore = charger.status === 'available' ? 1.0 : charger.status === 'unknown' ? 0.6 : 0.1
  const positionScore = 1 - Math.abs(optimalStopPosition - chargerPosition)

  return 0.30 * detourScore + 0.25 * powerScore + 0.20 * availScore + 0.25 * positionScore
}

function recommendChargerStop(
  arrivalSoc: number,
  _energy: number,
  effCapacity: number,
  initialSoc: number,
  chargers: ChargerInfo[],
  routeCoords: LatLng[],
  segmentCosts: number[]
): string | null {
  // Check if mid-route SoC drops dangerously low
  let currentSoc = initialSoc
  let criticalPosition = -1
  const totalSegEnergy = segmentCosts.reduce((a, b) => a + Math.abs(b), 0)

  for (let i = 0; i < segmentCosts.length; i++) {
    const segEnergyPct = (Math.abs(segmentCosts[i]) / effCapacity) * 100
    currentSoc -= segEnergyPct
    if (currentSoc < 10 && criticalPosition < 0) {
      criticalPosition = segmentCosts.length > 1 ? i / (segmentCosts.length - 1) : 0.5
    }
  }

  // Determine if a charger stop is needed
  const needsStop = arrivalSoc < 20 || criticalPosition >= 0
  if (!needsStop) return null

  if (chargers.length === 0) {
    return `⚠ Warning: arrival SoC is ${arrivalSoc.toFixed(0)}% — no charger data available`
  }

  // Find optimal stop position based on initial SoC and critical points
  let optimalPos: number
  if (initialSoc <= 40) {
    // If starting with low SoC, prioritize chargers nearer to the start.
    // e.g. at 10% SoC -> pos 0.0 (start), at 40% SoC -> pos 0.3
    optimalPos = Math.max(0, (initialSoc - 10) / 100)
  } else if (criticalPosition >= 0) {
    optimalPos = Math.max(0, criticalPosition - 0.1) // stop slightly before critical point
  } else {
    optimalPos = 0.6 // default: 60% along route
  }

  const maxPower = Math.max(...chargers.map(c => c.powerKw), 1)
  const usable = chargers.filter(c => {
    if (c.status !== 'available' && c.status !== 'unknown') return false
    
    // If we have a critical battery failure point, filter out chargers beyond that point!
    // We add a tiny buffer (0.05 = 5% of route) to account for estimation variance.
    if (criticalPosition >= 0) {
      const pos = positionAlongRoute({ lat: c.lat, lng: c.lng }, routeCoords)
      if (pos > criticalPosition + 0.05) {
        return false // Charger is physically out of range
      }
    }
    return true
  })

  if (usable.length === 0) {
    return `⚠ Warning: arrival SoC is ${arrivalSoc.toFixed(0)}% but no reachable/available chargers found along route`
  }

  // Score and rank chargers
  // If starting SoC is very low, we heavily penalize chargers that are further away.
  const scored = usable
    .map(c => {
      let score = scoreCharger(c, routeCoords, optimalPos, maxPower)
      // Extreme penalty for distance if we are starting with very low SoC
      if (initialSoc <= 25) {
         const pos = positionAlongRoute({ lat: c.lat, lng: c.lng }, routeCoords)
         score -= (pos * 2.0) // Penalize chargers further down the route heavily
      }
      return { charger: c, score }
    })
    .sort((a, b) => b.score - a.score)

  const best = scored[0].charger
  const detour = minDistToRoute({ lat: best.lat, lng: best.lng }, routeCoords)
  let msg = `Recommended stop: ${best.name} (${best.powerKw} kW, ${best.operator})`
  if (detour > 0.5) msg += ` — ${detour.toFixed(1)} km detour`
  if (criticalPosition >= 0) {
    msg += ` — battery critically low mid-route (SoC drops below 10% at ${(criticalPosition * 100).toFixed(0)}% of trip)`
  } else {
    msg += ` — arrival SoC is ${arrivalSoc.toFixed(0)}%`
  }
  return msg
}

// ─── Main Prediction Function ────────────────────────────────────────────────

export async function runHybridPrediction(
  ctx: RouteContext,
  vehicle: VehicleProfile = DEFAULT_VEHICLE,
  initialSoc: number = 85,
  onStageChange?: (stage: string) => void,
  modelType: 'onnx' | 'student' | 'teacher' = 'onnx',
  options?: Partial<PredictionOptions>
): Promise<PredictionResult> {
  // Merge options — backward compatible (initialSoc param still works)
  const opts: PredictionOptions = {
    initialSoc: options?.initialSoc ?? initialSoc,
    sohPercent: options?.sohPercent ?? 100,
    driverAggression: options?.driverAggression ?? 0.5,
    batteryTempC: options?.batteryTempC,
  }

  const route = ctx.routes[ctx.selectedRouteIndex]
  const dist_km = route.distance_m / 1000
  const duration_min = route.duration_s / 60
  const avg_speed_kmh = duration_min > 0 ? (dist_km / duration_min) * 60 : 50

  // Effective battery capacity accounting for SoH and temperature
  const effCap = effectiveCapacity(vehicle.battery_capacity_kwh, opts.sohPercent, opts.batteryTempC)

  // ── Step 1: ML Fast Path
  onStageChange?.('ml_prediction')
  
  let mlEnergy = 0;
  let mlConfidence = 0;
  let usedBackend = false;
  
  try {
    const apiResult = await predictEnergy({
      distance_km: dist_km,
      speed_kmh: avg_speed_kmh,
      temperature_c: ctx.weather.temperature,
      initial_soc: opts.initialSoc,
      initial_soh: opts.sohPercent,
      mass_kg: vehicle.mass_kg,
      drag_coeff: vehicle.drag_coefficient,
      model_type: modelType
    });
    // Scale backend result by aggression factor
    mlEnergy = apiResult.energy_kwh * aggressionMultiplier(opts.driverAggression);
    mlConfidence = apiResult.confidence;
    usedBackend = true;
  } catch(e) {
    console.warn("Backend ML failed, using local fallback math", e);
    const localMl = mlFastPath(ctx, vehicle, opts);
    mlEnergy = localMl.energy;
    mlConfidence = localMl.confidence;
  }

  // Get breakdown for UI
  const localBreakdown = mlFastPath(ctx, vehicle, opts).breakdown;

  let energy: number
  let confidence: number
  let method: PredictionResult['method']
  let segmentCosts: number[] = []
  let explanation: string

  if (mlConfidence >= CONFIDENCE_THRESHOLD) {
    onStageChange?.('physics_validation')
    const physics = physicsValidation(ctx, vehicle, opts)
    segmentCosts = physics.segmentCosts

    const relError = Math.abs(mlEnergy - physics.energy) / Math.max(physics.energy, 0.01)

    if (relError <= DIVERGENCE_THRESHOLD) {
      energy = mlEnergy
      confidence = mlConfidence
      method = 'ml_validated'
      const prefix = usedBackend ? 'ONNX Backend ML' : 'Local ML'
      explanation = `${prefix} prediction (${mlEnergy.toFixed(1)} kWh) validated by physics engine (${physics.energy.toFixed(1)} kWh). ${(relError * 100).toFixed(1)}% divergence — within tolerance.`
    } else {
      energy = physics.energy
      confidence = Math.max(0.6, mlConfidence - 0.15)
      method = 'physics_fallback'
      const prefix = usedBackend ? 'ONNX Backend ML' : 'Local ML'
      explanation = `${prefix} predicted ${mlEnergy.toFixed(1)} kWh but physics computed ${physics.energy.toFixed(1)} kWh (${(relError * 100).toFixed(1)}% divergence). Using physics result for safety.`
    }
  } else {
    onStageChange?.('physics_validation')
    const physics = physicsValidation(ctx, vehicle, opts)
    energy = physics.energy
    segmentCosts = physics.segmentCosts
    confidence = 0.70
    method = 'physics_fallback'
    explanation = `ML confidence too low (${(mlConfidence * 100).toFixed(0)}% < 75% threshold). Route evaluated with physics engine for accuracy.`
  }

  // ── SoC and Degradation (using effective capacity)
  const soc_used_pct = (energy / effCap) * 100
  const arrival_soc = Math.max(0, Math.min(100, opts.initialSoc - soc_used_pct))
  const soh_impact = estimateDegradation(energy, effCap, ctx.weather.temperature, opts.initialSoc, opts.sohPercent)

  // ── Intelligent charger recommendation
  const routeCoords = route.geometry
  const charger_stop = recommendChargerStop(
    arrival_soc, energy, effCap, opts.initialSoc, ctx.chargers, routeCoords, segmentCosts
  )

  // ── Weather & elevation impact text
  let weatherImpact = `${ctx.weather.temperature}°C, ${ctx.weather.description}`
  if (ctx.weather.rain) weatherImpact += ' — wet road (+4% energy)'
  if (ctx.weather.windSpeed > 5) weatherImpact += ` — wind ${ctx.weather.windSpeed.toFixed(0)} m/s (+3%)`
  if (ctx.weather.isEstimated) weatherImpact += ' (estimated)'

  let elevImpact = `↑${ctx.elevation.gain}m ↓${ctx.elevation.loss}m, avg gradient ${ctx.elevation.avgGradient}%`
  if (ctx.elevation.isEstimated) elevImpact += ' (estimated flat)'

  // ── Route cost score (adaptive weights)
  const routeCost = computeRouteCost(energy, soh_impact, duration_min, opts)

  // ── Confidence level
  const confidenceLevel: PredictionResult['confidenceLevel'] =
    confidence >= 0.75 ? 'HIGH' : confidence >= 0.5 ? 'MEDIUM' : 'LOW'

  // ── Enhance explanation with vehicle-specific context
  explanation += ` Route: ${dist_km.toFixed(1)} km, ${duration_min.toFixed(0)} min.`
  if (ctx.elevation.gain > 100) explanation += ` Significant climb: +${ctx.elevation.gain}m.`
  if (ctx.weather.rain) explanation += ' Rain increases rolling resistance.'
  if (opts.sohPercent < 85) explanation += ` Battery health ${opts.sohPercent}% — effective capacity ${effCap.toFixed(1)} kWh.`
  if (opts.driverAggression > 0.7) explanation += ' Aggressive driving mode increases consumption ~20%.'
  else if (opts.driverAggression < 0.3) explanation += ' Eco driving mode saves ~15% energy.'

  return {
    energy_kwh: energy,
    arrival_soc,
    soh_impact,
    confidence,
    confidenceLevel,
    method,
    duration_minutes: Math.round(duration_min),
    avg_speed_kmh: Math.round(avg_speed_kmh),
    route_explanation: explanation,
    charger_stop,
    energy_breakdown: localBreakdown,
    segment_costs: segmentCosts,
    route_cost: Math.round(routeCost * 1000) / 1000,
    weather_impact: weatherImpact,
    elevation_impact: elevImpact,
  }
}

// ─── Select Best Route ───────────────────────────────────────────────────────

export async function selectBestRoute(
  ctx: RouteContext, 
  vehicle: VehicleProfile = DEFAULT_VEHICLE,
  modelType: 'onnx' | 'student' | 'teacher' = 'onnx',
  options?: Partial<PredictionOptions>
): Promise<{
  bestIndex: number
  predictions: PredictionResult[]
}> {
  const predictions: PredictionResult[] = []

  for (let i = 0; i < ctx.routes.length; i++) {
    const ctxCopy = { ...ctx, selectedRouteIndex: i }
    predictions.push(await runHybridPrediction(ctxCopy, vehicle, options?.initialSoc ?? 85, undefined, modelType, options))
  }

  // Select lowest cost route
  let bestIndex = 0
  let bestCost = Infinity
  predictions.forEach((p, i) => {
    if (p.route_cost < bestCost) {
      bestCost = p.route_cost
      bestIndex = i
    }
  })

  return { bestIndex, predictions }
}
