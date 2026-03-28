/**
 * Hybrid ML/Physics Prediction Engine (Client-Side)
 *
 * Mirrors the backend HybridPredictor logic:
 * - FastAPI ML path (ONNX/Teacher/Student)
 * - Physics validation: per-segment detailed computation
 * - Decision: ML confidence ≥ 0.75 → accept; else physics fallback
 * - If ML/physics diverge > 10% → choose physics for safety
 * - Route cost = w_energy * energy + w_degradation * degradation + w_time * time
 */

import type { RouteContext } from './routeService'
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

// ─── Constants ───────────────────────────────────────────────────────────────

const G = 9.81           // m/s²
const AIR_DENSITY = 1.225 // kg/m³
const CONFIDENCE_THRESHOLD = 0.75
const DIVERGENCE_THRESHOLD = 0.10  // 10%

// Cost weights (paper values)
const W_ENERGY = 0.5
const W_DEGRADATION = 0.3
const W_TIME = 0.2

// ─── ML Fast Path ────────────────────────────────────────────────────────────

function mlFastPath(
  ctx: RouteContext,
  vehicle: VehicleProfile,
  _initialSoc: number
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

  const E_grade_positive = (vehicle.mass_kg * G * elev.gain) / (3.6e6 * vehicle.motor_efficiency) // kWh for climbing
  const E_grade_regen = (vehicle.mass_kg * G * elev.loss * vehicle.regen_efficiency) / 3.6e6 // kWh recovered
  const E_grade = E_grade_positive - E_grade_regen

  // Auxiliary power
  const E_aux = (vehicle.auxiliary_power_kw * duration_s) / 3600 // kWh

  // Weather adjustments
  let weatherFactor = 1.0
  const weather = ctx.weather
  if (weather.temperature < 10) weatherFactor += 0.08
  else if (weather.temperature < 20) weatherFactor += 0.03
  if (weather.rain) weatherFactor += 0.04
  // Headwind approximation (assume travel bearing ~ 0°, very rough)
  if (weather.windSpeed > 5) weatherFactor += 0.03

  const totalEnergy = Math.max(0.1, (E_roll + E_aero + Math.max(0, E_grade) + E_aux) * weatherFactor)

  // Confidence scoring
  let confidence = 0.85  // base confidence for physics-derived ML
  if (elev.isEstimated) confidence -= 0.15
  if (weather.isEstimated) confidence -= 0.10
  if (dist_km > 200) confidence -= 0.10  // long routes have more uncertainty
  if (dist_km < 5) confidence -= 0.05    // very short routes are noisy
  confidence = Math.max(0.1, Math.min(1.0, confidence))

  return {
    energy: Math.round(totalEnergy * 100) / 100,
    confidence,
    breakdown: {
      rolling: Math.round(E_roll * 100) / 100,
      aero: Math.round(E_aero * 100) / 100,
      grade: Math.round(E_grade * 100) / 100,
      auxiliary: Math.round(E_aux * 100) / 100,
      regen_recovered: Math.round(E_grade_regen * 100) / 100,
    },
  }
}

// ─── Physics Validation ──────────────────────────────────────────────────────

function physicsValidation(
  ctx: RouteContext,
  vehicle: VehicleProfile
): { energy: number; segmentCosts: number[] } {
  const route = ctx.routes[ctx.selectedRouteIndex]
  const segments = route.segments
  if (segments.length === 0) {
    // No segments, fall back to simple calculation
    const ml = mlFastPath(ctx, vehicle, 90)
    return { energy: ml.energy, segmentCosts: [ml.energy] }
  }

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
      // Regeneration
      E_seg = (F_total * dist * vehicle.regen_efficiency) / 3.6e6
    }

    // Auxiliary
    E_seg += (vehicle.auxiliary_power_kw * segDuration) / 3600

    segmentCosts.push(Math.round(E_seg * 1000) / 1000)
    totalEnergy += E_seg
  }

  // Weather adjustment (same as ML path)
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

// ─── Route Cost Scoring ──────────────────────────────────────────────────────

function computeRouteCost(energy: number, degradation: number, durationMin: number): number {
  // Normalize to 0–1 ranges (rough)
  const normEnergy = Math.min(energy / 50, 1)        // 50 kWh = max
  const normDeg = Math.min(degradation / 5, 1)       // 5% = max degradation
  const normTime = Math.min(durationMin / 180, 1)    // 3 hours = max

  return W_ENERGY * normEnergy + W_DEGRADATION * normDeg + W_TIME * normTime
}

// ─── Battery Degradation Risk ────────────────────────────────────────────────

function estimateDegradation(
  energy: number,
  capacity: number,
  temperature: number,
  initialSoc: number
): number {
  const dod = (energy / capacity) * 100
  let risk = dod * 0.005 // base: 0.5% per 100% DoD

  // Temperature stress
  if (temperature > 35) risk *= 1.5
  else if (temperature < 5) risk *= 1.3

  // Deep discharge stress
  const finalSoc = initialSoc - (energy / capacity) * 100
  if (finalSoc < 10) risk *= 2.0
  else if (finalSoc < 20) risk *= 1.5

  return Math.round(risk * 100) / 100
}

// ─── Main Prediction Function ────────────────────────────────────────────────

export async function runHybridPrediction(
  ctx: RouteContext,
  vehicle: VehicleProfile = DEFAULT_VEHICLE,
  initialSoc: number = 85,
  onStageChange?: (stage: string) => void,
  modelType: 'onnx' | 'student' | 'teacher' = 'onnx'
): Promise<PredictionResult> {
  const route = ctx.routes[ctx.selectedRouteIndex]
  const dist_km = route.distance_m / 1000
  const duration_min = route.duration_s / 60
  const avg_speed_kmh = duration_min > 0 ? (dist_km / duration_min) * 60 : 50

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
      initial_soc: initialSoc,
      initial_soh: 100, // Assuming 100 SoH for planning default
      mass_kg: vehicle.mass_kg,
      drag_coeff: vehicle.drag_coefficient,
      model_type: modelType
    });
    mlEnergy = apiResult.energy_kwh;
    mlConfidence = apiResult.confidence;
    usedBackend = true;
  } catch(e) {
    console.warn("Backend ML failed, using local fallback math", e);
    const localMl = mlFastPath(ctx, vehicle, initialSoc);
    mlEnergy = localMl.energy;
    mlConfidence = localMl.confidence;
  }

  // Get breakdown for UI UI
  const localBreakdown = mlFastPath(ctx, vehicle, initialSoc).breakdown;

  let energy: number
  let confidence: number
  let method: PredictionResult['method']
  let segmentCosts: number[] = []
  let explanation: string

  if (mlConfidence >= CONFIDENCE_THRESHOLD) {
    // ML is confident — accept (but still run physics for validation)
    onStageChange?.('physics_validation')
    const physics = physicsValidation(ctx, vehicle)
    segmentCosts = physics.segmentCosts

    const relError = Math.abs(mlEnergy - physics.energy) / Math.max(physics.energy, 0.01)

    if (relError <= DIVERGENCE_THRESHOLD) {
      // ML and physics agree
      energy = mlEnergy
      confidence = mlConfidence
      method = 'ml_validated'
      const prefix = usedBackend ? 'ONNX Backend ML' : 'Local ML'
      explanation = `${prefix} prediction (${mlEnergy.toFixed(1)} kWh) validated by physics engine (${physics.energy.toFixed(1)} kWh). ${(relError * 100).toFixed(1)}% divergence — within tolerance.`
    } else {
      // Divergence > 10% — use physics for safety
      energy = physics.energy
      confidence = Math.max(0.6, mlConfidence - 0.15)
      method = 'physics_fallback'
      const prefix = usedBackend ? 'ONNX Backend ML' : 'Local ML'
      explanation = `${prefix} predicted ${mlEnergy.toFixed(1)} kWh but physics computed ${physics.energy.toFixed(1)} kWh (${(relError * 100).toFixed(1)}% divergence). Using physics result for safety.`
    }
  } else {
    // Low confidence — skip ML, use physics directly
    onStageChange?.('physics_validation')
    const physics = physicsValidation(ctx, vehicle)
    energy = physics.energy
    segmentCosts = physics.segmentCosts
    confidence = 0.70 // physics-only confidence
    method = 'physics_fallback'
    explanation = `ML confidence too low (${(mlConfidence * 100).toFixed(0)}% < 75% threshold). Route evaluated with physics engine for accuracy.`
  }

  // ── SoC and Degradation
  const soc_used_pct = (energy / vehicle.battery_capacity_kwh) * 100
  const arrival_soc = Math.max(0, Math.min(100, initialSoc - soc_used_pct))
  const soh_impact = estimateDegradation(energy, vehicle.battery_capacity_kwh, ctx.weather.temperature, initialSoc)

  // ── Charger recommendation
  let charger_stop: string | null = null
  if (arrival_soc < 15 && ctx.chargers.length > 0) {
    const available = ctx.chargers.filter((c) => c.status === 'available' || c.status === 'unknown')
    if (available.length > 0) {
      const best = available.reduce((a, b) => (a.powerKw > b.powerKw ? a : b))
      charger_stop = `Recommended stop: ${best.name} (${best.powerKw} kW) — arrival SoC is low (${arrival_soc.toFixed(0)}%)`
    } else {
      charger_stop = `Warning: arrival SoC is ${arrival_soc.toFixed(0)}% but no available chargers found along route`
    }
  } else if (arrival_soc < 15) {
    charger_stop = `Warning: arrival SoC is ${arrival_soc.toFixed(0)}% — no charger data available`
  }

  // ── Weather & elevation impact text
  let weatherImpact = `${ctx.weather.temperature}°C, ${ctx.weather.description}`
  if (ctx.weather.rain) weatherImpact += ' — wet road (+4% energy)'
  if (ctx.weather.windSpeed > 5) weatherImpact += ` — wind ${ctx.weather.windSpeed.toFixed(0)} m/s (+3%)`
  if (ctx.weather.isEstimated) weatherImpact += ' (estimated)'

  let elevImpact = `↑${ctx.elevation.gain}m ↓${ctx.elevation.loss}m, avg gradient ${ctx.elevation.avgGradient}%`
  if (ctx.elevation.isEstimated) elevImpact += ' (estimated flat)'

  // ── Route cost score
  const routeCost = computeRouteCost(energy, soh_impact, duration_min)

  // ── Confidence level
  const confidenceLevel: PredictionResult['confidenceLevel'] =
    confidence >= 0.75 ? 'HIGH' : confidence >= 0.5 ? 'MEDIUM' : 'LOW'

  // ── Enhance explanation
  explanation += ` Route: ${dist_km.toFixed(1)} km, ${duration_min.toFixed(0)} min.`
  if (ctx.elevation.gain > 100) explanation += ` Significant climb: +${ctx.elevation.gain}m.`
  if (ctx.weather.rain) explanation += ' Rain increases rolling resistance.'

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
  modelType: 'onnx' | 'student' | 'teacher' = 'onnx'
): Promise<{
  bestIndex: number
  predictions: PredictionResult[]
}> {
  const predictions: PredictionResult[] = []

  for (let i = 0; i < ctx.routes.length; i++) {
    const ctxCopy = { ...ctx, selectedRouteIndex: i }
    predictions.push(await runHybridPrediction(ctxCopy, vehicle, 85, undefined, modelType))
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
