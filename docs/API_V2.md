# API V2 Documentation

## Overview

The V2 API adds ML-enhanced prediction endpoints for faster energy consumption estimates with confidence scoring.

## Base URL

```
http://localhost:8000/simulation
```

## Endpoints

### POST /v2/predict-quick

Fast ML-only prediction (~100ms).

**Use when**: Speed is priority, accuracy within 10% acceptable.

**Request Body:**

```json
{
  "vehicle": {
    "mass_kg": 1800,
    "drag_coefficient": 0.28,
    "frontal_area_m2": 2.3,
    "rolling_resistance_coefficient": 0.01,
    "battery_capacity_kwh": 75,
    "battery_voltage_nominal": 400,
    "battery_internal_resistance_ohm": 0.1,
    "motor_efficiency": 0.92,
    "regen_efficiency": 0.7,
    "auxiliary_power_kw": 0.5
  },
  "route": {
    "distance_km": 50,
    "elevation_profile": [],
    "initial_elevation_m": 0,
    "target_velocity_mps": 27.78
  },
  "environment": {
    "temperature_c": 20,
    "wind_speed_mps": 2,
    "wind_direction_deg": 0,
    "road_grade_percent": 0
  },
  "initial_soc": 90,
  "driver_aggression": 0.5
}
```

**Response:**

```json
{
  "energy_kwh": 12.5,
  "duration_minutes": 35.2,
  "final_soc": 73.4,
  "avg_speed_kmh": 85.2,
  "confidence": {
    "score": 0.87,
    "interpretation": "HIGH",
    "model_uncertainty": 0.92,
    "data_quality": 0.95,
    "in_distribution": true
  },
  "method_used": "ml",
  "execution_time_ms": 95,
  "recommendations": []
}
```

**Error Response (503):**

```json
{
  "detail": "ML model not available. Use /v2/predict-accurate instead."
}
```

---

### POST /v2/predict-accurate

Full physics simulation (~2000ms).

**Use when**: Maximum accuracy required.

**Request Body:** Same as `/v2/predict-quick`

**Response:**

```json
{
  "energy_kwh": 12.3,
  "duration_minutes": 35.5,
  "final_soc": 73.6,
  "avg_speed_kmh": 84.5,
  "confidence": {
    "score": 0.95,
    "interpretation": "HIGH",
    "in_distribution": true
  },
  "method_used": "physics",
  "execution_time_ms": 2150,
  "recommendations": ["Physics simulation provides highest accuracy"]
}
```

---

### POST /v2/predict-hybrid

Intelligent routing between ML and physics (100-2000ms).

**Use when**: Best balance of speed and accuracy.

**Request Body:** Same as `/v2/predict-quick`

**Response:**

```json
{
  "energy_kwh": 12.4,
  "duration_minutes": 35.3,
  "final_soc": 73.5,
  "avg_speed_kmh": 84.8,
  "confidence": {
    "score": 0.87,
    "interpretation": "HIGH",
    "model_uncertainty": 0.92,
    "data_quality": 0.95,
    "in_distribution": true
  },
  "method_used": "ml_validated",
  "execution_time_ms": 450,
  "recommendations": ["✓ All confidence checks passed"]
}
```

**Routing Logic:**

- If confidence ≥ 75% and in-distribution → ML prediction
- Otherwise → Physics fallback
- Periodic cross-validation for quality assurance

---

### GET /v2/validation-stats

Get ML model validation statistics.

**Response:**

```json
{
  "ml_model_loaded": true,
  "validation_stats": {
    "n_validations": 150,
    "mean_relative_error": 0.045,
    "max_relative_error": 0.12,
    "within_5_percent": 78.5,
    "within_10_percent": 96.2
  },
  "confidence_threshold": 0.75,
  "prediction_count": 1250
}
```

---

## Request Schema

### VehicleParameters

| Field                           | Type  | Required | Default | Description             |
| ------------------------------- | ----- | -------- | ------- | ----------------------- |
| mass_kg                         | float | Yes      | -       | Vehicle mass (kg)       |
| drag_coefficient                | float | Yes      | -       | Aerodynamic drag (Cd)   |
| frontal_area_m2                 | float | Yes      | -       | Frontal area (m²)       |
| rolling_resistance_coefficient  | float | Yes      | -       | Rolling resistance (Cr) |
| battery_capacity_kwh            | float | Yes      | -       | Battery capacity (kWh)  |
| battery_voltage_nominal         | float | Yes      | -       | Nominal voltage (V)     |
| battery_internal_resistance_ohm | float | Yes      | -       | Internal resistance (Ω) |
| motor_efficiency                | float | No       | 0.92    | Motor efficiency (0-1)  |
| regen_efficiency                | float | No       | 0.70    | Regen efficiency (0-1)  |
| auxiliary_power_kw              | float | No       | 0.5     | Auxiliary load (kW)     |

### RouteParameters

| Field               | Type  | Required | Default | Description                      |
| ------------------- | ----- | -------- | ------- | -------------------------------- |
| distance_km         | float | Yes      | -       | Total distance (km)              |
| elevation_profile   | array | No       | []      | [(distance_m, elevation_m), ...] |
| initial_elevation_m | float | No       | 0       | Starting elevation (m)           |
| target_velocity_mps | float | Yes      | -       | Target speed (m/s)               |

### EnvironmentParameters

| Field              | Type  | Required | Default | Description              |
| ------------------ | ----- | -------- | ------- | ------------------------ |
| temperature_c      | float | No       | 25      | Ambient temperature (°C) |
| wind_speed_mps     | float | No       | 0       | Wind speed (m/s)         |
| wind_direction_deg | float | No       | 0       | Wind direction (degrees) |
| road_grade_percent | float | No       | 0       | Constant road grade (%)  |

### Prediction Parameters

| Field             | Type  | Required | Default | Description             |
| ----------------- | ----- | -------- | ------- | ----------------------- |
| initial_soc       | float | No       | 90      | Initial battery SOC (%) |
| driver_aggression | float | No       | 0.5     | Driver aggression (0-1) |

---

## Response Schema

### PredictionResponse

| Field             | Type           | Description                                         |
| ----------------- | -------------- | --------------------------------------------------- |
| energy_kwh        | float          | Predicted energy consumption (kWh)                  |
| duration_minutes  | float          | Estimated trip duration (min)                       |
| final_soc         | float          | Predicted final SOC (%)                             |
| avg_speed_kmh     | float          | Average speed (km/h)                                |
| confidence        | ConfidenceInfo | Confidence details                                  |
| method_used       | string         | "ml", "physics", "ml_validated", "physics_fallback" |
| execution_time_ms | float          | Execution time (ms)                                 |
| recommendations   | array          | List of recommendations                             |

### ConfidenceInfo

| Field             | Type   | Description                               |
| ----------------- | ------ | ----------------------------------------- |
| score             | float  | Confidence score (0-1)                    |
| interpretation    | string | "HIGH", "MEDIUM", "LOW", "VERY_LOW"       |
| model_uncertainty | float  | Model uncertainty component               |
| data_quality      | float  | Data quality component                    |
| in_distribution   | bool   | Whether input is in training distribution |

---

## Error Codes

| Code | Description                               |
| ---- | ----------------------------------------- |
| 400  | Bad Request - Invalid parameters          |
| 503  | Service Unavailable - ML model not loaded |

---

## Example Usage

### cURL

```bash
# Quick prediction
curl -X POST http://localhost:8000/simulation/v2/predict-quick \
  -H "Content-Type: application/json" \
  -d @request.json

# Accurate prediction
curl -X POST http://localhost:8000/simulation/v2/predict-accurate \
  -H "Content-Type: application/json" \
  -d @request.json

# Hybrid prediction
curl -X POST http://localhost:8000/simulation/v2/predict-hybrid \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/simulation/v2/predict-hybrid",
    json={
        "vehicle": {
            "mass_kg": 1800,
            "drag_coefficient": 0.28,
            "frontal_area_m2": 2.3,
            "rolling_resistance_coefficient": 0.01,
            "battery_capacity_kwh": 75,
            "battery_voltage_nominal": 400,
            "battery_internal_resistance_ohm": 0.1
        },
        "route": {
            "distance_km": 50,
            "target_velocity_mps": 27.78
        },
        "environment": {
            "temperature_c": 20
        }
    }
)

result = response.json()
print(f"Energy: {result['energy_kwh']:.1f} kWh")
print(f"Confidence: {result['confidence']['interpretation']}")
```

### JavaScript

```javascript
const response = await fetch("/simulation/v2/predict-hybrid", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    vehicle: {
      mass_kg: 1800,
      drag_coefficient: 0.28,
      frontal_area_m2: 2.3,
      rolling_resistance_coefficient: 0.01,
      battery_capacity_kwh: 75,
      battery_voltage_nominal: 400,
      battery_internal_resistance_ohm: 0.1,
    },
    route: { distance_km: 50, target_velocity_mps: 27.78 },
    environment: { temperature_c: 20 },
  }),
});

const result = await response.json();
console.log(`Energy: ${result.energy_kwh.toFixed(1)} kWh`);
```
