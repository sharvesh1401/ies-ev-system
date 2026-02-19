# Phase 2: ML & Advanced Simulation Guide

This guide covers the machine learning features added in Phase 2 of the IES-EV system.

## Overview

Phase 2 adds:

- **ML-based predictions**: Fast energy consumption estimates using neural networks
- **Hybrid prediction system**: Intelligent routing between ML and physics
- **Confidence scoring**: Multi-factor reliability assessment
- **Driving profiles**: Realistic city, highway, and mixed driving patterns
- **Self-verification**: Automated validation of ML model quality

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pandas pyarrow scikit-learn
```

### 2. Train the ML Model

```bash
cd backend
python scripts/train_ml_model.py --num-scenarios 100000 --epochs 100
```

This will:

- Generate 100,000 training scenarios using the physics engine
- Train a neural network with uncertainty quantification
- Save the model to `models/energy_predictor.pth`

### 3. Validate the Model

```bash
python scripts/validate_ml_models.py --model models/energy_predictor.pth
```

Expected output:

```
✓ Physics Consistency: PASS (98.2% within 10%)
✓ Confidence Calibration: PASS
✓ No Hallucinations: PASS
✓ Uncertainty Correlation: PASS
```

### 4. Start the API Server

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### V2 Endpoints (ML-Enhanced)

| Endpoint                               | Description         | Speed      |
| -------------------------------------- | ------------------- | ---------- |
| `POST /simulation/v2/predict-quick`    | ML-only prediction  | ~100ms     |
| `POST /simulation/v2/predict-accurate` | Physics simulation  | ~2000ms    |
| `POST /simulation/v2/predict-hybrid`   | Intelligent routing | 100-2000ms |

### Example Request

```bash
curl -X POST http://localhost:8000/simulation/v2/predict-hybrid \
  -H "Content-Type: application/json" \
  -d '{
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
      "elevation_profile": [],
      "target_velocity_mps": 27.78
    },
    "environment": {
      "temperature_c": 20,
      "wind_speed_mps": 2
    },
    "initial_soc": 90,
    "driver_aggression": 0.5
  }'
```

### Example Response

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
  "method_used": "ml_validated",
  "execution_time_ms": 150,
  "recommendations": ["✓ All confidence checks passed"]
}
```

## Driving Profiles

### Creating a Profile

```python
from app.simulation.driving_profiles import (
    CityDrivingProfile,
    HighwayDrivingProfile,
    MixedDrivingProfile,
    DriverStyle
)

# City driving with eco driver
city_profile = CityDrivingProfile(seed=42)
speed_points = city_profile.generate_speed_profile(
    distance_m=10000,  # 10 km
    driver_style=DriverStyle.ECO
)

# Highway driving with aggressive driver
highway_profile = HighwayDrivingProfile(seed=42)
speed_points = highway_profile.generate_speed_profile(
    distance_m=100000,  # 100 km
    driver_style=DriverStyle.AGGRESSIVE,
    base_speed_mps=33.3  # 120 km/h
)
```

### Profile Characteristics

| Profile | Speed Range | Stops           | Use Case               |
| ------- | ----------- | --------------- | ---------------------- |
| City    | 0-60 km/h   | 10-20 per 30min | Urban commuting        |
| Highway | 90-130 km/h | Minimal         | Long-distance travel   |
| Mixed   | 0-130 km/h  | Variable        | Suburban/varied routes |

### Driver Styles

| Style      | Acceleration | Speed Offset | Braking       |
| ---------- | ------------ | ------------ | ------------- |
| Aggressive | 3.0 m/s²     | +10%         | Hard, late    |
| Moderate   | 2.0 m/s²     | Normal       | Normal        |
| Eco        | 1.2 m/s²     | -10%         | Gentle, early |

### Running Simulation with Profile

```python
from app.simulation.integrated_engine import IntegratedSimulationEngine

engine = IntegratedSimulationEngine()
result = engine.simulate_with_profile(
    vehicle_params=vehicle,
    route_params=route,
    environment_params=environment,
    speed_profile=speed_points
)
```

## ML Model Architecture

```
Input (17 features)
    │
    ▼
Linear(128) → ReLU → BatchNorm → Dropout(0.2)
    │
    ▼
Linear(64) → ReLU → BatchNorm → Dropout(0.2)
    │
    ▼
Linear(32) → ReLU → BatchNorm → Dropout(0.2)
    │
    ▼
Linear(2) → [mean, log_variance]
```

### Input Features (17)

**Vehicle (5):**

- mass_kg, drag_coefficient, frontal_area_m2
- battery_capacity_kwh, motor_efficiency

**Route (4):**

- distance_km, elevation_gain_m, elevation_loss_m
- target_speed_kmh

**Weather (4):**

- temperature_c, wind_speed_mps, wind_direction_deg
- humidity_percent

**Driver (4):**

- aggression_factor, regen_preference
- aux_power_factor, initial_soc

## Confidence Scoring

### Components

| Component           | Weight | Description              |
| ------------------- | ------ | ------------------------ |
| Model Uncertainty   | 40%    | From Monte Carlo Dropout |
| Physics Agreement   | 30%    | ML vs physics comparison |
| Historical Accuracy | 20%    | Past performance         |
| Data Quality        | 10%    | Input validation         |

### Interpretations

| Score     | Interpretation | Action               |
| --------- | -------------- | -------------------- |
| ≥ 0.85    | HIGH           | Use ML prediction    |
| 0.70-0.85 | MEDIUM         | Consider validation  |
| 0.50-0.70 | LOW            | Use physics fallback |
| < 0.50    | VERY_LOW       | Physics required     |

## Validation Suite

### Running Validation

```bash
python scripts/validate_ml_models.py --model models/energy_predictor.pth
```

### Test Categories

1. **Physics Consistency**: ML within 10% of physics on random scenarios
2. **Confidence Calibration**: Predicted confidence matches actual accuracy
3. **No Hallucinations**: No predictions outside physical limits
4. **Uncertainty Correlation**: Uncertainty correlates with error

### Success Criteria

| Metric                | Target |
| --------------------- | ------ |
| Test MAPE             | < 5%   |
| Within 10% of physics | ≥ 95%  |
| Hallucinations        | 0      |
| Calibration error     | < 15%  |

## Troubleshooting

### ML Model Not Loading

```
ERROR: ML model not available
```

**Solution**: Train the model first:

```bash
python scripts/train_ml_model.py
```

### High Validation Errors

If validation shows high errors:

1. Generate more training data
2. Train for more epochs
3. Check for data distribution issues

### Slow Training

Training taking too long:

1. Use GPU if available (`--device cuda`)
2. Reduce batch size for memory
3. Use early stopping

## File Reference

```
backend/
├── app/ml/
│   ├── data_generator.py      # Training data generation
│   ├── models/
│   │   └── energy_predictor.py # Neural network
│   ├── hybrid_predictor.py    # ML-physics routing
│   ├── confidence_scorer.py   # Multi-factor confidence
│   └── validation.py          # Self-verification
├── app/simulation/
│   └── driving_profiles.py    # Speed profiles
├── scripts/
│   ├── train_ml_model.py      # Training script
│   └── validate_ml_models.py  # Validation script
└── models/
    ├── energy_predictor.pth   # Trained model
    └── metrics.json           # Training metrics
```
