"""
Training Data Generator for ML Models.

Uses the Phase 1 physics engine to generate diverse training scenarios
with ground-truth energy consumption labels.
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from app.simulation.schemas import (
    VehicleParameters,
    RouteParameters, 
    EnvironmentParameters,
    SimulationResult
)
from app.simulation.integrated_engine import IntegratedSimulationEngine


@dataclass
class ScenarioFeatures:
    """17-dimensional feature vector for ML model input."""
    
    # Vehicle features (5)
    mass_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    battery_capacity_kwh: float
    motor_efficiency: float
    
    # Route features (4)
    distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    target_speed_kmh: float
    
    # Weather features (4)
    temperature_c: float
    wind_speed_mps: float
    wind_direction_deg: float
    humidity_percent: float
    
    # Driver features (4)
    aggression_factor: float  # 0.0 (eco) to 1.0 (aggressive)
    regen_preference: float   # 0.0 (none) to 1.0 (maximum)
    aux_power_factor: float   # Multiplier for auxiliary power
    initial_soc: float        # Starting state of charge (%)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML input."""
        return np.array([
            self.mass_kg,
            self.drag_coefficient,
            self.frontal_area_m2,
            self.battery_capacity_kwh,
            self.motor_efficiency,
            self.distance_km,
            self.elevation_gain_m,
            self.elevation_loss_m,
            self.target_speed_kmh,
            self.temperature_c,
            self.wind_speed_mps,
            self.wind_direction_deg,
            self.humidity_percent,
            self.aggression_factor,
            self.regen_preference,
            self.aux_power_factor,
            self.initial_soc
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered list of feature names."""
        return [
            "mass_kg", "drag_coefficient", "frontal_area_m2",
            "battery_capacity_kwh", "motor_efficiency",
            "distance_km", "elevation_gain_m", "elevation_loss_m", "target_speed_kmh",
            "temperature_c", "wind_speed_mps", "wind_direction_deg", "humidity_percent",
            "aggression_factor", "regen_preference", "aux_power_factor", "initial_soc"
        ]


@dataclass
class ScenarioLabels:
    """Ground-truth labels from physics simulation."""
    
    energy_consumption_kwh: float
    duration_minutes: float
    final_soc_percent: float
    avg_speed_kmh: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML output."""
        return np.array([
            self.energy_consumption_kwh,
            self.duration_minutes,
            self.final_soc_percent,
            self.avg_speed_kmh
        ], dtype=np.float32)
    
    @staticmethod
    def label_names() -> List[str]:
        """Return ordered list of label names."""
        return [
            "energy_consumption_kwh",
            "duration_minutes", 
            "final_soc_percent",
            "avg_speed_kmh"
        ]


@dataclass
class VehicleSpec:
    """Vehicle specification for sampling."""
    name: str
    mass_range: Tuple[float, float]  # kg
    drag_range: Tuple[float, float]  # Cd
    frontal_area_range: Tuple[float, float]  # m²
    battery_range: Tuple[float, float]  # kWh
    motor_eff_range: Tuple[float, float]  # fraction


# Predefined vehicle types for realistic sampling
VEHICLE_TYPES: List[VehicleSpec] = [
    VehicleSpec("compact_ev", (1200, 1500), (0.28, 0.32), (2.0, 2.3), (40, 60), (0.90, 0.94)),
    VehicleSpec("sedan_ev", (1500, 1900), (0.24, 0.30), (2.2, 2.5), (60, 85), (0.91, 0.95)),
    VehicleSpec("suv_ev", (1900, 2500), (0.30, 0.38), (2.6, 3.0), (75, 110), (0.89, 0.93)),
    VehicleSpec("luxury_ev", (2000, 2600), (0.22, 0.26), (2.3, 2.6), (90, 120), (0.92, 0.96)),
    VehicleSpec("light_ev", (1000, 1300), (0.30, 0.35), (1.8, 2.2), (30, 50), (0.88, 0.92)),
]


class PhysicsDataGenerator:
    """
    Generate training data using Phase 1 physics engine.
    
    This class creates diverse scenarios and runs physics simulations
    to generate ground-truth labels for ML model training.
    """
    
    def __init__(
        self,
        engine: Optional[IntegratedSimulationEngine] = None,
        seed: int = 42
    ):
        """
        Initialize the data generator.
        
        Args:
            engine: Physics simulation engine (created if not provided)
            seed: Random seed for reproducibility
        """
        self.engine = engine or IntegratedSimulationEngine()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
    def _sample_vehicle_params(self) -> Tuple[VehicleParameters, Dict[str, float]]:
        """Sample realistic vehicle parameters."""
        # Pick a vehicle type
        vehicle_type = self.rng.choice(VEHICLE_TYPES)
        
        # Sample within the type's ranges
        mass = self.rng.uniform(*vehicle_type.mass_range)
        drag = self.rng.uniform(*vehicle_type.drag_range)
        frontal_area = self.rng.uniform(*vehicle_type.frontal_area_range)
        battery = self.rng.uniform(*vehicle_type.battery_range)
        motor_eff = self.rng.uniform(*vehicle_type.motor_eff_range)
        
        # Standard values for other parameters
        voltage = 400 if battery < 80 else 800  # 400V or 800V system
        internal_r = self.rng.uniform(0.05, 0.15)  # Ohm
        rolling_r = self.rng.uniform(0.008, 0.012)  # Typical tire Cr
        regen_eff = self.rng.uniform(0.65, 0.80)
        aux_power = self.rng.uniform(0.3, 1.0)  # kW
        
        params = VehicleParameters(
            mass_kg=mass,
            drag_coefficient=drag,
            frontal_area_m2=frontal_area,
            rolling_resistance_coefficient=rolling_r,
            battery_capacity_kwh=battery,
            battery_voltage_nominal=voltage,
            battery_internal_resistance_ohm=internal_r,
            motor_efficiency=motor_eff,
            regen_efficiency=regen_eff,
            auxiliary_power_kw=aux_power
        )
        
        extra_features = {
            "mass_kg": mass,
            "drag_coefficient": drag,
            "frontal_area_m2": frontal_area,
            "battery_capacity_kwh": battery,
            "motor_efficiency": motor_eff
        }
        
        return params, extra_features
    
    def _sample_route_params(self) -> Tuple[RouteParameters, Dict[str, float]]:
        """Sample realistic route parameters."""
        # Distance: 5km to 200km
        distance = self.rng.uniform(5, 200)
        
        # Target speed: 30 km/h (city) to 130 km/h (highway)
        target_speed_kmh = self.rng.uniform(30, 130)
        target_speed_mps = target_speed_kmh / 3.6
        
        # Elevation profile (simplified)
        # Generate random elevation changes
        n_segments = max(2, int(distance / 10))  # 1 segment per 10km
        elevations = [(0, 0)]  # Start at (0m distance, 0m elevation)
        
        current_elev = 0
        elev_gain = 0
        elev_loss = 0
        
        for i in range(1, n_segments + 1):
            dist_m = (i / n_segments) * distance * 1000
            # Random elevation change: -50m to +50m per segment
            delta = self.rng.uniform(-50, 50)
            current_elev += delta
            
            if delta > 0:
                elev_gain += delta
            else:
                elev_loss += abs(delta)
                
            elevations.append((dist_m, current_elev))
        
        params = RouteParameters(
            distance_km=distance,
            elevation_profile=elevations,
            initial_elevation_m=0,
            target_velocity_mps=target_speed_mps
        )
        
        extra_features = {
            "distance_km": distance,
            "elevation_gain_m": elev_gain,
            "elevation_loss_m": elev_loss,
            "target_speed_kmh": target_speed_kmh
        }
        
        return params, extra_features
    
    def _sample_environment_params(self) -> Tuple[EnvironmentParameters, Dict[str, float]]:
        """Sample realistic environment parameters."""
        # Temperature: -10°C to 40°C
        temp = self.rng.uniform(-10, 40)
        
        # Wind: 0 to 15 m/s
        wind_speed = self.rng.uniform(0, 15)
        wind_dir = self.rng.uniform(0, 360)
        
        # Humidity: 20% to 95%
        humidity = self.rng.uniform(20, 95)
        
        params = EnvironmentParameters(
            temperature_c=temp,
            wind_speed_mps=wind_speed,
            wind_direction_deg=wind_dir,
            road_grade_percent=0  # We use elevation profile instead
        )
        
        extra_features = {
            "temperature_c": temp,
            "wind_speed_mps": wind_speed,
            "wind_direction_deg": wind_dir,
            "humidity_percent": humidity
        }
        
        return params, extra_features
    
    def _sample_driver_params(self) -> Dict[str, float]:
        """Sample driver behavior parameters."""
        return {
            "aggression_factor": self.rng.uniform(0, 1),
            "regen_preference": self.rng.uniform(0.5, 1.0),
            "aux_power_factor": self.rng.uniform(0.8, 1.5),
            "initial_soc": self.rng.uniform(50, 100)
        }
    
    def generate_scenario(self) -> Tuple[ScenarioFeatures, ScenarioLabels, bool]:
        """
        Generate a single training scenario.
        
        Returns:
            Tuple of (features, labels, success)
        """
        try:
            # Sample parameters
            vehicle_params, vehicle_features = self._sample_vehicle_params()
            route_params, route_features = self._sample_route_params()
            env_params, env_features = self._sample_environment_params()
            driver_features = self._sample_driver_params()
            
            # Run physics simulation
            result = self.engine.simulate(
                vehicle_params=vehicle_params,
                route_params=route_params,
                environment_params=env_params,
                dt=1.0
            )
            
            # Extract features
            features = ScenarioFeatures(
                mass_kg=vehicle_features["mass_kg"],
                drag_coefficient=vehicle_features["drag_coefficient"],
                frontal_area_m2=vehicle_features["frontal_area_m2"],
                battery_capacity_kwh=vehicle_features["battery_capacity_kwh"],
                motor_efficiency=vehicle_features["motor_efficiency"],
                distance_km=route_features["distance_km"],
                elevation_gain_m=route_features["elevation_gain_m"],
                elevation_loss_m=route_features["elevation_loss_m"],
                target_speed_kmh=route_features["target_speed_kmh"],
                temperature_c=env_features["temperature_c"],
                wind_speed_mps=env_features["wind_speed_mps"],
                wind_direction_deg=env_features["wind_direction_deg"],
                humidity_percent=env_features["humidity_percent"],
                aggression_factor=driver_features["aggression_factor"],
                regen_preference=driver_features["regen_preference"],
                aux_power_factor=driver_features["aux_power_factor"],
                initial_soc=driver_features["initial_soc"]
            )
            
            # Extract labels
            duration_minutes = (result.trajectory[-1].time_s if result.trajectory else 0) / 60.0
            avg_speed_kmh = result.avg_velocity_mps * 3.6
            
            labels = ScenarioLabels(
                energy_consumption_kwh=result.total_energy_kwh,
                duration_minutes=duration_minutes,
                final_soc_percent=result.final_soc_percent,
                avg_speed_kmh=avg_speed_kmh
            )
            
            return features, labels, True
            
        except Exception as e:
            # Return dummy values on failure
            print(f"Scenario generation failed: {e}")
            return None, None, False
    
    def generate_scenarios(
        self,
        n_samples: int,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple training scenarios.
        
        Args:
            n_samples: Number of scenarios to generate
            show_progress: Whether to print progress updates
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        features_list = []
        labels_list = []
        successful = 0
        
        for i in range(n_samples):
            features, labels, success = self.generate_scenario()
            
            if success and features is not None and labels is not None:
                features_list.append(features.to_array())
                labels_list.append(labels.to_array())
                successful += 1
                
            if show_progress and (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_samples} scenarios ({successful} successful)")
        
        print(f"Total: {successful}/{n_samples} successful scenarios")
        
        return np.stack(features_list), np.stack(labels_list)
    
    def generate_and_save_splits(
        self,
        n_samples: int,
        output_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        show_progress: bool = True
    ) -> Dict[str, str]:
        """
        Generate scenarios and save train/val/test splits to parquet.
        
        Args:
            n_samples: Total number of scenarios
            output_dir: Directory to save parquet files
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set (test = 1 - train - val)
            show_progress: Whether to print progress
            
        Returns:
            Dictionary with paths to saved files
        """
        if pd is None:
            raise ImportError("pandas is required for saving to parquet")
        
        # Generate all scenarios
        print(f"Generating {n_samples} scenarios...")
        features, labels = self.generate_scenarios(n_samples, show_progress)
        
        # Create DataFrame
        feature_names = ScenarioFeatures.feature_names()
        label_names = ScenarioLabels.label_names()
        
        df = pd.DataFrame(features, columns=feature_names)
        for i, name in enumerate(label_names):
            df[name] = labels[:, i]
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train.parquet"
        val_path = output_path / "val.parquet"
        test_path = output_path / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        print(f"Saved train ({len(train_df)}) to {train_path}")
        print(f"Saved val ({len(val_df)}) to {val_path}")
        print(f"Saved test ({len(test_df)}) to {test_path}")
        
        return {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path)
        }


def main():
    """Generate training data when run as script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ML training data")
    parser.add_argument("--num-scenarios", type=int, default=100000,
                        help="Number of scenarios to generate")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generator = PhysicsDataGenerator(seed=args.seed)
    generator.generate_and_save_splits(
        n_samples=args.num_scenarios,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
