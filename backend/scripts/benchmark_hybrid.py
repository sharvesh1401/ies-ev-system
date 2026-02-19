"""
Benchmark Hybrid Predictor — CLI script.

Runs timing benchmarks comparing physics-only vs variable-speed
prediction performance.

Usage:
    python scripts/benchmark_hybrid.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("  HYBRID PREDICTOR BENCHMARK")
    print("=" * 60)

    from app.simulation.integrated_engine import IntegratedSimulationEngine
    from app.simulation.schemas import (
        VehicleParameters, RouteParameters, EnvironmentParameters
    )

    vehicle = VehicleParameters(
        mass_kg=1500,
        drag_coefficient=0.30,
        frontal_area_m2=2.2,
        rolling_resistance_coefficient=0.012,
        battery_capacity_kwh=60,
        battery_voltage_nominal=400,
        battery_internal_resistance_ohm=0.05,
    )
    route = RouteParameters(distance_km=20, target_velocity_mps=25)
    env = EnvironmentParameters(temperature_c=25)

    engine = IntegratedSimulationEngine()

    # --- Benchmark: Physics constant-speed ---
    n_runs = 5
    print(f"\n1. Physics (constant speed) — {n_runs} runs")
    physics_times = []
    for i in range(n_runs):
        t0 = time.time()
        result = engine.simulate(vehicle, route, env)
        elapsed = (time.time() - t0) * 1000
        physics_times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.1f} ms — Energy: {result.total_energy_kwh:.3f} kWh")

    avg_physics = sum(physics_times) / len(physics_times)
    print(f"   Average: {avg_physics:.1f} ms")

    # --- Benchmark: Variable speed (city) ---
    print(f"\n2. Variable speed (city) — {n_runs} runs")
    try:
        from app.simulation.variable_speed_engine import simulate_variable_speed

        variable_times = []
        for i in range(n_runs):
            t0 = time.time()
            result = simulate_variable_speed(
                vehicle_params=vehicle,
                route_params=route,
                environment_params=env,
                profile_type="city",
                driver_style="moderate",
                seed=42 + i,
            )
            elapsed = (time.time() - t0) * 1000
            variable_times.append(elapsed)
            print(
                f"   Run {i+1}: {elapsed:.1f} ms — "
                f"Energy: {result.total_energy_kwh:.3f} kWh, "
                f"Stops: {result.num_stops}"
            )

        avg_variable = sum(variable_times) / len(variable_times)
        print(f"   Average: {avg_variable:.1f} ms")

    except Exception as e:
        print(f"   Error: {e}")
        avg_variable = 0

    # --- Benchmark: Variable speed (highway) ---
    print(f"\n3. Variable speed (highway) — {n_runs} runs")
    try:
        highway_times = []
        for i in range(n_runs):
            t0 = time.time()
            result = simulate_variable_speed(
                vehicle_params=vehicle,
                route_params=route,
                environment_params=env,
                profile_type="highway",
                driver_style="moderate",
                seed=42 + i,
            )
            elapsed = (time.time() - t0) * 1000
            highway_times.append(elapsed)
            print(
                f"   Run {i+1}: {elapsed:.1f} ms — "
                f"Energy: {result.total_energy_kwh:.3f} kWh"
            )

        avg_highway = sum(highway_times) / len(highway_times)
        print(f"   Average: {avg_highway:.1f} ms")

    except Exception as e:
        print(f"   Error: {e}")
        avg_highway = 0

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Physics (constant speed): {avg_physics:>8.1f} ms avg")
    if avg_variable:
        print(f"  Variable speed (city):    {avg_variable:>8.1f} ms avg")
    if avg_highway:
        print(f"  Variable speed (highway): {avg_highway:>8.1f} ms avg")
    print("=" * 60)


if __name__ == "__main__":
    main()
