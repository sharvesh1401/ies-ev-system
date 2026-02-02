
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure backend is in path
sys.path.insert(0, os.getcwd())

from app.simulation import schemas
from app.simulation.integrated_engine import IntegratedSimulationEngine

def run_visual_verification():
    print("Running visual verification simulation...")
    
    # Initialize Engine
    engine = IntegratedSimulationEngine()
    
    # 1. Define Vehicle (Tesla Model 3 Standard Range Plus approx)
    vehicle = schemas.VehicleParameters(
        mass_kg=1611,  # Curb weight + driver
        drag_coefficient=0.23,
        frontal_area_m2=2.22,
        rolling_resistance_coefficient=0.011,
        battery_capacity_kwh=54.0,
        battery_voltage_nominal=350.0,
        battery_internal_resistance_ohm=0.04,
        max_power_kw=211.0,
        recuperation_efficiency=0.85
    )
    
    # 2. Define Route (Hilly route)
    # 20 km total, with a hill in the middle
    route = schemas.RouteParameters(
        distance_km=20.0,
        target_velocity_mps=25.0,  # 90 km/h
        elevation_profile=[
            (0, 50),     # Start at 50m
            (5000, 50),  # Flat for 5km
            (10000, 150), # Up to 150m (2% grade)
            (15000, 50),  # Down to 50m (-2% grade)
            (20000, 50)   # Flat
        ]
    )
    
    # 3. Define Environment
    environment = schemas.EnvironmentParameters(
        temperature_c=20.0,
        wind_speed_mps=5.0, # Headwind
        pressure_pa=101325
    )
    
    # Run Simulation
    print("Simulating...")
    result = engine.simulate(vehicle, route, environment, dt=1.0)
    
    print(f"Simulation Complete!")
    print(f"Total Energy: {result.total_energy_kwh:.2f} kWh")
    print(f"Final SOC: {result.final_soc_percent:.1f}%")
    print(f"Confidence: {result.confidence_score.overall:.2f} ({result.confidence_score.interpretation})")
    print(f"Validation: {result.validation_report.interpretation}")
    
    # Extract Data for Plotting
    traj = result.trajectory
    time = [t.time_s for t in traj]
    dist_km = [t.distance_m / 1000.0 for t in traj]
    velocity_kmh = [t.velocity_mps * 3.6 for t in traj]
    elevation_m = [t.elevation_m for t in traj]
    power_kw = [t.power_battery_kw for t in traj]
    soc = [t.soc_percent for t in traj]
    temps = [t.temperature_c for t in traj]
    
    # Create Plots
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'IES-EV Physics Verification (Energy: {result.total_energy_kwh:.2f} kWh)', fontsize=16)
    
    # 1. Elevation Profile
    axs[0, 0].plot(dist_km, elevation_m, 'g-', linewidth=2)
    axs[0, 0].set_title('Elevation Profile')
    axs[0, 0].set_ylabel('Elevation (m)')
    axs[0, 0].grid(True)
    
    # 2. Velocity Profile
    axs[0, 1].plot(dist_km, velocity_kmh, 'b-', linewidth=2)
    axs[0, 1].set_title('Velocity Profile')
    axs[0, 1].set_ylabel('Speed (km/h)')
    axs[0, 1].grid(True)
    
    # 3. Power Consumption
    axs[1, 0].plot(dist_km, power_kw, 'r-', linewidth=1)
    axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1, 0].set_title('Battery Power Flow')
    axs[1, 0].set_ylabel('Power (kW)')
    axs[1, 0].fill_between(dist_km, power_kw, 0, where=np.array(power_kw)>0, color='red', alpha=0.1)
    axs[1, 0].fill_between(dist_km, power_kw, 0, where=np.array(power_kw)<0, color='green', alpha=0.1)
    axs[1, 0].text(1, max(power_kw)*0.8, 'Discharge (Red)', color='red')
    axs[1, 0].text(1, min(power_kw)*0.8, 'Regen (Green)', color='green')
    axs[1, 0].grid(True)
    
    # 4. Battery SOC
    axs[1, 1].plot(dist_km, soc, 'm-', linewidth=2)
    axs[1, 1].set_title('State of Charge')
    axs[1, 1].set_ylabel('SOC (%)')
    axs[1, 1].set_ylim(min(soc)-1, max(soc)+1)
    axs[1, 1].grid(True)
    
    # 5. Battery Temperature
    axs[2, 0].plot(dist_km, temps, 'orange', linewidth=2)
    axs[2, 0].set_title('Battery Temperature')
    axs[2, 0].set_ylabel('Temp (°C)')
    axs[2, 0].set_xlabel('Distance (km)')
    axs[2, 0].grid(True)
    
    # 6. Energy Breakdown (Pie Chart)
    bd = result.energy_breakdown
    labels = ['Aerodynamic', 'Rolling', 'Potential (Net)', 'Auxiliary', 'Kinetic']
    sizes = [
        bd.aerodynamic_kwh, 
        bd.rolling_kwh, 
        max(0, bd.potential_kwh), # Net potential might be near zero for loop, show positive component
        bd.auxiliary_kwh,
        bd.kinetic_kwh
    ]
    # Filter small values
    labels_filtered = [l for l, s in zip(labels, sizes) if s > 0.01]
    sizes_filtered = [s for s in sizes if s > 0.01]
    
    axs[2, 1].pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', startangle=90)
    axs[2, 1].set_title('Energy Consumption Breakdown')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_file = 'simulation_results.png'
    plt.savefig(output_file)
    print(f"Visual verification saved to {output_file}")

if __name__ == "__main__":
    run_visual_verification()
