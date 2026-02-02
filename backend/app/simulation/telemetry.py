import numpy as np
from typing import List, Dict
from app.simulation.schemas import SimulationStep, SimulationResult

class TelemetryGenerator:
    """
    Generate realistic synthetic vehicle telemetry by adding noise to simulation results.
    """
    
    def __init__(self, noise_level: str = 'realistic'):
        # Standard deviations
        if noise_level == 'realistic':
            self.noise_std = {
                'velocity_mps': 0.1,      # GPS speed noise
                'acceleration_mps2': 0.05, # IMU noise
                'soc_percent': 0.5,       # BMS estimation error
                'temperature_c': 0.5,     # Thermistor noise
                'current_a': 1.0,         # Current sensor noise
                'voltage_v': 0.2          # Voltage sensor noise
            }
        elif noise_level == 'low':
            self.noise_std = {
                'velocity_mps': 0.01,
                'acceleration_mps2': 0.01,
                'soc_percent': 0.1,
                'temperature_c': 0.1,
                'current_a': 0.1,
                'voltage_v': 0.05
            }
        else: # High
            self.noise_std = {
                'velocity_mps': 0.5,
                'acceleration_mps2': 0.2,
                'soc_percent': 2.0,
                'temperature_c': 2.0,
                'current_a': 5.0,
                'voltage_v': 1.0
            }

    def add_noise_to_trajectory(self, clean_trajectory: List[SimulationStep]) -> List[SimulationStep]:
        """
        Return a new list of SimulationSteps with added noise.
        """
        noisy_trajectory = []
        
        for step in clean_trajectory:
            # Create a copy (Pydantic model .model_copy() or dict copy)
            # Pydantic v2 use model_copy()
            # We will manually construct or use copy
            
            # Since SimulationStep is immutable-ish (default pydantic), we can use model_copy(update=...)
            # But values need to be calculated first.
            
            updates = {}
            
            # Velocity
            noise_v = np.random.normal(0, self.noise_std['velocity_mps'])
            new_v = max(0.0, step.velocity_mps + noise_v)
            updates['velocity_mps'] = new_v
            
            # Acceleration
            noise_a = np.random.normal(0, self.noise_std['acceleration_mps2'])
            updates['acceleration_mps2'] = step.acceleration_mps2 + noise_a
            
            # SOC
            noise_soc = np.random.normal(0, self.noise_std['soc_percent'])
            new_soc = max(0.0, min(100.0, step.soc_percent + noise_soc))
            updates['soc_percent'] = new_soc
            
            # Temperature
            noise_t = np.random.normal(0, self.noise_std['temperature_c'])
            updates['temperature_c'] = step.temperature_c + noise_t
            
            # Current
            noise_i = np.random.normal(0, self.noise_std['current_a'])
            updates['current_a'] = step.current_a + noise_i
            
            # Voltage
            noise_u = np.random.normal(0, self.noise_std['voltage_v'])
            updates['voltage_v'] = step.voltage_v + noise_u
            
            # Create new step
            noisy_step = step.model_copy(update=updates)
            noisy_trajectory.append(noisy_step)
            
        return noisy_trajectory
