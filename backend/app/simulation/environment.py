import math
import numpy as np
from typing import List, Tuple, Dict
from app.simulation.schemas import EnvironmentParameters, SimulationStep

class EnvironmentSimulator:
    """
    Model environmental conditions affecting vehicle energy consumption.
    """
    
    def __init__(self):
        # Physical constants
        self.g = 9.81  # m/s²
        self.rho_0 = 1.225  # kg/m³ at sea level, 15°C
        self.T_0 = 288.15  # K (15°C)
        self.P_0 = 101325  # Pa (sea level)
    
    def calculate_air_density(
        self,
        temperature_c: float,
        pressure_pa: float = 101325,
        humidity_percent: float = 50
    ) -> float:
        """
        Calculate air density from temperature and pressure.
        """
        T_kelvin = temperature_c + 273.15
        
        # Ideal gas law (simplified)
        rho = self.rho_0 * (self.T_0 / T_kelvin) * (pressure_pa / self.P_0)
        
        # Humidity correction (water vapor is less dense)
        # Approximate: -0.5% density per 10% humidity
        rho *= (1 - 0.005 * humidity_percent / 10)
        
        return rho
    
    def extract_road_grade(
        self,
        elevation_profile: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Extract road grade from elevation profile.
        Returns List of (distance_m, grade_percent)
        """
        if not elevation_profile:
            return []
            
        grades = []
        for i in range(len(elevation_profile) - 1):
            x1, e1 = elevation_profile[i]
            x2, e2 = elevation_profile[i + 1]
            
            dx = x2 - x1
            de = e2 - e1
            
            if dx == 0:
                grade_percent = 0.0
            else:
                grade_percent = (de / dx) * 100
            
            grades.append((x1, grade_percent))
            
        # Last point matches previous
        if grades:
            grades.append((elevation_profile[-1][0], grades[-1][1]))
            
        return grades

    def get_grade_at_distance(
        self,
        distance_m: float,
        grade_profile: List[Tuple[float, float]],
        default_grade: float = 0.0
    ) -> float:
        """
        Get interpolated grade at a specific distance.
        """
        if not grade_profile:
            return default_grade
            
        # Simple lookup / interpolation
        # Assuming grade_profile is sorted by distance
        distances = [g[0] for g in grade_profile]
        grades = [g[1] for g in grade_profile]
        
        # We want Step function for grade (grade is constant between points usually in this context)
        # or linear interpolation. Let's use numpy interp for simplicity and valid physics
        return float(np.interp(distance_m, distances, grades))

    def calculate_wind_effect(
        self,
        vehicle_velocity_mps: float,
        vehicle_heading_deg: float,
        wind_speed_mps: float,
        wind_direction_deg: float
    ) -> Dict[str, float]:
        """
        Calculate effective wind resistance components.
        """
        # Convert to radians
        vehicle_heading_rad = math.radians(vehicle_heading_deg)
        wind_direction_rad = math.radians(wind_direction_deg)
        
        # Angle between vehicle and wind
        angle_diff = wind_direction_rad - vehicle_heading_rad
        
        # Wind components
        # Headwind is positive if blowing against vehicle
        # Wind blowing FROM 'wind_direction_deg'
        # Equation: v_wind_rel = v_wind * cos(wind_dir - vehicle_dir)
        # If wind is North (0), Vehicle is North (0), wind is from North (Headwind)
        # cos(0) = 1. Positive means helping? No, "From North" means blowing South.
        # Usually wind direction is "From". 
        # If Wind From North, blowing South. Vehicle moving North. Headwind. v_rel = v - (-wind_speed)
        # Let's standardize: Headwind Component is POSITIVE if opposing motion.
        
        headwind_component = wind_speed_mps * math.cos(angle_diff)
        crosswind_component = wind_speed_mps * math.sin(angle_diff)
        
        # Effective drag coefficient multiplier for crosswinds
        # Simplified linear model
        Cd_multiplier = 1.0 + 0.1 * abs(crosswind_component) / max(vehicle_velocity_mps, 1.0)
        
        return {
            'headwind_component_mps': headwind_component,
            'crosswind_component_mps': crosswind_component,
            'Cd_multiplier': Cd_multiplier
        }
