import math
import numpy as np
from typing import List, Tuple, Dict
from app.simulation.schemas import VehicleParameters, RouteParameters, EnvironmentParameters, SimulationStep

class VehicleDynamicsEngine:
    """
    Simulate longitudinal vehicle motion following Newton's laws of motion.
    """
    
    def __init__(self):
        self.g = 9.81  # m/s²
        # Initial state variables
        self.v = 0.0
        self.x = 0.0
        self.t = 0.0
    
    def simulate_step(
        self,
        vehicle: VehicleParameters,
        environment: EnvironmentParameters,
        target_velocity: float,
        grade_rad: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Calculate one time step of vehicle dynamics.
        
        Args:
            vehicle: Vehicle parameters
            environment: Environment parameters (air density, wind)
            target_velocity: Desired speed (m/s)
            grade_rad: Road grade angle in radians
            dt: Time step duration (s)
            
        Returns:
            Dictionary of calculated forces and state updates
        """
        # 1. Environmental Forces
        
        # Wind effect
        # Simple longitudinal model: v_rel = v_vehicle - v_wind * cos(wind_angle - heading)
        # Assuming simplified headwind component for now
        headwind = environment.wind_speed_mps  # Simplified
        v_relative = self.v - headwind
        
        # Calculate Air Density (simplified or passed in)
        # If environment doesn't have density, calculate or use standard
        # For now assume standard or simple correction
        rho = 1.225 # Standard
        
        # Aerodynamic Drag: F = 0.5 * rho * Cd * A * v_rel^2
        # Direction opposes relative velocity
        # Use abs(v_relative) * v_relative to preserve directionality if needed, 
        # but drag usually opposes motion.
        F_aero = 0.5 * rho * vehicle.drag_coefficient * vehicle.frontal_area_m2 * (v_relative ** 2)
        if v_relative < 0: F_aero = -F_aero # Drag acts in opposite direction if wind pushes execution
        
        # Rolling Resistance: F = m * g * Cr * cos(theta)
        # Always opposes motion direction
        F_roll = vehicle.mass_kg * self.g * vehicle.rolling_resistance_coefficient * math.cos(grade_rad)
        
        # Gravitational Force: F = m * g * sin(theta)
        # Positive grade (uphill) -> Positive force resisting motion (in resistive context)
        # Or standard F_g = mg sin theta, usually defined as force vehicle must overcome
        F_grade = vehicle.mass_kg * self.g * math.sin(grade_rad)
        
        # Total Resistive Force
        # Note: If v=0, F_roll is bounded by Tractive Force (static friction logic), 
        # but for simple simulation we assume it acts if v>0 or Tractive > Resists
        F_resistive = F_aero + F_roll + F_grade
        
        # 2. Tractive Force (Control)
        # Simple P-controller for driver
        kp = 1000.0 # Gain
        error = target_velocity - self.v
        F_tractive = kp * error + F_resistive # Feed-forward resistive
        
        # Limit by Motor Power
        # P = F * v -> F_max = P_max / v
        # Assume a max motor power (e.g. 150kW) if not specified, 
        # but vehicle params usually have limits. Added dummy max for now or derived.
        # Let's assume a realistic max force or power limit. 
        # For this phase, we'll assume infinite or very high limit unless params added later.
        # BUT, schemas has motor_efficiency but not max_power. 
        # Let's clamp F_tractive loosely to physics like 10000N or so if needed, 
        # or better: assume motor can deliver required power for the route within reason.
        
        # 3. Newton's 2nd Law: F_net = ma
        # ma = F_tractive - F_resistive
        # a = (F_tractive - F_resistive) / m
        
        # Add basic friction/brakes logic: if v=0 and F_tractive < F_resistive (downhill), 
        # handled by math naturally?
        # Careful with F_roll at zero speed.
        
        if self.v < 0.01 and F_tractive < F_resistive and error <= 0:
             # Stopped and not trying to move
             a = 0.0
             F_tractive = F_resistive # Balanced
        else:
             a = (F_tractive - F_resistive) / vehicle.mass_kg
        
        # 4. Integrate State (Euler)
        v_new = self.v + a * dt
        if v_new < 0: v_new = 0 # No reverse gear for now
        
        x_new = self.x + self.v * dt
        
        # Update internal state
        self.v = v_new
        self.x = x_new
        self.t += dt
        
        return {
            'time': self.t,
            'velocity': self.v,
            'acceleration': a,
            'position': self.x,
            'F_tractive': F_tractive,
            'F_aero': F_aero,
            'F_roll': F_roll,
            'F_grade': F_grade,
            'grade_rad': grade_rad
        }

    def simulate_trajectory(
        self, 
        vehicle: VehicleParameters,
        route: RouteParameters,
        environment: EnvironmentParameters,
        dt: float = 1.0
    ) -> List[Dict[str, float]]:
        """
        Run full trajectory simulation
        """
        trajectory = []
        self.v = 0.0 # Start from stop
        self.x = 0.0
        self.t = 0.0
        
        total_steps = int(route.distance_km * 1000 / (route.target_velocity_mps * dt)) 
        # Rough estimate, better to loop until distance reached
        
        max_time = 3600 * 20 # 20 hour limit
        
        while self.x < route.distance_km * 1000 and self.t < max_time:
            # Get local grade
            grade_pct = environment.road_grade_percent # Simplification
            # If profile exists, interpolate:
            if route.elevation_profile:
                # TODO: Implement interpolation
                pass
                
            grade_rad = math.atan(grade_pct / 100.0)
            
            step_result = self.simulate_step(
                vehicle, environment, route.target_velocity_mps, grade_rad, dt
            )
            trajectory.append(step_result)
            
        return trajectory
