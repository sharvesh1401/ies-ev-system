from typing import List, Dict
from app.simulation.schemas import EnergyBreakdown, VehicleParameters, SimulationStep

class EnergyCalculator:
    """
    Integrate power flow and calculate total energy consumption.
    """
    
    def __init__(self, auxiliary_power_kw: float = 0.5):
        self.aux_power_kw = auxiliary_power_kw
    
    def calculate_power_flow(
        self,
        force_tractive: float,
        velocity_mps: float,
        vehicle: VehicleParameters,
        dt: float
    ) -> Dict[str, float]:
        """
        Calculate power flow through the drivetrain.
        
        Returns:
            Dictionary with motor_power, battery_power, energy_step
        """
        # Mechanical Power at Wheels: P = F * v
        power_wheels_kw = (force_tractive * velocity_mps) / 1000.0
        
        # Calculate Battery Power
        # Positive Power = Discharging (Driving)
        # Negative Power = Charging (Regen)
        
        if power_wheels_kw >= 0:
            # Driving Mode
            # Battery must supply MORE power due to losses
            # P_bat = P_wheels / (eta_motor * eta_inverter...) + P_aux
            efficiency = vehicle.motor_efficiency # Combined drivetrain efficiency
            power_battery_kw = (power_wheels_kw / efficiency) + vehicle.auxiliary_power_kw
            
        else:
            # Braking/Regen Mode
            # Battery receives LESS power due to losses
            # P_bat = P_wheels * eta_regen + P_aux
            # Note: P_wheels is negative. P_bat will be less negative (closer to 0) or positive if Aux > Regen
            efficiency = vehicle.regen_efficiency
            power_battery_kw = (power_wheels_kw * efficiency) + vehicle.auxiliary_power_kw
            
            # Limit regen? Usually handled by battery model acceptance, 
            # but calculator just computes potential flow here.
        
        # Energy for this step (kWh)
        energy_step_kwh = power_battery_kw * (dt / 3600.0)
        
        return {
            'power_wheels_kw': power_wheels_kw,
            'power_battery_kw': power_battery_kw,
            'energy_step_kwh': energy_step_kwh
        }
    
    def calculate_energy_breakdown(
        self,
        trajectory: List[SimulationStep],
        vehicle: VehicleParameters
    ) -> EnergyBreakdown:
        """
        Break down total energy into physical components.
        """
        if not trajectory:
            return EnergyBreakdown(
                kinetic_kwh=0, potential_kwh=0, aerodynamic_kwh=0, 
                rolling_kwh=0, auxiliary_kwh=0, total_kwh=0
            )
            
        # Initialize
        e_aero = 0.0
        e_roll = 0.0
        e_aux = 0.0
        
        # Integrate resistive forces
        for i in range(len(trajectory) - 1):
            step = trajectory[i]
            dt = trajectory[i+1].time_s - step.time_s
            if dt <= 0: continue
            
            # Average velocity for integration
            v_avg = (step.velocity_mps + trajectory[i+1].velocity_mps) / 2.0
            
            # Power = Force * Velocity
            # Energy = Power * Time
            
            # Aero
            # Force is always opposing, work is Force * distance
            # distance = v * dt
            dist = v_avg * dt
            
            e_aero += abs(step.force_aero_n) * dist
            e_roll += abs(step.force_roll_n) * dist
            e_aux += vehicle.auxiliary_power_kw * 1000.0 * dt # Watts * seconds
            
        # Convert Joules to kWh
        to_kwh = 1.0 / 3_600_000.0
        e_aero_kwh = e_aero * to_kwh
        e_roll_kwh = e_roll * to_kwh
        e_aux_kwh = e_aux * to_kwh
        
        # State differences
        start = trajectory[0]
        end = trajectory[-1]
        
        # Kinetic Energy: 0.5 * m * v^2
        ke_start = 0.5 * vehicle.mass_kg * (start.velocity_mps ** 2)
        ke_end = 0.5 * vehicle.mass_kg * (end.velocity_mps ** 2)
        e_kinetic_kwh = (ke_end - ke_start) * to_kwh
        
        # Potential Energy: m * g * h
        g = 9.81
        pe_start = vehicle.mass_kg * g * start.elevation_m
        pe_end = vehicle.mass_kg * g * end.elevation_m
        e_potential_kwh = (pe_end - pe_start) * to_kwh
        
        # Total
        # E_total = Delta_KE + Delta_PE + E_resistive
        # This is the "Physical" total energy change of the system
        total_kwh = e_kinetic_kwh + e_potential_kwh + e_aero_kwh + e_roll_kwh + e_aux_kwh
        
        return EnergyBreakdown(
            kinetic_kwh=e_kinetic_kwh,
            potential_kwh=e_potential_kwh,
            aerodynamic_kwh=e_aero_kwh,
            rolling_kwh=e_roll_kwh,
            auxiliary_kwh=e_aux_kwh,
            total_kwh=total_kwh
        )
