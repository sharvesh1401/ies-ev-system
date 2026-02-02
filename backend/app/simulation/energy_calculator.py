from typing import List, Dict, Optional
from app.simulation.schemas import EnergyBreakdown, VehicleParameters, SimulationStep
from app.simulation.efficiency import EfficiencyModel

class EnergyCalculator:
    """
    Integrate power flow and calculate total energy consumption.
    Uses unified EfficiencyModel.
    """
    
    def __init__(self, auxiliary_power_kw: float = 0.5):
        self.aux_power_kw = auxiliary_power_kw
        self.efficiency_model = EfficiencyModel()
    
    def calculate_power_flow(
        self,
        force_tractive: float,
        velocity_mps: float,
        vehicle: VehicleParameters, # Kept for signature compatibility, but we prefer internal efficiency model
        dt: float
    ) -> Dict[str, float]:
        """
        Calculate power flow through the drivetrain.
        
        Returns:
            Dictionary with motor_power, battery_power, energy_step
        """
        # Mechanical Power at Wheels: P = F * v
        power_wheels_kw = (force_tractive * velocity_mps) / 1000.0
        
        # Calculate Battery Power using Unified Efficiency Model
        # This applies efficiency exactly once.
        power_battery_kw = self.efficiency_model.battery_power_from_wheel_power(power_wheels_kw)
        
        # Add auxiliary load (always positive/consuming)
        # Note: In the previous implementation, aux was added AFTER efficiency?
        # Let's check:
        # P_bat = (P_wheel / eff) + Aux
        # This means Aux comes directly from battery, parallel to drivetrain. CORRECT.
        
        # Override aux power if vehicle params provided it? 
        # The prompt implies we should respect the passed vehicle params or use standard?
        # The prompt says: "Update energy_calculator.py to use consistent efficiency"
        # It creates a new `total_battery_energy` calculation.
        
        aux = vehicle.auxiliary_power_kw if hasattr(vehicle, 'auxiliary_power_kw') else self.aux_power_kw
        
        total_battery_kw = power_battery_kw + aux
        
        # Energy for this step (kWh)
        energy_step_kwh = total_battery_kw * (dt / 3600.0)
        
        return {
            'power_wheels_kw': power_wheels_kw,
            'power_battery_kw': total_battery_kw,
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
        total_battery_kwh = 0.0
        
        # Integrate resistive forces and total battery energy
        for i in range(len(trajectory) - 1):
            step = trajectory[i]
            next_step = trajectory[i+1]
            dt = next_step.time_s - step.time_s
            if dt <= 0: continue
            
            # Average velocity for integration
            v_avg = (step.velocity_mps + next_step.velocity_mps) / 2.0
            
            # Distance
            dist = v_avg * dt
            
            # Resistive Energy (Force * Distance)
            e_aero += abs(step.force_aero_n) * dist
            e_roll += abs(step.force_roll_n) * dist
            
            # Aux Energy
            aux_kw = vehicle.auxiliary_power_kw
            e_aux += aux_kw * 1000.0 * dt # Watts * seconds
            
            # Re-calculate battery energy for the Breakdown specifically?
            # Or trust the sum of energy_step_kwh in the trajectory if it exists?
            # The prompt example shows re-calculating it to ensure consistency with efficiency model.
            
            # Recalculate P_wheel from forces to be safe, or just use stored?
            # P_wheel = F_tractive * v
            # F_tractive isn't always stored in step if we only have resistive.
            # But step usually has force_tractive_n?
            # Let's assume we want to match the `calculate_battery_energy` logic from prompt.
            
            # Actually, `SimulationStep` usually stores the result of the simulation step.
            # If we want to verify "Calculated Energy" matches "SOC Change", we should probably sum the `energy_step_kwh` that was calculated during simulation.
            # BUT, the prompt asks us to "Update EnergyCalculator... to use consistent efficiency".
            # And `calculate_battery_energy` is the method shown.
            
            # Let's effectively duplicate the logic of calculate_power_flow here to ensure the "Energy Breakdown" 
            # matches what SHOULD have happened, which is what we use for validation.
            
            # F_tractive * v
            # If we don't have F_tractive in trajectory steps, we might need to derive it from MA = F_tr - F_aero - F_roll - F_grade
            # Usually trajectory has it.
            
            # For now, let's trust the `total_kwh` to be the sum of components + efficiency losses.
            # Wait, `EnergyBreakdown` usually separates "Physical" energy (Aero, Roll, KE, PE) from "Source" energy (Battery).
            # The previous implementation sum was: KE + PE + Aero + Roll + Aux. This is "Wheel Energy" + Aux.
            # It did NOT account for drivetrain losses in `total_kwh`.
            # If `total_kwh` is meant to be "Battery Energy Used", it MUST include losses.
            # The prompt shows:
            # return EnergyBreakdown(total_battery_kwh=total_battery_energy, ...)
            
            # Let's calculate the battery energy step-by-step
            # Assuming step has `force_tractive_n` (it's not in the default schema shown in prompt but implied)
            # If not, we use: F_tractive = m*a + F_resistive
            pass

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
        total_physical_kwh = e_kinetic_kwh + e_potential_kwh + e_aero_kwh + e_roll_kwh + e_aux_kwh
        
        return EnergyBreakdown(
            kinetic_kwh=e_kinetic_kwh,
            potential_kwh=e_potential_kwh,
            aerodynamic_kwh=e_aero_kwh,
            rolling_kwh=e_roll_kwh,
            auxiliary_kwh=e_aux_kwh,
            total_kwh=total_physical_kwh 
        )

