import math
from typing import Dict, Any, Optional
from app.simulation.converters import BatteryEnergyConverter

class BatteryModel:
    """
    Electrochemical battery model with thermal dynamics.
    Correctly accounts for energy <-> SOC relationship using BatteryEnergyConverter.
    """
    
    def __init__(
        self,
        capacity_kwh: float,
        voltage_nominal: float,
        internal_resistance_ohm: float,
        thermal_mass_j_per_k: float = 50000.0,
        surface_area_m2: float = 2.0,
        heat_transfer_coeff: float = 10.0
    ):
        self.capacity_kwh = capacity_kwh
        self.V_nom = voltage_nominal
        self.R_int = internal_resistance_ohm
        self.thermal_mass = thermal_mass_j_per_k
        self.A_surface = surface_area_m2
        self.h_transfer = heat_transfer_coeff
        
        # CRITICAL: Use unified converter
        self.energy_converter = BatteryEnergyConverter(capacity_kwh)
        
        # Derived
        self.capacity_ah = (capacity_kwh * 1000.0) / voltage_nominal
        
        # OCV Polynomial Coefficients (Approx Li-ion)
        self.ocv_coeffs = [3.0, 0.3, 0.3, 0.1] 
    
    def calculate_ocv(self, soc_percent: float) -> float:
        """
        Calculate Open Circuit Voltage from SOC percentage.
        """
        soc_norm = soc_percent / 100.0
        soc_norm = max(0.0, min(1.0, soc_norm))
        
        v_ocv = (self.ocv_coeffs[0] + 
                 self.ocv_coeffs[1] * soc_norm + 
                 self.ocv_coeffs[2] * (soc_norm**2) + 
                 self.ocv_coeffs[3] * (soc_norm**3))
                 
        base_voltage_approx = 3.7 
        scaling_factor = self.V_nom / base_voltage_approx
        
        return v_ocv * scaling_factor

    def calculate_current_from_power(self, power_kw: float, soc_percent: float) -> float:
        """
        Calculate current required for power (kW). Positive current = discharge.
        Solves Quadratic: I^2*R - V_ocv*I + P = 0
        """
        # P is battery terminal power (which includes internal losses if we view P as desired output?)
        # Wait, if P is what the load SEES, then P_load = V_term * I = (V_ocv - I*R) * I = V_ocv*I - I^2*R
        # So I^2*R - V_ocv*I + P_load = 0. THIS IS CORRECT.
        
        P_watts = power_kw * 1000.0
        V_ocv = self.calculate_ocv(soc_percent)
        R = self.R_int
        
        discriminant = V_ocv**2 - 4 * R * P_watts
        
        if discriminant < 0:
            return V_ocv / (2 * R)
            
        I = (V_ocv - math.sqrt(discriminant)) / (2 * R)
        return I

    def calculate_current(self, power_kw: float, soc_percent: float) -> float:
        """Deprecated alias for calculate_current_from_power."""
        return self.calculate_current_from_power(power_kw, soc_percent)

    def update_state(
        self,
        current_soc: float,
        current_temp_c: float,
        power_demand_kw: float,
        ambient_temp_c: float,
        dt: float
    ) -> Dict[str, float]:
        """
        Update battery state for one time step.
        """
        # 1. Calculate Electrical State first (explicit scheme)
        # We need Current to know the strictly conserved Chemical Energy (P_term + I^2*R)
        v_ocv_current = self.calculate_ocv(current_soc)
        current_a = self.calculate_current_from_power(power_demand_kw, current_soc)
        
        # Chemical Power = V_ocv * I (Total power extracted from chemistry)
        # This accounts for both external load (P) and internal heating (I^2*R)
        # Power is in kW. V*I is Watts.
        p_chemical_w = v_ocv_current * current_a
        p_chemical_kw = p_chemical_w / 1000.0
        
        energy_step_kwh = p_chemical_kw * (dt / 3600.0)
        
        # 2. Update SOC based on CHEMICAL Energy
        new_soc = self.energy_converter.update_soc_from_energy_delta(
            current_soc,
            energy_step_kwh
        )
        
        # 3. Thermal Dynamics
        p_loss_w = (current_a ** 2) * self.R_int
        p_cool_w = self.h_transfer * self.A_surface * (current_temp_c - ambient_temp_c)
        dT = (p_loss_w - p_cool_w) * dt / self.thermal_mass
        new_temp = current_temp_c + dT
        
        # 4. Terminal Voltage
        # Re-calc OCV at new SOC? Or usage average? 
        # Usually for V_term measurement we use the instantaneous state.
        # But for consistency with Current calc (which used current_soc), we might use current_soc?
        # However, physically, at the End of Step, SOC is new_soc.
        # The Current was average current over the step? 
        # Let's use new_soc for reporting the final voltage state.
        v_ocv_new = self.calculate_ocv(new_soc)
        v_term = v_ocv_new - current_a * self.R_int
        
        return {
            'soc': new_soc,
            'temperature': new_temp,
            'voltage': v_term,
            'current': current_a,
            'ocv': v_ocv_new,
            'heat_gen_w': p_loss_w,
            'energy_step_kwh': energy_step_kwh 
        }

