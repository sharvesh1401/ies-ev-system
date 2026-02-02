import math
from typing import Dict, Any

class BatteryModel:
    """
    Electrochemical battery model with thermal dynamics.
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
        
        # Derived
        self.capacity_ah = (capacity_kwh * 1000.0) / voltage_nominal
        
        # OCV Polynomial Coefficients (Approx Li-ion)
        # V = 3.0 + 0.3*SOC + 0.3*SOC^2 + 0.1*SOC^3 (where SOC is 0-1 normalized? Let's assume normalized)
        # Prompt said: V = a + b*SOC + c*SOC^2 + d*SOC^3
        self.ocv_coeffs = [3.0, 0.3, 0.3, 0.1] 
    
    def calculate_ocv(self, soc_percent: float) -> float:
        """
        Calculate Open Circuit Voltage from SOC percentage.
        """
        soc_norm = soc_percent / 100.0
        # Typical Li-ion curve (simplified)
        # Preventing OCV exploding if SOC > 100 or < 0
        soc_norm = max(0.0, min(1.0, soc_norm))
        
        v_ocv = (self.ocv_coeffs[0] + 
                 self.ocv_coeffs[1] * soc_norm + 
                 self.ocv_coeffs[2] * (soc_norm**2) + 
                 self.ocv_coeffs[3] * (soc_norm**3))
                 
        # Scale to pack voltage usually? 
        # The prompt implies this formula gives cell/module voltage, usually 3.7V range.
        # But V_nom is like 400V.
        # So we need to scale this base curve to match V_nom.
        # Base curve avg is roughly 3.0 + 0.15 + 0.1 + 0.05 ~ 3.3V?
        # Let's assume the coefficients provided in prompt are for a "unit" voltage ~ 3.7V.
        # Scaling factor = V_nom / 3.7 (approx)
        
        base_voltage_approx = 3.7 
        scaling_factor = self.V_nom / base_voltage_approx
        
        return v_ocv * scaling_factor

    def calculate_current(self, power_kw: float, soc_percent: float) -> float:
        """
        Calculate current required for power (kW). Positive current = discharge.
        Solves Quadratic: I^2*R - V_ocv*I + P = 0
        """
        P_watts = power_kw * 1000.0
        V_ocv = self.calculate_ocv(soc_percent)
        R = self.R_int
        
        # Discriminant: b^2 - 4ac -> (-V)^2 - 4(R)(P) = V^2 - 4RP
        discriminant = V_ocv**2 - 4 * R * P_watts
        
        if discriminant < 0:
            # Cannot deliver this power (Voltage collapse)
            # Return max current corresponding to max power point
            # I_max = V / 2R
            return V_ocv / (2 * R)
            
        # Solutions: (V +/- sqrt(D)) / 2R
        # For Discharge (P>0): usually lower current is the stable one (V terminal > V/2)
        # For Charge (P<0): D is > V^2, sqrt(D) > V.
        # minus root: (V - large) / 2R -> Negative current (Charging). Correct.
        # plus root: (V + large) / 2R -> Positive huge current. Incorrect.
        
        # So minus root works for both charge and discharge in standard usage?
        # Discharge: (V - small) / 2R -> Positive. Correct. 
        # Charge: (V - large) / 2R -> Negative. Correct.
        
        I = (V_ocv - math.sqrt(discriminant)) / (2 * R)
        return I

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
        # 1. Calculate Current
        current_a = self.calculate_current(power_demand_kw, current_soc)
        
        # 2. Update SOC
        # I = dQ/dt -> Q = I * dt (Coulombs or Amp-seconds)
        # dSOC = -I * dt / Capacity_Ah * 3600 (if dt in seconds)
        # Note: I is positive for discharge, so SOC decreases.
        
        # Capacity in As
        cap_as = self.capacity_ah * 3600.0
        delta_soc = -(current_a * dt / cap_as) * 100.0
        
        new_soc = current_soc + delta_soc
        new_soc = max(0.0, min(100.0, new_soc))
        
        # 3. Thermal Dynamics
        # Heat Gen = I^2 * R
        p_loss_w = (current_a ** 2) * self.R_int
        
        # Heat Transfer = h * A * (T - T_amb)
        p_cool_w = self.h_transfer * self.A_surface * (current_temp_c - ambient_temp_c)
        
        # dT/dt = (Gen - Cool) / ThermalMass
        dT = (p_loss_w - p_cool_w) * dt / self.thermal_mass
        
        new_temp = current_temp_c + dT
        
        # 4. Terminal Voltage
        # V_term = V_ocv - I * R
        # Discharge (I>0) -> V_term < V_ocv
        # Charge (I<0) -> V_term > V_ocv
        v_ocv = self.calculate_ocv(new_soc)
        v_term = v_ocv - current_a * self.R_int
        
        return {
            'soc': new_soc,
            'temperature': new_temp,
            'voltage': v_term,
            'current': current_a,
            'ocv': v_ocv,
            'heat_gen_w': p_loss_w
        }
