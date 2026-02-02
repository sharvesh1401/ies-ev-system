class EfficiencyModel:
    """
    Centralized efficiency model.
    
    Power flow chain:
    Battery -> Inverter -> Motor -> Transmission -> Wheels
    
    Efficiency applied ONCE in forward direction.
    """
    
    def __init__(self):
        # Component efficiencies
        self.eta_inverter = 0.97  # DC-AC conversion
        self.eta_motor = 0.95     # Electrical to mechanical
        self.eta_transmission = 0.98  # Gearbox
        
        # Combined drivetrain efficiency
        self.eta_drive = self.eta_inverter * self.eta_motor * self.eta_transmission  # ≈ 0.903
        
        # Regenerative path (worse efficiency due to bi-directional losses)
        # Wheels -> Transmission -> Motor(Gen) -> Inverter -> Battery
        # Usually similar chain, but often derated for safety/thermal or just observed lower.
        # User prompt specified 0.70 for regen.
        self.eta_regen = 0.70 
    
    def battery_power_from_wheel_power(self, p_wheel_kw: float) -> float:
        """
        Calculate battery power needed for given wheel power.
        
        Args:
            p_wheel_kw: Mechanical power at wheels (kW).
                       Positive = Driving (consuming energy).
                       Negative = Braking/Regen (generating energy).
        
        Returns:
            Electrical power at battery terminals (kW).
            Positive = Discharging.
            Negative = Charging.
        """
        if p_wheel_kw >= 0:
            # Driving: Battery -> Wheels
            # Battery must supply MORE power due to losses
            return p_wheel_kw / self.eta_drive
        else:
            # Regenerative braking: Wheels -> Battery
            # Battery receives LESS power due to losses
            return p_wheel_kw * self.eta_regen
    
    def wheel_power_from_battery_power(self, p_battery_kw: float) -> float:
        """
        Inverse: Calculate wheel power from battery power.
        """
        if p_battery_kw >= 0:
            # Discharging: Battery -> Wheels
            # Wheels receive LESS power
            return p_battery_kw * self.eta_drive
        else:
            # Charging: Wheels -> Battery
            # Wheels provided MORE power (magnitude) than what arrived at battery
            # P_bat (neg) = P_wheel (neg) * eta
            # P_wheel = P_bat / eta
            return p_battery_kw / self.eta_regen
