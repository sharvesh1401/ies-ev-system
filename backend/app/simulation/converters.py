class BatteryEnergyConverter:
    """
    Single source of truth for energy <-> SOC conversion.
    
    INVARIANT: These two operations must be exact inverses:
    - energy_to_soc(soc_to_energy(x)) == x
    - soc_to_energy(energy_to_soc(x)) == x
    """
    
    def __init__(self, capacity_kwh: float):
        self.capacity_kwh = capacity_kwh
    
    def soc_to_energy(self, soc_percent: float) -> float:
        """
        Convert SOC percentage to absolute energy (kWh).
        
        Args:
            soc_percent: State of charge (0-100%)
        
        Returns:
            Energy in kWh
        """
        return (soc_percent / 100.0) * self.capacity_kwh
    
    def energy_to_soc(self, energy_kwh: float) -> float:
        """
        Convert absolute energy (kWh) to SOC percentage.
        
        Args:
            energy_kwh: Energy in kWh
        
        Returns:
            SOC percentage (0-100%)
        """
        return (energy_kwh / self.capacity_kwh) * 100.0
    
    def update_soc_from_energy_delta(
        self,
        current_soc: float,
        energy_delta_kwh: float
    ) -> float:
        """
        Update SOC based on energy change.
        
        CRITICAL: This is the ONLY method that should update SOC!
        
        Args:
            current_soc: Current SOC (%)
            energy_delta_kwh: Energy change (positive = discharge, negative = charge)
        
        Returns:
            New SOC (%)
        """
        # Convert current SOC to energy
        current_energy = self.soc_to_energy(current_soc)
        
        # Apply delta
        new_energy = current_energy - energy_delta_kwh
        
        # Convert back to SOC
        new_soc = self.energy_to_soc(new_energy)
        
        # Clamp to valid range
        return max(0.0, min(100.0, new_soc))
    
    def verify_invertibility(self) -> bool:
        """
        Self-test: Verify conversion is invertible.
        
        Returns:
            True if conversions are exact inverses
        """
        test_socs = [0, 25, 50, 75, 100]
        for soc in test_socs:
            energy = self.soc_to_energy(soc)
            recovered_soc = self.energy_to_soc(energy)
            error = abs(soc - recovered_soc)
            if error >= 1e-10:
                return False
        return True
