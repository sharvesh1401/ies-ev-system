import pytest
from app.simulation.converters import BatteryEnergyConverter
from app.simulation.battery_model import BatteryModel
from app.simulation.energy_calculator import EnergyCalculator
from app.simulation.efficiency import EfficiencyModel
from app.simulation.schemas import VehicleParameters, SimulationStep

class TestEnergyConservation:
    
    def test_battery_energy_converter_invertibility(self):
        """Verify that SOC <-> Energy conversion is perfectly invertible."""
        converter = BatteryEnergyConverter(capacity_kwh=75.0)
        assert converter.verify_invertibility()
        
        # Spot checks
        assert abs(converter.soc_to_energy(50.0) - 37.5) < 1e-9
        assert abs(converter.energy_to_soc(37.5) - 50.0) < 1e-9
        
    def test_battery_energy_converter_update(self):
        """Verify SOC update logic."""
        converter = BatteryEnergyConverter(capacity_kwh=100.0) # Easy math
        
        # Start at 80%, remove 10 kWh -> should be 70%
        new_soc = converter.update_soc_from_energy_delta(current_soc=80.0, energy_delta_kwh=10.0)
        assert abs(new_soc - 70.0) < 1e-9
        
        # Start at 80%, add 10 kWh (regenerate) -> should be 90%
        # energy_delta is negative for charging (removing negative energy)
        new_soc = converter.update_soc_from_energy_delta(current_soc=80.0, energy_delta_kwh=-10.0)
        assert abs(new_soc - 90.0) < 1e-9

    def test_efficiency_model_consistency(self):
        """Verify efficiency model directionality."""
        eff = EfficiencyModel()
        
        # Case 1: Driving (Battery -> Wheels)
        p_wheel = 50.0 # kW
        p_bat = eff.battery_power_from_wheel_power(p_wheel)
        assert p_bat > p_wheel # Must supply more power due to losses
        assert abs(p_bat - (p_wheel / eff.eta_drive)) < 1e-9
        
        # Verify inverse
        p_wheel_recovered = eff.wheel_power_from_battery_power(p_bat)
        assert abs(p_wheel_recovered - p_wheel) < 1e-9
        
        # Case 2: Regen (Wheels -> Battery)
        p_wheel = -50.0 # kW
        p_bat = eff.battery_power_from_wheel_power(p_wheel)
        assert abs(p_bat) < abs(p_wheel) # Battery receives less due to losses
        assert abs(p_bat - (p_wheel * eff.eta_regen)) < 1e-9
        
        # Verify inverse
        p_wheel_recovered = eff.wheel_power_from_battery_power(p_bat)
        assert abs(p_wheel_recovered - p_wheel) < 1e-9

    def test_energy_soc_consistency_integration(self):
        """
        CRITICAL TEST: Verify Energy Calculator matches Battery Model SOC change.
        Mocks a simulation step and checks if the accounting aligns.
        """
        # Setup
        cap_kwh = 75.0
        battery = BatteryModel(capacity_kwh=cap_kwh, voltage_nominal=400, internal_resistance_ohm=0.1)
        energy_calc = EnergyCalculator()
        
        # Mock Vehicle Params
        # Mock Vehicle Params
        vehicle = VehicleParameters(
            mass_kg=2000, 
            frontal_area_m2=2.5, 
            drag_coefficient=0.3,
            rolling_resistance_coefficient=0.015,
            battery_capacity_kwh=cap_kwh,
            battery_voltage_nominal=400,
            battery_internal_resistance_ohm=0.1,
            auxiliary_power_kw=0.5
        )
        
        # Simulate a step
        # 1. Calculate Power Flow (Energy Calculator)
        p_wheel = 30.0 # kW constant load
        velocity = 20.0 # m/s (not used for power calc directly here, but passed)
        dt = 3600.0 # 1 hour for easy math
        
        flow = energy_calc.calculate_power_flow(
            force_tractive=p_wheel*1000/velocity, # F = P/v
            velocity_mps=velocity,
            vehicle=vehicle,
            dt=dt
        )
        
        p_bat_calc = flow['power_battery_kw']
        e_step_calc = flow['energy_step_kwh']
        
        # 2. Update Battery State
        initial_soc = 80.0
        # Battery model calculate_current needs power demand BEFORE logic? 
        # No, update_state takes `power_demand_kw` which implies Terminal Power.
        # This matches `p_bat_calc` from EnergyCalculator (includes Aux + Drivetrain losses).
        
        # Important: Does EnergyCalculator's `power_battery_kw` include Aux?
        # Yes, code: total_battery_kw = power_battery_kw + aux
        
        new_state = battery.update_state(
            current_soc=initial_soc,
            current_temp_c=25.0,
            power_demand_kw=p_bat_calc,
            ambient_temp_c=25.0,
            dt=dt
        )
        
        # 3. Verify Consistency
        # Energy calculated by Calculator
        print(f"Energy Calculator Step (Terminal): {e_step_calc:.4f} kWh")
        
        # Internal Heat Loss (returned by battery model)
        heat_loss_w = new_state['heat_gen_w']
        heat_loss_kwh = (heat_loss_w / 1000.0) * (dt / 3600.0)
        print(f"Internal Heat Loss: {heat_loss_kwh:.4f} kWh")
        
        total_energy_consumed = e_step_calc + heat_loss_kwh
        print(f"Total Chemical Energy Consumed: {total_energy_consumed:.4f} kWh")
        
        # Energy implied by SOC change
        final_soc = new_state['soc']
        soc_delta = initial_soc - final_soc
        e_from_soc = battery.energy_converter.soc_to_energy(soc_delta) # Delta SOC -> Energy
        
        print(f"SOC Start: {initial_soc}% -> End: {final_soc}%")
        print(f"Energy from SOC: {e_from_soc:.4f} kWh")
        
        error_percent = abs(total_energy_consumed - e_from_soc) / total_energy_consumed * 100
        print(f"Error: {error_percent:.6f}%")
        
        assert error_percent < 0.01, f"Energy accounting mismatch! {error_percent}%"

if __name__ == "__main__":
    t = TestEnergyConservation()
    print("Running test_battery_energy_converter_invertibility...")
    t.test_battery_energy_converter_invertibility()
    print("PASS")
    
    print("Running test_battery_energy_converter_update...")
    t.test_battery_energy_converter_update()
    print("PASS")
    
    print("Running test_efficiency_model_consistency...")
    t.test_efficiency_model_consistency()
    print("PASS")
    
    print("Running test_energy_soc_consistency_integration...")
    t.test_energy_soc_consistency_integration()
    print("PASS")
    
    print("ALL TESTS PASSED MANUALLY")
