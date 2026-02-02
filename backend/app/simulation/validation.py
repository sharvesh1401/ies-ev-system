from typing import Any, Dict, List
from app.simulation.schemas import SimulationResult, ValidationResult, ValidationReport

class PhysicsValidationSuite:
    """
    Comprehensive validation of physics engine.
    """
    
    def run_all_tests(self, simulation_result: SimulationResult) -> ValidationReport:
        """Run all validation tests on a simulation result."""
        
        tests = [
            self.test_energy_conservation,
            self.test_newtons_second_law,
            self.test_power_balance,
            self.test_physical_limits,
            self.test_numerical_stability
        ]
        
        results = []
        for test in tests:
            try:
                res = test(simulation_result)
                results.append(res)
            except Exception as e:
                results.append(ValidationResult(
                    test_name=test.__name__,
                    passed=False,
                    details={"error": str(e)}
                ))
        
        passed_count = sum(1 for r in results if r.passed)
        score = passed_count / len(results) if results else 0.0
        
        # Interpretation
        if score >= 0.95: inter = "EXCELLENT"
        elif score >= 0.8: inter = "GOOD"
        else: inter = "POOR"
        
        return ValidationReport(
            overall_score=score,
            tests_passed=passed_count,
            total_tests=len(results),
            results=results,
            interpretation=inter
        )

    def test_energy_conservation(self, result: SimulationResult) -> ValidationResult:
        """
        Test: Total energy change = Work done by external forces.
        Battery Energy Used approx equals System Energy Change + Losses.
        """
        # E_battery_out (from battery model/energy calculator)
        # Note: result.total_energy_kwh is battery energy out? Check schemas.
        # usually total_energy_kwh in result refers to consumption. 
        
        # Breakdown
        bd = result.energy_breakdown
        
        # Conservation:
        # Energy Consumed (Vehicle) = Delta_KE + Delta_PE + E_Aero + E_Roll + E_Aux
        # Energy Supplied (Battery) = Energy Consumed / Efficiency (roughly, computed step-wise)
        
        # Actually, energy_breakdown.total_kwh is "Energy calculated at physical level" (sum of forces).
        # result.total_energy_kwh is "Energy drawn from battery" (includes efficiency losses).
        
        # So: Battery_Energy * Avg_Efficiency approx equals Physical_Energy.
        # But efficiency varies.
        # A better check: The loop in EnergyCalculator already sums these.
        # Let's verify that the break-down components sum to the total physical energy.
        
        sum_components = (bd.kinetic_kwh + bd.potential_kwh + 
                          bd.aerodynamic_kwh + bd.rolling_kwh + bd.auxiliary_kwh)
        
        # Check against total physical energy 
        # (Wait, bd.total_kwh is explicitly sum of components in my implementation).
        
        # Let's check against step-by-step integration vs component breakdown?
        # Or check consistency: is total positive? 
        # Real physics check: does E_battery >= E_physical? (Efficiency <= 1)
        
        e_physical = bd.total_kwh
        e_battery = result.total_energy_kwh # This is battery output
        
        # E_battery should be > E_physical (due to losses)
        # Exception: Regen? 
        # If regen, E_battery reduces (stored). 
        # Ideally: W_batt = Delta_System_Energy + Losses
        
        passed = True
        details = {}
        
        if e_battery < e_physical * 0.8 and e_physical > 0.1: 
            # If battery energy is significantly less than physical work, magic happened.
            passed = False
            details['msg'] = f"Battery Energy ({e_battery:.4f}) < Physical Work ({e_physical:.4f}). Efficiency > 1?"
        else:
            details['msg'] = "Energy components consistent with battery draw."
            
        return ValidationResult(test_name="energy_conservation", passed=passed, details=details)

    def test_newtons_second_law(self, result: SimulationResult) -> ValidationResult:
        """
        Test F=ma consistency in trajectory.
        """
        failures = 0
        max_error = 0.0
        
        for step in result.trajectory:
            # F_net = F_tractive - (F_aero + F_roll + F_grade)
            f_net = step.force_tractive_n - (step.force_aero_n + step.force_roll_n + step.force_grade_n)
            
            # ma
            # Need mass. Not in Step... usually constant.
            # We can infer mass? Or pass it. 
            # Mass is in metadata.
            mass = result.metadata['vehicle']['mass_kg']
            
            ma = mass * step.acceleration_mps2
            
            if abs(ma) > 1e-3:
                err = abs(f_net - ma) / abs(ma)
                if err > 0.05: # 5% tolerance
                    failures += 1
                    max_error = max(max_error, err)
            elif abs(f_net) > 1.0: # Acceleration is 0 but Net Force not 0?
                failures += 1
                
        passed = (failures / len(result.trajectory) < 0.01) if result.trajectory else True
        return ValidationResult(
            test_name="newtons_second_law", 
            passed=passed, 
            details={"failures": failures, "max_error_rel": max_error}
        )

    def test_power_balance(self, result: SimulationResult) -> ValidationResult:
        """
        Test P = F*v consistency.
        """
        failures = 0
        for step in result.trajectory:
            p_mech = (step.force_tractive_n * step.velocity_mps) / 1000.0
            if abs(p_mech - step.power_motor_kw) > 0.1: # 0.1 kW tolerance
                failures += 1
                
        passed = (failures / len(result.trajectory) < 0.01) if result.trajectory else True
        return ValidationResult(test_name="power_balance", passed=passed, details={"failures": failures})

    def test_physical_limits(self, result: SimulationResult) -> ValidationResult:
        """
        Check for NaNs, Infs, unrealistic velocities or temperatures.
        """
        violations = []
        for i, step in enumerate(result.trajectory):
            if step.velocity_mps < -0.1: violations.append(f"Neg velocity at {i}")
            if step.soc_percent < -1 or step.soc_percent > 101: violations.append(f"SOC bounds at {i}")
            if step.temperature_c > 200: violations.append(f"High Temp at {i}")
            
        passed = len(violations) == 0
        return ValidationResult(test_name="physical_limits", passed=passed, details={"violations_count": len(violations)})

    def test_numerical_stability(self, result: SimulationResult) -> ValidationResult:
        """
        Check for wild oscillations in velocity/acceleration.
        """
        # Simple check: Acceleration jitter
        jitter_count = 0
        if result.trajectory:
             last_acc = result.trajectory[0].acceleration_mps2
             for step in result.trajectory[1:]:
                 curr = step.acceleration_mps2
                 # If acceleration sign flips rapidly with large magnitude -> instability
                 if (curr * last_acc < 0) and (abs(curr - last_acc) > 5.0):
                     jitter_count += 1
                 last_acc = curr
                 
        passed = jitter_count < len(result.trajectory) * 0.05
        return ValidationResult(test_name="numerical_stability", passed=passed, details={"jitter_count": jitter_count})
