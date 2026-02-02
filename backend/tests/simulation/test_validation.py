from app.simulation.validation import PhysicsValidationSuite
from app.simulation import schemas
from typing import List

class TestValidationSuite:
    def test_detection_of_violations(self):
        """
        Ensure suite catches obvious physics violations.
        """
        suite = PhysicsValidationSuite()
        
        # Create a bogus result with energy violation
        # E_in (battery) = 1.0 kWh
        # E_out (physical) = 2.0 kWh (Impossible efficiency > 1)
        
        breakdown = schemas.EnergyBreakdown(
            kinetic_kwh=1.0, potential_kwh=1.0, 
            aerodynamic_kwh=0, rolling_kwh=0, auxiliary_kwh=0,
            total_kwh=2.0 
        )
        
        bogus_result = schemas.SimulationResult(
            trajectory=[],
            total_energy_kwh=1.0, # Battery out
            final_soc_percent=50,
            avg_velocity_mps=10,
            energy_breakdown=breakdown,
            validation_report=schemas.ValidationReport(overall_score=0, tests_passed=0, total_tests=0, results=[], interpretation=""),
            confidence_score=schemas.ConfidenceScore(overall=0, physics_validation=0, uncertainty=0, historical_accuracy=0, interpretation="", recommendations=[]),
            metadata={}
        )
        
        res = suite.test_energy_conservation(bogus_result)
        assert res.passed == False
        assert "Battery Energy" in res.details['msg']
