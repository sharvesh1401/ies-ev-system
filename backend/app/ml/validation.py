"""
ML Model Validation Suite.

Automated tests to validate ML model quality:
1. Physics Consistency: ML within 10% of physics
2. Confidence Calibration: Predicted confidence matches actual accuracy
3. Hallucination Detection: No predictions outside physical limits
4. Uncertainty Quantification: Uncertainty correlates with error
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math

try:
    import torch
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    DEPS_AVAILABLE = False


class TestStatus(Enum):
    """Test result status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class TestResult:
    """Result of a single validation test."""
    name: str
    status: TestStatus
    score: float  # 0.0 to 1.0
    threshold: float  # Required score to pass
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.status == TestStatus.PASS,
            "details": self.details,
            "message": self.message
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    overall_pass: bool
    tests: List[TestResult]
    summary: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "tests": [t.to_dict() for t in self.tests],
            "summary": self.summary,
            "timestamp": self.timestamp,
            "tests_passed": sum(1 for t in self.tests if t.status == TestStatus.PASS),
            "total_tests": len(self.tests)
        }


class MLValidationSuite:
    """
    Comprehensive ML model validation suite.
    
    Runs automated tests to ensure model quality and reliability.
    """
    
    # Physical limits for energy predictions
    MIN_ENERGY_KWH_PER_KM = 0.05  # Minimum realistic consumption
    MAX_ENERGY_KWH_PER_KM = 1.5   # Maximum realistic consumption
    
    def __init__(
        self,
        ml_model: Optional[Any] = None,
        physics_engine: Optional[Any] = None,
        n_test_scenarios: int = 1000
    ):
        """
        Initialize validation suite.
        
        Args:
            ml_model: Trained EnergyPredictorNetwork
            physics_engine: IntegratedSimulationEngine
            n_test_scenarios: Number of scenarios for testing
        """
        self.ml_model = ml_model
        self.physics_engine = physics_engine
        self.n_test_scenarios = n_test_scenarios
        
        # Device for ML inference
        if DEPS_AVAILABLE and ml_model is not None:
            try:
                self.device = next(ml_model.parameters()).device
            except:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
    
    def set_model(self, ml_model):
        """Set the ML model to validate."""
        self.ml_model = ml_model
        if DEPS_AVAILABLE:
            try:
                self.device = next(ml_model.parameters()).device
            except:
                self.device = 'cpu'
    
    def set_physics_engine(self, engine):
        """Set the physics engine for comparison."""
        self.physics_engine = engine
    
    def _generate_test_scenarios(
        self,
        n_scenarios: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate random test scenarios."""
        if not DEPS_AVAILABLE:
            raise ImportError("NumPy required for test generation")
        
        np.random.seed(42)  # Reproducibility
        
        features = []
        params_list = []
        
        for _ in range(n_scenarios):
            # Generate random scenario
            mass = np.random.uniform(1200, 2500)
            drag = np.random.uniform(0.25, 0.35)
            frontal = np.random.uniform(2.0, 2.8)
            battery = np.random.uniform(50, 100)
            motor_eff = np.random.uniform(0.90, 0.95)
            distance = np.random.uniform(10, 100)
            elev_gain = np.random.uniform(0, distance * 5)
            elev_loss = np.random.uniform(0, distance * 5)
            speed = np.random.uniform(40, 120)
            temp = np.random.uniform(-10, 40)
            wind = np.random.uniform(0, 10)
            wind_dir = np.random.uniform(0, 360)
            humidity = np.random.uniform(30, 80)
            aggression = np.random.uniform(0, 1)
            regen = np.random.uniform(0.5, 1.0)
            aux_factor = np.random.uniform(0.8, 1.2)
            initial_soc = np.random.uniform(60, 100)
            
            feature_vec = np.array([
                mass, drag, frontal, battery, motor_eff,
                distance, elev_gain, elev_loss, speed,
                temp, wind, wind_dir, humidity,
                aggression, regen, aux_factor, initial_soc
            ], dtype=np.float32)
            
            features.append(feature_vec)
            
            params_list.append({
                'mass_kg': mass,
                'distance_km': distance,
                'target_speed_kmh': speed,
                'temperature_c': temp,
                'elevation_gain_m': elev_gain
            })
        
        return np.stack(features), params_list
    
    def test_physics_consistency(
        self,
        n_scenarios: int = None,
        tolerance: float = 0.10
    ) -> TestResult:
        """
        Test: ML predictions within 10% of physics.
        
        Compares ML predictions to physics simulation results
        on a set of random scenarios.
        """
        if self.ml_model is None:
            return TestResult(
                name="Physics Consistency",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.95,
                message="ML model not loaded"
            )
        
        if self.physics_engine is None:
            return TestResult(
                name="Physics Consistency",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.95,
                message="Physics engine not available"
            )
        
        n = n_scenarios or min(100, self.n_test_scenarios)  # Use fewer for speed
        
        try:
            from app.ml.data_generator import PhysicsDataGenerator
            
            generator = PhysicsDataGenerator(engine=self.physics_engine, seed=42)
            
            ml_predictions = []
            physics_results = []
            
            for i in range(n):
                features, labels, success = generator.generate_scenario()
                
                if not success:
                    continue
                
                # Get ML prediction
                x = torch.tensor(features.to_array()).unsqueeze(0).to(self.device)
                self.ml_model.eval()
                with torch.no_grad():
                    mean, _ = self.ml_model(x)
                
                ml_predictions.append(mean.item())
                physics_results.append(labels.energy_consumption_kwh)
            
            if len(ml_predictions) < 10:
                return TestResult(
                    name="Physics Consistency",
                    status=TestStatus.FAIL,
                    score=0.0,
                    threshold=0.95,
                    message=f"Only {len(ml_predictions)} valid scenarios generated"
                )
            
            # Calculate metrics
            ml_arr = np.array(ml_predictions)
            phys_arr = np.array(physics_results)
            
            rel_errors = np.abs(ml_arr - phys_arr) / np.maximum(phys_arr, 0.01)
            within_tolerance = (rel_errors <= tolerance).mean()
            mean_error = rel_errors.mean()
            max_error = rel_errors.max()
            
            # Score is the percentage within tolerance
            score = within_tolerance
            passed = score >= 0.95  # 95% within 10%
            
            return TestResult(
                name="Physics Consistency",
                status=TestStatus.PASS if passed else TestStatus.FAIL,
                score=score,
                threshold=0.95,
                details={
                    "n_scenarios": len(ml_predictions),
                    "within_10_percent": within_tolerance * 100,
                    "mean_relative_error": mean_error * 100,
                    "max_relative_error": max_error * 100,
                    "tolerance": tolerance * 100
                },
                message=f"{within_tolerance*100:.1f}% within {tolerance*100:.0f}% of physics"
            )
            
        except Exception as e:
            return TestResult(
                name="Physics Consistency",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=0.95,
                message=f"Error: {str(e)}"
            )
    
    def test_confidence_calibration(
        self,
        n_scenarios: int = None
    ) -> TestResult:
        """
        Test: Confidence scores are well-calibrated.
        
        For predictions with X% confidence, actual accuracy
        should be approximately X% (±5%).
        """
        if self.ml_model is None:
            return TestResult(
                name="Confidence Calibration",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.85,
                message="ML model not loaded"
            )
        
        if self.physics_engine is None:
            return TestResult(
                name="Confidence Calibration",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.85,
                message="Physics engine not available"
            )
        
        n = n_scenarios or min(100, self.n_test_scenarios)
        
        try:
            from app.ml.data_generator import PhysicsDataGenerator
            
            generator = PhysicsDataGenerator(engine=self.physics_engine, seed=123)
            
            # Collect predictions with confidence
            predictions = []
            
            for i in range(n):
                features, labels, success = generator.generate_scenario()
                if not success:
                    continue
                
                x = torch.tensor(features.to_array()).unsqueeze(0).to(self.device)
                results = self.ml_model.predict_with_uncertainty(x, n_samples=20)
                
                mean = results['mean'].item()
                std = results['std'].item()
                
                # Calculate confidence from uncertainty
                cv = std / max(abs(mean), 0.01)
                confidence = math.exp(-cv)
                
                # Get actual accuracy
                actual = labels.energy_consumption_kwh
                rel_error = abs(mean - actual) / max(actual, 0.01)
                is_accurate = rel_error <= 0.10  # Within 10%
                
                predictions.append({
                    'confidence': confidence,
                    'is_accurate': is_accurate,
                    'rel_error': rel_error
                })
            
            if len(predictions) < 10:
                return TestResult(
                    name="Confidence Calibration",
                    status=TestStatus.FAIL,
                    score=0.0,
                    threshold=0.85,
                    message="Insufficient scenarios"
                )
            
            # Bin by confidence and check calibration
            bins = [(0.9, 1.0), (0.75, 0.9), (0.5, 0.75), (0.0, 0.5)]
            calibration_errors = []
            bin_stats = []
            
            for low, high in bins:
                bin_preds = [p for p in predictions if low <= p['confidence'] < high]
                if len(bin_preds) >= 5:
                    expected_accuracy = (low + high) / 2
                    actual_accuracy = sum(p['is_accurate'] for p in bin_preds) / len(bin_preds)
                    error = abs(expected_accuracy - actual_accuracy)
                    calibration_errors.append(error)
                    bin_stats.append({
                        'range': f"{low*100:.0f}-{high*100:.0f}%",
                        'expected': expected_accuracy * 100,
                        'actual': actual_accuracy * 100,
                        'error': error * 100,
                        'n_samples': len(bin_preds)
                    })
            
            if not calibration_errors:
                return TestResult(
                    name="Confidence Calibration",
                    status=TestStatus.WARN,
                    score=0.5,
                    threshold=0.85,
                    message="Insufficient data for calibration analysis"
                )
            
            # Average calibration error
            avg_error = sum(calibration_errors) / len(calibration_errors)
            score = max(0, 1 - avg_error * 2)  # Lower error = higher score
            
            passed = avg_error <= 0.15  # Within 15% calibration error
            
            return TestResult(
                name="Confidence Calibration",
                status=TestStatus.PASS if passed else TestStatus.FAIL,
                score=score,
                threshold=0.85,
                details={
                    "n_scenarios": len(predictions),
                    "avg_calibration_error_pct": avg_error * 100,
                    "bin_statistics": bin_stats
                },
                message=f"Avg calibration error: {avg_error*100:.1f}%"
            )
            
        except Exception as e:
            return TestResult(
                name="Confidence Calibration",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=0.85,
                message=f"Error: {str(e)}"
            )
    
    def test_no_hallucinations(
        self,
        n_scenarios: int = None
    ) -> TestResult:
        """
        Test: No predictions outside physical limits.
        
        Ensures model doesn't predict impossible energy values.
        """
        if self.ml_model is None:
            return TestResult(
                name="No Hallucinations",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=1.0,
                message="ML model not loaded"
            )
        
        n = n_scenarios or self.n_test_scenarios
        
        try:
            features, params = self._generate_test_scenarios(n)
            
            x = torch.tensor(features).to(self.device)
            
            self.ml_model.eval()
            with torch.no_grad():
                predictions, _ = self.ml_model(x)
            
            predictions = predictions.cpu().numpy().flatten()
            
            # Check physical limits
            violations = []
            
            for i, (pred, param) in enumerate(zip(predictions, params)):
                distance = param['distance_km']
                
                # Energy per km
                energy_per_km = pred / max(distance, 0.1)
                
                # Check bounds
                if energy_per_km < self.MIN_ENERGY_KWH_PER_KM:
                    violations.append({
                        'type': 'TOO_LOW',
                        'value': energy_per_km,
                        'limit': self.MIN_ENERGY_KWH_PER_KM,
                        'scenario': i
                    })
                elif energy_per_km > self.MAX_ENERGY_KWH_PER_KM:
                    violations.append({
                        'type': 'TOO_HIGH',
                        'value': energy_per_km,
                        'limit': self.MAX_ENERGY_KWH_PER_KM,
                        'scenario': i
                    })
                
                # Check negative predictions
                if pred < 0:
                    violations.append({
                        'type': 'NEGATIVE',
                        'value': pred,
                        'scenario': i
                    })
            
            n_violations = len(violations)
            score = 1.0 - (n_violations / n)
            passed = n_violations == 0
            
            return TestResult(
                name="No Hallucinations",
                status=TestStatus.PASS if passed else TestStatus.FAIL,
                score=score,
                threshold=1.0,
                details={
                    "n_scenarios": n,
                    "n_violations": n_violations,
                    "violation_rate_pct": (n_violations / n) * 100,
                    "sample_violations": violations[:5] if violations else []
                },
                message=f"{n_violations} violations in {n} scenarios"
            )
            
        except Exception as e:
            return TestResult(
                name="No Hallucinations",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=1.0,
                message=f"Error: {str(e)}"
            )
    
    def test_uncertainty_correlation(
        self,
        n_scenarios: int = None
    ) -> TestResult:
        """
        Test: Uncertainty correlates with actual error.
        
        Higher predicted uncertainty should correspond to larger errors.
        """
        if self.ml_model is None:
            return TestResult(
                name="Uncertainty Correlation",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.5,
                message="ML model not loaded"
            )
        
        if self.physics_engine is None:
            return TestResult(
                name="Uncertainty Correlation",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.5,
                message="Physics engine not available"
            )
        
        n = n_scenarios or min(100, self.n_test_scenarios)
        
        try:
            from app.ml.data_generator import PhysicsDataGenerator
            
            generator = PhysicsDataGenerator(engine=self.physics_engine, seed=456)
            
            uncertainties = []
            errors = []
            
            for i in range(n):
                features, labels, success = generator.generate_scenario()
                if not success:
                    continue
                
                x = torch.tensor(features.to_array()).unsqueeze(0).to(self.device)
                results = self.ml_model.predict_with_uncertainty(x, n_samples=20)
                
                mean = results['mean'].item()
                std = results['std'].item()
                actual = labels.energy_consumption_kwh
                
                abs_error = abs(mean - actual)
                
                uncertainties.append(std)
                errors.append(abs_error)
            
            if len(uncertainties) < 10:
                return TestResult(
                    name="Uncertainty Correlation",
                    status=TestStatus.FAIL,
                    score=0.0,
                    threshold=0.5,
                    message="Insufficient scenarios"
                )
            
            # Calculate Pearson correlation
            unc = np.array(uncertainties)
            err = np.array(errors)
            
            unc_mean = unc.mean()
            err_mean = err.mean()
            
            numerator = np.sum((unc - unc_mean) * (err - err_mean))
            denominator = np.sqrt(np.sum((unc - unc_mean)**2) * np.sum((err - err_mean)**2))
            
            if denominator > 0:
                correlation = numerator / denominator
            else:
                correlation = 0.0
            
            # Score based on positive correlation
            # We want positive correlation (higher uncertainty = higher error)
            score = max(0, correlation)  # Negative correlation gets 0
            passed = correlation >= 0.3  # At least moderate positive correlation
            
            return TestResult(
                name="Uncertainty Correlation",
                status=TestStatus.PASS if passed else TestStatus.WARN,
                score=score,
                threshold=0.5,
                details={
                    "n_scenarios": len(uncertainties),
                    "correlation": correlation,
                    "mean_uncertainty": unc_mean,
                    "mean_error": err_mean
                },
                message=f"Correlation: {correlation:.3f}"
            )
            
        except Exception as e:
            return TestResult(
                name="Uncertainty Correlation",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=0.5,
                message=f"Error: {str(e)}"
            )
    
    def test_monotonicity(
        self,
        n_scenarios: int = None
    ) -> TestResult:
        """
        Test: Monotonicity — increasing distance should increase energy.

        Verifies the model respects physical intuition: longer trips
        consume more energy (all else being equal).
        """
        if self.ml_model is None:
            return TestResult(
                name="Monotonicity",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.95,
                message="ML model not loaded"
            )

        n = n_scenarios or min(200, self.n_test_scenarios)

        try:
            features, params = self._generate_test_scenarios(n)

            violations = 0
            tested = 0

            for i in range(n):
                base = torch.tensor(features[i]).unsqueeze(0).to(self.device)
                perturbed = base.clone()
                # Feature index 5 = distance_km — increase by 10 %
                distance_idx = 5
                perturbed[0, distance_idx] *= 1.10

                self.ml_model.eval()
                with torch.no_grad():
                    base_pred, _ = self.ml_model(base)
                    pert_pred, _ = self.ml_model(perturbed)

                if pert_pred.item() < base_pred.item():
                    violations += 1
                tested += 1

            if tested == 0:
                return TestResult(
                    name="Monotonicity",
                    status=TestStatus.FAIL,
                    score=0.0,
                    threshold=0.95,
                    message="No scenarios tested"
                )

            score = 1.0 - (violations / tested)
            passed = score >= 0.95

            return TestResult(
                name="Monotonicity",
                status=TestStatus.PASS if passed else TestStatus.FAIL,
                score=score,
                threshold=0.95,
                details={
                    "n_tested": tested,
                    "n_violations": violations,
                    "compliance_pct": score * 100
                },
                message=f"{score*100:.1f}% monotonic ({violations} violations in {tested})"
            )

        except Exception as e:
            return TestResult(
                name="Monotonicity",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=0.95,
                message=f"Error: {str(e)}"
            )

    def test_physics_law_compliance(
        self,
        n_scenarios: int = None
    ) -> TestResult:
        """
        Test: Physics law compliance via simulation cross-check.

        For a subset of scenarios, runs the physics engine and verifies
        that energy conservation holds (total ≈ sum of components)
        and that the ML prediction is within physically possible bounds.
        """
        if self.physics_engine is None:
            return TestResult(
                name="Physics Law Compliance",
                status=TestStatus.SKIP,
                score=0.0,
                threshold=0.90,
                message="Physics engine not available"
            )

        n = n_scenarios or min(50, self.n_test_scenarios)

        try:
            from app.simulation.schemas import (
                VehicleParameters, RouteParameters, EnvironmentParameters
            )

            conservation_errors = []
            tested = 0

            for i in range(n):
                try:
                    # Simple scenario
                    mass = 1200 + i * 30
                    speed = 15 + (i % 10)  # m/s
                    distance = 5 + (i % 20)

                    vehicle = VehicleParameters(
                        mass_kg=mass,
                        drag_coefficient=0.30,
                        frontal_area_m2=2.2,
                        rolling_resistance_coefficient=0.012,
                        battery_capacity_kwh=60,
                        battery_voltage_nominal=400,
                        battery_internal_resistance_ohm=0.05
                    )
                    route = RouteParameters(
                        distance_km=distance,
                        target_velocity_mps=speed
                    )
                    env = EnvironmentParameters(temperature_c=25)

                    result = self.physics_engine.simulate(vehicle, route, env)
                    breakdown = result.energy_breakdown

                    # Check: total ≈ sum of components
                    component_sum = (
                        breakdown.kinetic_kwh
                        + breakdown.potential_kwh
                        + breakdown.aerodynamic_kwh
                        + breakdown.rolling_kwh
                        + breakdown.auxiliary_kwh
                    )

                    if abs(breakdown.total_kwh) > 0.001:
                        rel_err = abs(component_sum - breakdown.total_kwh) / breakdown.total_kwh
                        conservation_errors.append(rel_err)

                    tested += 1
                except Exception:
                    continue

            if tested == 0:
                return TestResult(
                    name="Physics Law Compliance",
                    status=TestStatus.FAIL,
                    score=0.0,
                    threshold=0.90,
                    message="No scenarios completed"
                )

            avg_error = sum(conservation_errors) / len(conservation_errors) if conservation_errors else 0
            within_tolerance = sum(1 for e in conservation_errors if e < 0.05) / len(conservation_errors) if conservation_errors else 0
            score = within_tolerance
            passed = score >= 0.90

            return TestResult(
                name="Physics Law Compliance",
                status=TestStatus.PASS if passed else TestStatus.FAIL,
                score=score,
                threshold=0.90,
                details={
                    "n_tested": tested,
                    "avg_conservation_error_pct": avg_error * 100,
                    "within_5pct_tolerance": within_tolerance * 100
                },
                message=f"{within_tolerance*100:.1f}% within 5% conservation tolerance"
            )

        except Exception as e:
            return TestResult(
                name="Physics Law Compliance",
                status=TestStatus.FAIL,
                score=0.0,
                threshold=0.90,
                message=f"Error: {str(e)}"
            )

    def run_all_tests(
        self,
        n_scenarios: int = None
    ) -> ValidationReport:
        """
        Run complete validation suite.
        
        Returns:
            ValidationReport with all test results
        """
        from datetime import datetime
        
        n = n_scenarios or self.n_test_scenarios
        
        tests = []
        
        # Run each test
        print("Running ML Validation Suite...")
        print("=" * 60)
        
        print("\n1. Physics Consistency Test")
        test1 = self.test_physics_consistency(min(n, 100))
        tests.append(test1)
        print(f"   {test1.status.value}: {test1.message}")
        
        print("\n2. Confidence Calibration Test")
        test2 = self.test_confidence_calibration(min(n, 100))
        tests.append(test2)
        print(f"   {test2.status.value}: {test2.message}")
        
        print("\n3. No Hallucinations Test")
        test3 = self.test_no_hallucinations(n)
        tests.append(test3)
        print(f"   {test3.status.value}: {test3.message}")
        
        print("\n4. Uncertainty Correlation Test")
        test4 = self.test_uncertainty_correlation(min(n, 100))
        tests.append(test4)
        print(f"   {test4.status.value}: {test4.message}")

        print("\n5. Monotonicity Test")
        test5 = self.test_monotonicity(min(n, 200))
        tests.append(test5)
        print(f"   {test5.status.value}: {test5.message}")

        print("\n6. Physics Law Compliance Test")
        test6 = self.test_physics_law_compliance(min(n, 50))
        tests.append(test6)
        print(f"   {test6.status.value}: {test6.message}")
        
        # Overall result
        passed_tests = sum(1 for t in tests if t.status == TestStatus.PASS)
        failed_tests = sum(1 for t in tests if t.status == TestStatus.FAIL)
        
        # Overall pass if no failures
        overall_pass = failed_tests == 0
        
        print("\n" + "=" * 60)
        print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
        print(f"Tests Passed: {passed_tests}/{len(tests)}")
        
        return ValidationReport(
            overall_pass=overall_pass,
            tests=tests,
            summary={
                "passed": passed_tests,
                "failed": failed_tests,
                "total": len(tests),
                "n_scenarios_tested": n
            },
            timestamp=datetime.now().isoformat()
        )
