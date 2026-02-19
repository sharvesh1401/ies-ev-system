"""
ML Model Validation Script.

Runs the complete ML validation suite and generates a report.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Validate ML energy prediction model")
    
    parser.add_argument("--model", type=str, default="models/energy_predictor.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--num-scenarios", type=int, default=1000,
                        help="Number of test scenarios")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for validation report (JSON)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation with fewer scenarios")
    
    args = parser.parse_args()
    
    # Import dependencies
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch numpy")
        sys.exit(1)
    
    from app.ml.validation import MLValidationSuite
    from app.ml.models.energy_predictor import EnergyPredictorTrainer
    from app.simulation.integrated_engine import IntegratedSimulationEngine
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train a model first with: python scripts/train_ml_model.py")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = EnergyPredictorTrainer.load_checkpoint(str(model_path))
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Initialize physics engine
    print("Initializing physics engine...")
    physics = IntegratedSimulationEngine()
    
    # Determine number of scenarios
    n_scenarios = 100 if args.quick else args.num_scenarios
    
    # Initialize validation suite
    print(f"\nRunning validation with {n_scenarios} scenarios...")
    print("This may take several minutes...\n")
    
    suite = MLValidationSuite(
        ml_model=model,
        physics_engine=physics,
        n_test_scenarios=n_scenarios
    )
    
    # Run validation
    start_time = datetime.now()
    report = suite.run_all_tests(n_scenarios)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Add timing to report
    report.summary['duration_seconds'] = duration
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    for test in report.tests:
        print(f"\n{test.name}")
        print("-" * 40)
        print(f"  Status: {test.status.value}")
        print(f"  Score: {test.score:.2f} (threshold: {test.threshold:.2f})")
        print(f"  Message: {test.message}")
        
        if test.details:
            print("  Details:")
            for key, value in test.details.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                elif isinstance(value, list) and len(value) <= 3:
                    print(f"    {key}: {value}")
                elif not isinstance(value, list):
                    print(f"    {key}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Duration: {duration:.1f} seconds")
    print(f"Scenarios Tested: {n_scenarios}")
    print(f"Tests Passed: {report.summary['passed']}/{report.summary['total']}")
    print(f"Overall Result: {'✅ PASS' if report.overall_pass else '❌ FAIL'}")
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
    
    # Exit code based on overall result
    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
