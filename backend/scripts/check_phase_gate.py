#!/usr/bin/env python3
"""
Phase 1 → Phase 2 Gate Decision Script

Evaluates all metrics and makes GO/NO-GO decision.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.metric_thresholds import (
    METRICS, check_metric_passes, get_critical_metrics, get_high_priority_metrics
)


def load_metrics():
    """Load latest validation metrics."""
    reports_dir = Path(__file__).parent.parent / "reports"
    metrics_file = reports_dir / "validation_metrics.json"
    
    if not metrics_file.exists():
        print("ERROR: No validation metrics found. Run validate_metrics.py first.")
        sys.exit(1)
    
    with open(metrics_file) as f:
        return json.load(f)


def make_gate_decision(data: dict) -> dict:
    """
    Make gate decision based on metrics.
    
    Returns:
        dict with gate_status, failures, warnings
    """
    metrics = data.get("metrics", {})
    
    critical_failures = []
    warnings = []
    
    # Check critical metrics
    critical_metrics = get_critical_metrics()
    for metric in critical_metrics:
        if metric not in metrics:
            critical_failures.append(f"{metric}: NOT MEASURED")
        elif not check_metric_passes(metric, metrics[metric]):
            critical_failures.append(
                f"{metric}: {metrics[metric]} fails threshold ({METRICS[metric]['threshold']})"
            )
    
    # Check high priority metrics
    high_metrics = get_high_priority_metrics()
    high_passed = sum(
        1 for m in high_metrics
        if m in metrics and check_metric_passes(m, metrics[m])
    )
    high_total = len(high_metrics)
    high_pass_rate = high_passed / high_total if high_total > 0 else 1.0
    
    if high_pass_rate < 0.8:
        warnings.append(
            f"Only {high_pass_rate*100:.0f}% of high priority metrics pass (need 80%)"
        )
    
    # Determine gate status
    if critical_failures:
        gate_status = "NO-GO"
    elif high_pass_rate < 0.8:
        gate_status = "CONDITIONAL"
    else:
        gate_status = "GO"
    
    return {
        "gate_status": gate_status,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "high_priority_pass_rate": high_pass_rate,
        "critical_passed": len(critical_metrics) - len(critical_failures),
        "critical_total": len(critical_metrics)
    }


def print_gate_decision(decision: dict):
    """Print formatted gate decision."""
    print("\n" + "=" * 70)
    print("PHASE 1 → PHASE 2 GATE DECISION")
    print("=" * 70)
    
    # Summary
    print(f"\nCritical Metrics: {decision['critical_passed']}/{decision['critical_total']} passed")
    print(f"High Priority Pass Rate: {decision['high_priority_pass_rate']*100:.0f}%")
    
    if decision["critical_failures"]:
        print("\nCRITICAL FAILURES:")
        for failure in decision["critical_failures"]:
            print(f"  ✗ {failure}")
    
    if decision["warnings"]:
        print("\nWARNINGS:")
        for warning in decision["warnings"]:
            print(f"  ⚠ {warning}")
    
    # Final decision
    print("\n" + "-" * 70)
    if decision["gate_status"] == "GO":
        print("║  ✓ GATE STATUS: GO                                                 ║")
        print("║  Phase 1 COMPLETE - Approved to proceed to Phase 2                 ║")
    elif decision["gate_status"] == "CONDITIONAL":
        print("║  ⚠ GATE STATUS: CONDITIONAL                                        ║")
        print("║  Phase 1 NEEDS REVIEW - Instructor approval required               ║")
    else:
        print("║  ✗ GATE STATUS: NO-GO                                              ║")
        print("║  Phase 1 INCOMPLETE - Fix critical failures before Phase 2         ║")
    print("-" * 70)


def save_decision(decision: dict):
    """Save gate decision to file."""
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "gate_decision.json", "w") as f:
        json.dump(decision, f, indent=2)


def main():
    """Main gate check."""
    print("Checking Phase 1 gate requirements...")
    
    # Load metrics
    data = load_metrics()
    
    # Make decision
    decision = make_gate_decision(data)
    
    # Save decision
    save_decision(decision)
    
    # Print decision
    print_gate_decision(decision)
    
    # Exit code
    exit_codes = {"GO": 0, "CONDITIONAL": 0, "NO-GO": 1}
    sys.exit(exit_codes[decision["gate_status"]])


if __name__ == "__main__":
    main()
