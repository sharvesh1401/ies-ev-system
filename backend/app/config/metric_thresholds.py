"""
Metric Thresholds Configuration for Phase 1 Validation

Defines all performance metrics and their pass/fail thresholds.
"""

METRICS = {
    # Category A: Physics Accuracy (CRITICAL)
    "energy_conservation_error_pct": {
        "threshold": 5.0,
        "optimal": 2.0,
        "priority": "CRITICAL",
        "description": "Energy balance error between calculated and SOC-derived energy"
    },
    "charge_conservation_error_pct": {
        "threshold": 3.0,
        "optimal": 1.0,
        "priority": "CRITICAL",
        "description": "SOC-energy consistency error"
    },
    "newtons_law_error_pct": {
        "threshold": 5.0,
        "optimal": 2.0,
        "priority": "CRITICAL",
        "description": "F=ma compliance error rate"
    },
    "power_balance_error_pct": {
        "threshold": 5.0,
        "optimal": 2.0,
        "priority": "CRITICAL",
        "description": "P=F·v compliance error rate"
    },
    
    # Category B: Prediction Accuracy (HIGH)
    "energy_prediction_error_pct": {
        "threshold": 10.0,
        "optimal": 5.0,
        "priority": "HIGH",
        "description": "Energy prediction error vs theoretical"
    },
    "soc_endpoint_error_pct": {
        "threshold": 5.0,
        "optimal": 2.0,
        "priority": "HIGH",
        "description": "Final SOC prediction error"
    },
    "thermal_equilibrium_error_pct": {
        "threshold": 10.0,
        "optimal": 5.0,
        "priority": "HIGH",
        "description": "Heat balance error at steady state"
    },
    
    # Category C: Numerical Stability (CRITICAL)
    "oscillation_rate": {
        "threshold": 0.05,
        "optimal": 0.02,
        "priority": "CRITICAL",
        "description": "Velocity oscillation index (fraction of steps)"
    },
    "nan_inf_rate": {
        "threshold": 0.0,
        "optimal": 0.0,
        "priority": "CRITICAL",
        "description": "Rate of NaN/Inf values (must be 0)"
    },
    
    # Category D: Physical Limits (CRITICAL)
    "velocity_bound_violations": {
        "threshold": 0,
        "optimal": 0,
        "priority": "CRITICAL",
        "description": "Count of negative velocity occurrences"
    },
    "soc_bound_violations": {
        "threshold": 0,
        "optimal": 0,
        "priority": "CRITICAL",
        "description": "Count of SOC outside 0-100% range"
    },
    
    # Category E: Test Results (CRITICAL)
    "test_pass_rate": {
        "threshold": 1.0,  # 100%
        "optimal": 1.0,
        "priority": "CRITICAL",
        "description": "Fraction of tests passing"
    },
    "test_coverage": {
        "threshold": 0.85,  # 85%
        "optimal": 0.90,
        "priority": "HIGH",
        "description": "Code coverage fraction"
    },
    
    # Category F: Confidence (MEDIUM)
    "min_confidence_score": {
        "threshold": 0.75,
        "optimal": 0.85,
        "priority": "MEDIUM",
        "description": "Minimum confidence score for normal conditions"
    }
}


def get_critical_metrics():
    """Return list of critical metric names."""
    return [k for k, v in METRICS.items() if v.get("priority") == "CRITICAL"]


def get_high_priority_metrics():
    """Return list of high priority metric names."""
    return [k for k, v in METRICS.items() if v.get("priority") == "HIGH"]


def check_metric_passes(metric_name: str, value: float) -> bool:
    """
    Check if a metric value passes its threshold.
    
    Args:
        metric_name: Name of the metric
        value: Measured value
    
    Returns:
        True if passes, False otherwise
    """
    if metric_name not in METRICS:
        return False
    
    threshold = METRICS[metric_name]["threshold"]
    
    # For rate/error metrics, lower is better
    # For coverage/pass_rate, higher is better
    if metric_name in ["test_pass_rate", "test_coverage", "min_confidence_score"]:
        return value >= threshold
    else:
        return value <= threshold
