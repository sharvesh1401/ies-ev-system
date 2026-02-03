#!/usr/bin/env python3
"""
Validation Report Generator

Generates human-readable reports from validation data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_data():
    """Load all validation data."""
    reports_dir = Path(__file__).parent.parent / "reports"
    
    data = {}
    
    # Load metrics
    metrics_file = reports_dir / "validation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data["metrics"] = json.load(f)
    
    # Load gate decision
    gate_file = reports_dir / "gate_decision.json"
    if gate_file.exists():
        with open(gate_file) as f:
            data["gate"] = json.load(f)
    
    return data


def generate_markdown_report(data: dict) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append("# Phase 1 Validation Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Gate Decision
    if "gate" in data:
        gate = data["gate"]
        status_emoji = {"GO": "✅", "NO-GO": "❌", "CONDITIONAL": "⚠️"}.get(gate["gate_status"], "❓")
        lines.append(f"\n## Gate Status: {status_emoji} {gate['gate_status']}")
        lines.append(f"\n- Critical Metrics: {gate['critical_passed']}/{gate['critical_total']} passed")
        lines.append(f"- High Priority Pass Rate: {gate['high_priority_pass_rate']*100:.0f}%")
    
    # Test Results
    if "metrics" in data:
        metrics_data = data["metrics"]
        test_results = metrics_data.get("test_results", {})
        
        lines.append("\n## Test Results")
        lines.append(f"\n- **Passed:** {test_results.get('passed', 'N/A')}")
        lines.append(f"- **Failed:** {test_results.get('failed', 0)}")
        lines.append(f"- **Total:** {test_results.get('total', 'N/A')}")
        lines.append(f"- **Pass Rate:** {test_results.get('pass_rate', 0)*100:.1f}%")
        
        # Metrics Table
        lines.append("\n## Metrics")
        lines.append("\n| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        
        metrics = metrics_data.get("metrics", {})
        for name, value in sorted(metrics.items()):
            status = "✅" if name not in [f.split(":")[0] for f in metrics_data.get("failures", [])] else "❌"
            if isinstance(value, float):
                lines.append(f"| {name} | {value:.4f} | {status} |")
            else:
                lines.append(f"| {name} | {value} | {status} |")
    
    # Failures
    if "metrics" in data and data["metrics"].get("failures"):
        lines.append("\n## Failures")
        for failure in data["metrics"]["failures"]:
            lines.append(f"- ❌ {failure}")
    
    return "\n".join(lines)


def main():
    """Generate reports."""
    print("Generating validation reports...")
    
    # Load data
    data = load_data()
    
    if not data:
        print("No validation data found. Run validate_metrics.py first.")
        sys.exit(1)
    
    # Generate markdown report
    markdown = generate_markdown_report(data)
    
    # Save report
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with open(reports_dir / "validation_report.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"Report saved to: {reports_dir / 'validation_report.md'}")
    print("\n" + "=" * 50)
    # Print with ASCII-safe format for terminal compatibility
    print(markdown.replace("✅", "[PASS]").replace("❌", "[FAIL]").replace("⚠️", "[WARN]").replace("❓", "[?]"))


if __name__ == "__main__":
    main()
