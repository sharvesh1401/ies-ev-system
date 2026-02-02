from typing import List, Dict, Any
import numpy as np
from app.simulation.schemas import ConfidenceScore, SimulationResult, ValidationReport

class ConfidenceScorer:
    """
    Calculate confidence based on validation, numerical stability, and input parameters.
    Confidence = weighted average of:
    - Physics validation (40%)
    - Uncertainty quantification (30%)
    - Historical accuracy (30%) - Simulated/Placeholder for now
    """
    
    def calculate_confidence(
        self,
        simulation_result: SimulationResult,
        context: Dict[str, Any]
    ) -> ConfidenceScore:
        
        # 1. Physics Validation Score
        physics_score = simulation_result.validation_report.overall_score
        
        # 2. Uncertainty Score
        # Based on "difficulty" of prediction
        # Extremes reduce confidence
        temp = context.get('temperature_c', 25.0)
        temp_uncert = min(1.0, abs(temp - 25.0) / 50.0) # 0 at 25, 1 at 75 or -25
        
        # Grade uncertainty
        max_grade = 0.0
        if simulation_result.trajectory:
            grades = [abs(s.grade_percent) for s in simulation_result.trajectory]
            max_grade = max(grades) if grades else 0.0
        grade_uncert = min(1.0, max_grade / 15.0) # 0 at 0%, 1 at 15%
        
        # Velocity variability
        velocities = [s.velocity_mps for s in simulation_result.trajectory]
        if velocities:
            cv = np.std(velocities) / (np.mean(velocities) + 0.1)
            vel_uncert = min(1.0, cv)
        else:
            vel_uncert = 1.0
            
        avg_uncert = (temp_uncert + grade_uncert + vel_uncert) / 3.0
        uncertainty_score = 1.0 - avg_uncert
        
        # 3. Historical Accuracy
        # Placeholder: Assume moderate-high unless flagged
        historical_score = 0.85
        
        # Weighted Average
        overall = (0.4 * physics_score) + (0.3 * uncertainty_score) + (0.3 * historical_score)
        
        # Interpretation
        if overall >= 0.9: interpretation = "HIGH CONFIDENCE"
        elif overall >= 0.75: interpretation = "MEDIUM CONFIDENCE"
        else: interpretation = "LOW CONFIDENCE"
        
        # Recommendations
        recs = []
        if physics_score < 0.85: recs.append("Physics validation warnings detected.")
        if uncertainty_score < 0.6: recs.append("High uncertainty due to extreme conditions.")
        if not recs: recs.append("Simulation parameters within normal range.")
        
        return ConfidenceScore(
            overall=overall,
            physics_validation=physics_score,
            uncertainty=uncertainty_score,
            historical_accuracy=historical_score,
            interpretation=interpretation,
            recommendations=recs
        )
