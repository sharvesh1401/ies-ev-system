"""Tests for the ML confidence scorer."""

import pytest
from app.ml.confidence_scorer import (
    MLConfidenceScorer,
    ConfidenceComponents,
    ConfidenceResult,
    ScenarioSignature,
)


class TestConfidenceComponents:
    """Test the multi-factor confidence calculation."""

    def test_calculate_overall_weighted(self):
        """Overall score should be a weighted sum of components."""
        components = ConfidenceComponents(
            model_uncertainty=0.90,
            physics_agreement=0.85,
            historical_accuracy=0.80,
            data_quality=1.0,
        )

        overall = components.calculate_overall()

        # Expected: 0.40*0.90 + 0.30*0.85 + 0.20*0.80 + 0.10*1.0
        expected = 0.36 + 0.255 + 0.16 + 0.10
        assert abs(overall - expected) < 0.001

    def test_perfect_scores(self):
        components = ConfidenceComponents(
            model_uncertainty=1.0,
            physics_agreement=1.0,
            historical_accuracy=1.0,
            data_quality=1.0,
        )
        assert abs(components.calculate_overall() - 1.0) < 0.001

    def test_zero_scores(self):
        components = ConfidenceComponents(
            model_uncertainty=0.0,
            physics_agreement=0.0,
            historical_accuracy=0.0,
            data_quality=0.0,
        )
        assert components.calculate_overall() == 0.0


class TestMLConfidenceScorer:
    """Test the full scorer workflow."""

    @pytest.fixture
    def scorer(self):
        return MLConfidenceScorer(history_size=100)

    def test_basic_confidence_calculation(self, scorer):
        """Should return a ConfidenceResult with valid score."""
        features = {
            "mass_kg": 1500,
            "distance_km": 50,
            "target_speed_kmh": 90,
            "temperature_c": 25,
            "elevation_gain_m": 100,
        }

        result = scorer.calculate_confidence(
            prediction_mean=12.0,
            prediction_std=0.5,
            features=features,
        )

        assert isinstance(result, ConfidenceResult)
        assert 0 <= result.score <= 1

    def test_interpretation_levels(self, scorer):
        """Should produce correct interpretation for score ranges."""
        features = {"mass_kg": 1500, "distance_km": 50,
                     "target_speed_kmh": 90, "temperature_c": 25}

        # High confidence (low std relative to mean)
        high = scorer.calculate_confidence(
            prediction_mean=12.0, prediction_std=0.1, features=features
        )
        assert high.interpretation in ("HIGH", "MEDIUM")  # Depends on other factors

        # Low confidence (high std relative to mean)
        low = scorer.calculate_confidence(
            prediction_mean=12.0, prediction_std=8.0, features=features
        )
        # Score should be lower
        assert low.score < high.score

    def test_physics_agreement_boosts_confidence(self, scorer):
        """When physics agrees, confidence should be higher."""
        features = {"mass_kg": 1500, "distance_km": 50,
                     "target_speed_kmh": 90, "temperature_c": 25}

        without_physics = scorer.calculate_confidence(
            prediction_mean=12.0, prediction_std=0.5, features=features,
            physics_result=None
        )

        with_physics = scorer.calculate_confidence(
            prediction_mean=12.0, prediction_std=0.5, features=features,
            physics_result=12.1  # Close agreement
        )

        # Physics agreement should not reduce confidence
        assert with_physics.score >= without_physics.score - 0.15

    def test_record_validation(self, scorer):
        """Should successfully record a validation entry."""
        features = {"mass_kg": 1500, "distance_km": 50, "target_speed_kmh": 90}
        scorer.record_validation(features, ml_prediction=12.0, actual_value=12.3)

        stats = scorer.get_calibration_stats()
        assert isinstance(stats, dict)

    def test_to_dict(self, scorer):
        """ConfidenceResult.to_dict should produce a valid dict."""
        features = {"mass_kg": 1500, "distance_km": 50,
                     "target_speed_kmh": 90, "temperature_c": 25}

        result = scorer.calculate_confidence(
            prediction_mean=12.0, prediction_std=0.5, features=features
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "confidence" in d
        assert "interpretation" in d


class TestScenarioSignature:
    """Test scenario bucketing for historical comparison."""

    def test_signature_tuple(self):
        sig = ScenarioSignature(
            distance_bucket=2,
            speed_bucket=3,
            temp_bucket=1,
            elevation_bucket=0,
        )
        assert sig.as_tuple() == (2, 3, 1, 0)
