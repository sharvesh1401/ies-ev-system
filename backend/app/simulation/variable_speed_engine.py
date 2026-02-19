"""
Variable Speed Simulation Engine.

Combines driving profile generators with the physics simulation engine
to produce realistic variable-speed simulation results.

Wraps IntegratedSimulationEngine.simulate_with_profile() with
convenient profile selection and result formatting.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.simulation.integrated_engine import IntegratedSimulationEngine
from app.simulation.driving_profiles import (
    CityDrivingProfile,
    HighwayDrivingProfile,
    MixedDrivingProfile,
    DriverStyle,
    SpeedPoint,
)
from app.simulation.schemas import (
    VehicleParameters,
    RouteParameters,
    EnvironmentParameters,
)


PROFILE_MAP = {
    "city": CityDrivingProfile,
    "highway": HighwayDrivingProfile,
    "mixed": MixedDrivingProfile,
}

DRIVER_STYLE_MAP = {
    "aggressive": DriverStyle.AGGRESSIVE,
    "moderate": DriverStyle.MODERATE,
    "eco": DriverStyle.ECO,
}


@dataclass
class DrivingProfileResult:
    """Result of generating a driving profile."""

    profile_type: str
    driver_style: str
    speed_points: List[Dict[str, Any]]
    total_distance_m: float
    duration_s: float
    num_stops: int
    avg_speed_mps: float
    max_speed_mps: float
    num_acceleration_events: int
    num_deceleration_events: int


@dataclass
class VariableSpeedResult:
    """Result of a variable-speed simulation."""

    # Core results
    total_energy_kwh: float
    duration_minutes: float
    final_soc: float
    avg_speed_kmh: float

    # Profile info
    profile_type: str
    driver_style: str
    num_stops: int
    max_speed_kmh: float

    # Energy breakdown
    energy_breakdown: Dict[str, float]

    # Full trajectory (optional, can be large)
    trajectory_summary: Dict[str, Any] = field(default_factory=dict)


def generate_driving_profile(
    distance_km: float,
    profile_type: str = "city",
    driver_style: str = "moderate",
    base_speed_kmh: Optional[float] = None,
    seed: Optional[int] = None,
) -> DrivingProfileResult:
    """
    Generate a realistic driving profile.

    Args:
        distance_km: Total route distance in km.
        profile_type: One of 'city', 'highway', 'mixed'.
        driver_style: One of 'aggressive', 'moderate', 'eco'.
        base_speed_kmh: Optional base speed override (km/h).
        seed: Random seed for reproducibility.

    Returns:
        DrivingProfileResult with speed points and statistics.
    """
    profile_cls = PROFILE_MAP.get(profile_type)
    if profile_cls is None:
        raise ValueError(
            f"Unknown profile type: {profile_type}. "
            f"Choose from: {list(PROFILE_MAP.keys())}"
        )

    style = DRIVER_STYLE_MAP.get(driver_style)
    if style is None:
        raise ValueError(
            f"Unknown driver style: {driver_style}. "
            f"Choose from: {list(DRIVER_STYLE_MAP.keys())}"
        )

    profile = profile_cls(seed=seed)

    distance_m = distance_km * 1000.0
    base_speed_mps = (base_speed_kmh / 3.6) if base_speed_kmh else None

    speed_points: List[SpeedPoint] = profile.generate_speed_profile(
        distance_m=distance_m,
        driver_style=style,
        base_speed_mps=base_speed_mps,
    )

    # Compute stats
    num_stops = sum(1 for sp in speed_points if sp.is_stop)
    speeds = [sp.target_speed_mps for sp in speed_points]
    max_speed = max(speeds) if speeds else 0.0
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    duration_s = speed_points[-1].time_s if speed_points else 0.0

    # Count acceleration / deceleration events
    accel_events = 0
    decel_events = 0
    for i in range(1, len(speed_points)):
        delta = speed_points[i].target_speed_mps - speed_points[i - 1].target_speed_mps
        if delta > 0.5:
            accel_events += 1
        elif delta < -0.5:
            decel_events += 1

    # Serialise speed points
    points_dicts = [
        {
            "time_s": round(sp.time_s, 2),
            "speed_mps": round(sp.target_speed_mps, 3),
            "speed_kmh": round(sp.target_speed_mps * 3.6, 1),
            "distance_m": round(sp.distance_m, 1),
            "is_stop": sp.is_stop,
            "event_type": sp.event_type,
        }
        for sp in speed_points
    ]

    return DrivingProfileResult(
        profile_type=profile_type,
        driver_style=driver_style,
        speed_points=points_dicts,
        total_distance_m=distance_m,
        duration_s=duration_s,
        num_stops=num_stops,
        avg_speed_mps=round(avg_speed, 3),
        max_speed_mps=round(max_speed, 3),
        num_acceleration_events=accel_events,
        num_deceleration_events=decel_events,
    )


def simulate_variable_speed(
    vehicle_params: VehicleParameters,
    route_params: RouteParameters,
    environment_params: EnvironmentParameters,
    profile_type: str = "city",
    driver_style: str = "moderate",
    base_speed_kmh: Optional[float] = None,
    seed: Optional[int] = None,
    dt: float = 1.0,
) -> VariableSpeedResult:
    """
    Run a full variable-speed simulation.

    Generates a driving profile and runs it through the physics engine.

    Args:
        vehicle_params: Vehicle specifications.
        route_params: Route details (distance, etc.).
        environment_params: Environmental conditions.
        profile_type: 'city', 'highway', or 'mixed'.
        driver_style: 'aggressive', 'moderate', or 'eco'.
        base_speed_kmh: Override base speed (km/h).
        seed: Random seed for profile reproducibility.
        dt: Simulation timestep in seconds.

    Returns:
        VariableSpeedResult with energy, duration, SOC, and breakdown.
    """
    # 1. Generate profile
    profile_result = generate_driving_profile(
        distance_km=route_params.distance_km,
        profile_type=profile_type,
        driver_style=driver_style,
        base_speed_kmh=base_speed_kmh,
        seed=seed,
    )

    # 2. Convert back to SpeedPoint objects for the engine
    speed_points = [
        SpeedPoint(
            time_s=p["time_s"],
            target_speed_mps=p["speed_mps"],
            distance_m=p["distance_m"],
            is_stop=p["is_stop"],
            event_type=p["event_type"],
        )
        for p in profile_result.speed_points
    ]

    # 3. Run physics simulation with profile
    engine = IntegratedSimulationEngine()
    sim_result = engine.simulate_with_profile(
        vehicle_params=vehicle_params,
        route_params=route_params,
        environment_params=environment_params,
        speed_profile=speed_points,
        dt=dt,
    )

    # 4. Extract results
    trajectory = sim_result.trajectory
    final_step = trajectory[-1] if trajectory else None

    total_energy = sim_result.total_energy_kwh
    duration_min = (final_step.time_s / 60.0) if final_step else 0.0
    final_soc = sim_result.final_soc_percent
    avg_speed_kmh = sim_result.avg_velocity_mps * 3.6

    # Energy breakdown
    breakdown = sim_result.energy_breakdown
    energy_breakdown = {
        "kinetic_kwh": breakdown.kinetic_kwh,
        "potential_kwh": breakdown.potential_kwh,
        "aerodynamic_kwh": breakdown.aerodynamic_kwh,
        "rolling_kwh": breakdown.rolling_kwh,
        "auxiliary_kwh": breakdown.auxiliary_kwh,
        "total_kwh": breakdown.total_kwh,
    }

    # Trajectory summary (avoid sending huge payload)
    trajectory_summary = {
        "num_steps": len(trajectory),
        "duration_s": final_step.time_s if final_step else 0.0,
        "max_velocity_mps": max(s.velocity_mps for s in trajectory) if trajectory else 0.0,
        "min_soc_percent": min(s.soc_percent for s in trajectory) if trajectory else 0.0,
    }

    return VariableSpeedResult(
        total_energy_kwh=round(total_energy, 4),
        duration_minutes=round(duration_min, 2),
        final_soc=round(final_soc, 2),
        avg_speed_kmh=round(avg_speed_kmh, 2),
        profile_type=profile_type,
        driver_style=driver_style,
        num_stops=profile_result.num_stops,
        max_speed_kmh=round(profile_result.max_speed_mps * 3.6, 1),
        energy_breakdown=energy_breakdown,
        trajectory_summary=trajectory_summary,
    )
