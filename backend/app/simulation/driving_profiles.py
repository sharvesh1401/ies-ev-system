"""
Driving Profiles for Realistic Variable-Speed Simulation.

Generates realistic speed profiles for city, highway, and mixed driving
with support for different driver styles (aggressive, moderate, eco).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import math
import random


class DriverStyle(Enum):
    """Driver behavior styles."""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    ECO = "eco"


@dataclass
class SpeedPoint:
    """A point in the speed profile."""
    time_s: float
    target_speed_mps: float
    distance_m: float = 0.0
    is_stop: bool = False
    event_type: str = ""  # "acceleration", "cruise", "deceleration", "stop"


@dataclass
class DriverParameters:
    """Parameters defining driver behavior."""
    max_acceleration_mps2: float  # Maximum acceleration
    max_deceleration_mps2: float  # Maximum braking deceleration
    speed_offset_factor: float     # Multiplier for target speed (>1 = faster)
    stop_duration_factor: float    # Multiplier for stop durations
    following_distance_s: float    # Time gap to vehicle ahead
    reaction_time_s: float         # Delay before responding


# Predefined driver parameter sets
DRIVER_PARAMS = {
    DriverStyle.AGGRESSIVE: DriverParameters(
        max_acceleration_mps2=3.0,
        max_deceleration_mps2=4.0,
        speed_offset_factor=1.10,  # 10% over target
        stop_duration_factor=0.7,  # Short stops
        following_distance_s=1.0,
        reaction_time_s=0.3
    ),
    DriverStyle.MODERATE: DriverParameters(
        max_acceleration_mps2=2.0,
        max_deceleration_mps2=3.0,
        speed_offset_factor=1.0,
        stop_duration_factor=1.0,
        following_distance_s=2.0,
        reaction_time_s=0.5
    ),
    DriverStyle.ECO: DriverParameters(
        max_acceleration_mps2=1.2,
        max_deceleration_mps2=2.0,
        speed_offset_factor=0.90,  # 10% under target
        stop_duration_factor=1.3,  # Longer stops OK
        following_distance_s=3.0,
        reaction_time_s=0.7
    )
}


class DrivingProfile(ABC):
    """Base class for driving profiles."""
    
    def __init__(self, seed: int = None):
        """
        Initialize profile generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
    
    @abstractmethod
    def generate_speed_profile(
        self,
        distance_m: float,
        driver_style: DriverStyle = DriverStyle.MODERATE,
        base_speed_mps: float = None
    ) -> List[SpeedPoint]:
        """
        Generate a list of speed points for the profile.
        
        Args:
            distance_m: Total route distance in meters
            driver_style: Driver behavior style
            base_speed_mps: Base target speed (profile-specific default if None)
            
        Returns:
            List of SpeedPoint objects
        """
        pass
    
    def _get_driver_params(self, style: DriverStyle) -> DriverParameters:
        """Get driver parameters for the given style."""
        return DRIVER_PARAMS[style]
    
    def _calculate_acceleration_distance(
        self,
        v_start: float,
        v_end: float,
        acceleration: float
    ) -> Tuple[float, float]:
        """
        Calculate distance and time for acceleration/deceleration.
        
        Returns:
            (distance_m, duration_s)
        """
        if abs(v_end - v_start) < 0.01:
            return 0.0, 0.0
            
        # v² = v₀² + 2as → s = (v² - v₀²) / (2a)
        # t = (v - v₀) / a
        
        acc = abs(acceleration)
        if v_end > v_start:
            # Acceleration
            distance = (v_end**2 - v_start**2) / (2 * acc)
            duration = (v_end - v_start) / acc
        else:
            # Deceleration
            distance = (v_start**2 - v_end**2) / (2 * acc)
            duration = (v_start - v_end) / acc
            
        return distance, duration
    
    def _create_acceleration_segment(
        self,
        t_start: float,
        d_start: float,
        v_start: float,
        v_end: float,
        acceleration: float,
        n_points: int = 5
    ) -> List[SpeedPoint]:
        """Create interpolated points for acceleration/deceleration."""
        points = []
        
        if n_points < 2:
            n_points = 2
            
        duration_total, distance_total = self._calculate_acceleration_distance(
            v_start, v_end, acceleration
        )
        
        for i in range(n_points):
            frac = i / (n_points - 1)
            t = t_start + frac * duration_total
            
            # v = v₀ + at
            if v_end > v_start:
                v = v_start + acceleration * frac * duration_total
            else:
                v = v_start - acceleration * frac * duration_total
            
            # s = v₀t + 0.5at²
            if v_end > v_start:
                d = d_start + v_start * (frac * duration_total) + \
                    0.5 * acceleration * (frac * duration_total)**2
            else:
                d = d_start + v_start * (frac * duration_total) - \
                    0.5 * acceleration * (frac * duration_total)**2
            
            event = "acceleration" if v_end > v_start else "deceleration"
            
            points.append(SpeedPoint(
                time_s=t,
                target_speed_mps=max(0, v),
                distance_m=d,
                event_type=event
            ))
        
        return points


class CityDrivingProfile(DrivingProfile):
    """
    Urban driving profile with frequent stops.
    
    Characteristics:
    - Speed range: 0-60 km/h
    - Frequent stops (10-20 per 30 minutes)
    - Stop-and-go traffic patterns
    - Traffic lights and intersections
    """
    
    DEFAULT_SPEED_MPS = 40 / 3.6  # 40 km/h
    MIN_STOP_INTERVAL_M = 200    # Minimum distance between stops
    MAX_STOP_INTERVAL_M = 800    # Maximum distance between stops
    MIN_STOP_DURATION_S = 5      # Minimum stop duration
    MAX_STOP_DURATION_S = 45     # Maximum stop duration
    
    def generate_speed_profile(
        self,
        distance_m: float,
        driver_style: DriverStyle = DriverStyle.MODERATE,
        base_speed_mps: float = None
    ) -> List[SpeedPoint]:
        """Generate city driving profile with stops."""
        params = self._get_driver_params(driver_style)
        target_speed = (base_speed_mps or self.DEFAULT_SPEED_MPS) * params.speed_offset_factor
        
        # Clamp to city speed limits
        target_speed = min(target_speed, 60 / 3.6)  # Max 60 km/h
        
        points = []
        current_t = 0.0
        current_d = 0.0
        current_v = 0.0
        
        # Start from stop
        points.append(SpeedPoint(
            time_s=0.0,
            target_speed_mps=0.0,
            distance_m=0.0,
            is_stop=True,
            event_type="stop"
        ))
        
        while current_d < distance_m:
            # Determine next stop location
            next_stop_dist = current_d + self.rng.uniform(
                self.MIN_STOP_INTERVAL_M,
                self.MAX_STOP_INTERVAL_M
            )
            
            if next_stop_dist > distance_m:
                next_stop_dist = distance_m
            
            segment_dist = next_stop_dist - current_d
            
            # Safety check - ensure we make progress
            if segment_dist < 10:  # Minimum 10m progress
                current_d = next_stop_dist
                continue
            
            # Phase 1: Acceleration from current speed to target
            if current_v < target_speed:
                accel_dist, accel_time = self._calculate_acceleration_distance(
                    current_v, target_speed, params.max_acceleration_mps2
                )
                
                if accel_dist < segment_dist * 0.4:  # Limit to 40% of segment
                    accel_points = self._create_acceleration_segment(
                        current_t, current_d, current_v, target_speed,
                        params.max_acceleration_mps2, n_points=3
                    )
                    points.extend(accel_points)
                    current_t += accel_time
                    current_d += accel_dist
                    current_v = target_speed
            
            # Recalculate remaining distance
            remaining = next_stop_dist - current_d
            
            # Phase 2: Cruise at target speed
            cruise_speed = current_v if current_v > 0 else target_speed * 0.5  # Use some speed
            decel_dist, _ = self._calculate_acceleration_distance(
                cruise_speed, 0, params.max_deceleration_mps2
            )
            
            cruise_dist = remaining - decel_dist
            if cruise_dist > 10:  # Only cruise if > 10m
                cruise_time = cruise_dist / cruise_speed
                
                # Add cruise points
                n_cruise_points = max(2, int(cruise_dist / 100))  # Point every ~100m
                for i in range(n_cruise_points):
                    frac = i / (n_cruise_points - 1) if n_cruise_points > 1 else 1
                    points.append(SpeedPoint(
                        time_s=current_t + frac * cruise_time,
                        target_speed_mps=cruise_speed,
                        distance_m=current_d + frac * cruise_dist,
                        event_type="cruise"
                    ))
                
                current_t += cruise_time
                current_d += cruise_dist
                current_v = cruise_speed
            else:
                # Not enough room to cruise - just move forward
                if remaining > 0 and cruise_speed > 0:
                    move_time = remaining / cruise_speed
                    current_t += move_time
                    current_d += remaining
            
            # Phase 3: Deceleration to stop
            if current_v > 0:
                decel_points = self._create_acceleration_segment(
                    current_t, current_d, current_v, 0,
                    params.max_deceleration_mps2, n_points=3
                )
                points.extend(decel_points)
                
                decel_dist, decel_time = self._calculate_acceleration_distance(
                    current_v, 0, params.max_deceleration_mps2
                )
                current_t += decel_time
                current_d += decel_dist
                current_v = 0
            
            # Phase 4: Stop
            if current_d < distance_m:
                stop_duration = self.rng.uniform(
                    self.MIN_STOP_DURATION_S,
                    self.MAX_STOP_DURATION_S
                ) * params.stop_duration_factor
                
                points.append(SpeedPoint(
                    time_s=current_t + stop_duration,
                    target_speed_mps=0.0,
                    distance_m=current_d,
                    is_stop=True,
                    event_type="stop"
                ))
                
                current_t += stop_duration
        
        return points


class HighwayDrivingProfile(DrivingProfile):
    """
    Highway driving profile with steady speeds.
    
    Characteristics:
    - Speed range: 90-130 km/h
    - Minimal stops
    - Occasional speed variations (traffic, overtaking)
    - Long cruise segments
    """
    
    DEFAULT_SPEED_MPS = 110 / 3.6  # 110 km/h
    SPEED_VARIATION_MPS = 5 / 3.6  # ±5 km/h variation
    
    def generate_speed_profile(
        self,
        distance_m: float,
        driver_style: DriverStyle = DriverStyle.MODERATE,
        base_speed_mps: float = None
    ) -> List[SpeedPoint]:
        """Generate highway driving profile."""
        params = self._get_driver_params(driver_style)
        target_speed = (base_speed_mps or self.DEFAULT_SPEED_MPS) * params.speed_offset_factor
        
        # Clamp to highway limits
        target_speed = max(target_speed, 80 / 3.6)   # Min 80 km/h
        target_speed = min(target_speed, 130 / 3.6)  # Max 130 km/h
        
        points = []
        current_t = 0.0
        current_d = 0.0
        current_v = 0.0
        
        # Start: On-ramp acceleration
        points.append(SpeedPoint(
            time_s=0.0,
            target_speed_mps=0.0,
            distance_m=0.0,
            event_type="start"
        ))
        
        # Accelerate to highway speed
        accel_dist, accel_time = self._calculate_acceleration_distance(
            0, target_speed, params.max_acceleration_mps2
        )
        
        accel_points = self._create_acceleration_segment(
            0, 0, 0, target_speed, params.max_acceleration_mps2, n_points=5
        )
        points.extend(accel_points)
        
        current_t = accel_time
        current_d = accel_dist
        current_v = target_speed
        
        # Main highway segment with variations
        segment_length = 5000  # ~5km segments
        
        while current_d < distance_m - 500:  # Leave room for exit
            segment_end = min(current_d + segment_length, distance_m - 500)
            segment_dist = segment_end - current_d
            
            # Determine segment speed (with slight variation)
            segment_speed = target_speed + self.rng.uniform(
                -self.SPEED_VARIATION_MPS,
                self.SPEED_VARIATION_MPS
            )
            
            # Occasional traffic slowdown (10% chance)
            if self.rng.random() < 0.10:
                segment_speed = target_speed * 0.7  # Slow to 70%
            
            # Transition to new speed if different
            if abs(segment_speed - current_v) > 1:  # More than 1 m/s difference
                if segment_speed > current_v:
                    trans_points = self._create_acceleration_segment(
                        current_t, current_d, current_v, segment_speed,
                        params.max_acceleration_mps2 * 0.5, n_points=2
                    )
                else:
                    trans_points = self._create_acceleration_segment(
                        current_t, current_d, current_v, segment_speed,
                        params.max_deceleration_mps2 * 0.3, n_points=2
                    )
                points.extend(trans_points)
                
                trans_dist, trans_time = self._calculate_acceleration_distance(
                    current_v, segment_speed,
                    params.max_acceleration_mps2 * 0.5 if segment_speed > current_v 
                    else params.max_deceleration_mps2 * 0.3
                )
                current_t += trans_time
                current_d += trans_dist
                current_v = segment_speed
            
            # Cruise through segment
            remaining = segment_end - current_d
            if remaining > 0:
                cruise_time = remaining / current_v
                
                # Add a few cruise points
                n_points = max(2, int(remaining / 1000))  # Point every ~1km
                for i in range(n_points):
                    frac = i / (n_points - 1) if n_points > 1 else 1
                    points.append(SpeedPoint(
                        time_s=current_t + frac * cruise_time,
                        target_speed_mps=current_v,
                        distance_m=current_d + frac * remaining,
                        event_type="cruise"
                    ))
                
                current_t += cruise_time
                current_d += remaining
        
        # Exit: Decelerate to stop
        if current_v > 0:
            decel_points = self._create_acceleration_segment(
                current_t, current_d, current_v, 0,
                params.max_deceleration_mps2, n_points=5
            )
            points.extend(decel_points)
            
            decel_dist, decel_time = self._calculate_acceleration_distance(
                current_v, 0, params.max_deceleration_mps2
            )
            current_t += decel_time
            current_d += decel_dist
        
        # Final stop
        points.append(SpeedPoint(
            time_s=current_t,
            target_speed_mps=0.0,
            distance_m=current_d,
            is_stop=True,
            event_type="stop"
        ))
        
        return points


class MixedDrivingProfile(DrivingProfile):
    """
    Mixed driving profile combining city and highway.
    
    Characteristics:
    - Starts with city driving
    - Transitions to highway
    - Returns to city driving
    - Configurable split ratios
    """
    
    def __init__(
        self,
        seed: int = None,
        city_ratio: float = 0.4,  # 40% city, 60% highway
    ):
        super().__init__(seed)
        self.city_ratio = city_ratio
        self.city_profile = CityDrivingProfile(seed)
        self.highway_profile = HighwayDrivingProfile(seed)
    
    def generate_speed_profile(
        self,
        distance_m: float,
        driver_style: DriverStyle = DriverStyle.MODERATE,
        base_speed_mps: float = None
    ) -> List[SpeedPoint]:
        """Generate mixed driving profile."""
        # Split: city → highway → city
        city_dist = distance_m * self.city_ratio
        highway_dist = distance_m * (1 - self.city_ratio)
        
        # First city segment (half of city distance)
        city1_dist = city_dist / 2
        city1_points = self.city_profile.generate_speed_profile(
            city1_dist, driver_style, 40 / 3.6
        )
        
        # Get end state of city1
        last_time = city1_points[-1].time_s if city1_points else 0
        last_dist = city1_points[-1].distance_m if city1_points else 0
        
        # Highway segment
        highway_points = self.highway_profile.generate_speed_profile(
            highway_dist, driver_style, base_speed_mps
        )
        
        # Offset highway points
        for p in highway_points:
            p.time_s += last_time
            p.distance_m += last_dist
        
        if highway_points:
            last_time = highway_points[-1].time_s
            last_dist = highway_points[-1].distance_m
        
        # Second city segment
        city2_dist = city_dist / 2
        city2_points = self.city_profile.generate_speed_profile(
            city2_dist, driver_style, 40 / 3.6
        )
        
        # Offset city2 points
        for p in city2_points:
            p.time_s += last_time
            p.distance_m += last_dist
        
        # Combine all segments
        all_points = city1_points + highway_points[1:] + city2_points[1:]  # Skip duplicate start points
        
        return all_points


def create_profile(
    profile_type: str,
    seed: int = None
) -> DrivingProfile:
    """
    Factory function to create driving profiles.
    
    Args:
        profile_type: "city", "highway", or "mixed"
        seed: Random seed
        
    Returns:
        DrivingProfile instance
    """
    profile_map = {
        "city": CityDrivingProfile,
        "highway": HighwayDrivingProfile,
        "mixed": MixedDrivingProfile
    }
    
    if profile_type.lower() not in profile_map:
        raise ValueError(f"Unknown profile type: {profile_type}. Use 'city', 'highway', or 'mixed'")
    
    return profile_map[profile_type.lower()](seed)


def speed_profile_to_dict(points: List[SpeedPoint]) -> dict:
    """Convert speed profile to dictionary for JSON serialization."""
    return {
        "points": [
            {
                "time_s": p.time_s,
                "target_speed_mps": p.target_speed_mps,
                "target_speed_kmh": p.target_speed_mps * 3.6,
                "distance_m": p.distance_m,
                "is_stop": p.is_stop,
                "event_type": p.event_type
            }
            for p in points
        ],
        "summary": {
            "total_time_s": points[-1].time_s if points else 0,
            "total_distance_m": points[-1].distance_m if points else 0,
            "num_stops": sum(1 for p in points if p.is_stop),
            "avg_speed_mps": (
                points[-1].distance_m / points[-1].time_s 
                if points and points[-1].time_s > 0 else 0
            )
        }
    }
