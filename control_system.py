"""
Control System - Generates realistic control inputs for flight simulation.
Includes predefined maneuvers and random control variations.
"""

import numpy as np
from typing import Dict, Callable, Optional
from dataclasses import dataclass


@dataclass
class ControlInputs:
    """Represents the control surface positions and throttle setting."""
    elevator: float    # Elevator deflection (-1 to 1, negative = nose down)
    aileron: float     # Aileron deflection (-1 to 1, negative = left roll)
    rudder: float      # Rudder deflection (-1 to 1, negative = left yaw)
    throttle: float    # Throttle setting (0 to 1)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for aerodynamic calculations."""
        return {
            'elevator': self.elevator,
            'aileron': self.aileron,
            'rudder': self.rudder,
            'throttle': self.throttle
        }


class ControlSystem:
    """
    Generates control inputs for various flight maneuvers and scenarios.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Control limits (realistic for small aircraft)
        self.max_elevator = 0.8    # ±0.8 radians (≈ ±46°)
        self.max_aileron = 0.7     # ±0.7 radians (≈ ±40°)
        self.max_rudder = 0.6      # ±0.6 radians (≈ ±34°)
        
        # Default control positions
        self.trim_elevator = 0.05  # Slight nose-up trim
        self.trim_aileron = 0.0
        self.trim_rudder = 0.0
        self.cruise_throttle = 0.7
    
    def straight_and_level(self, time: float) -> ControlInputs:
        """
        Straight and level flight with small random variations.
        """
        # Small random variations around trim position
        noise_scale = 0.02
        
        return ControlInputs(
            elevator=self.trim_elevator + np.random.normal(0, noise_scale),
            aileron=self.trim_aileron + np.random.normal(0, noise_scale * 0.5),
            rudder=self.trim_rudder + np.random.normal(0, noise_scale * 0.3),
            throttle=self.cruise_throttle + np.random.normal(0, 0.01)
        )
    
    def coordinated_turn(self, time: float, turn_rate: float = 0.1) -> ControlInputs:
        """
        Coordinated turn maneuver.
        
        Args:
            time: Current simulation time
            turn_rate: Turn rate in rad/s (positive = right turn)
        """
        # Banking angle for coordinated turn (approximately)
        bank_angle = np.arctan(turn_rate * 50 / 9.81)  # Assuming 50 m/s airspeed
        
        aileron_input = np.clip(bank_angle * 2.0, -self.max_aileron, self.max_aileron)
        rudder_input = turn_rate * 0.5  # Coordinated rudder
        
        return ControlInputs(
            elevator=self.trim_elevator + 0.1 * np.sin(time * turn_rate),  # Slight pitch up in turn
            aileron=aileron_input,
            rudder=np.clip(rudder_input, -self.max_rudder, self.max_rudder),
            throttle=self.cruise_throttle
        )
    
    def climb_maneuver(self, time: float, climb_rate: float = 5.0) -> ControlInputs:
        """
        Climbing maneuver.
        
        Args:
            time: Current simulation time
            climb_rate: Desired climb rate in m/s
        """
        # Elevator input for climb
        elevator_input = self.trim_elevator + climb_rate * 0.02
        
        return ControlInputs(
            elevator=np.clip(elevator_input, -self.max_elevator, self.max_elevator),
            aileron=self.trim_aileron + np.random.normal(0, 0.01),
            rudder=self.trim_rudder + np.random.normal(0, 0.01),
            throttle=min(0.9, self.cruise_throttle + 0.2)  # Increase power for climb
        )
    
    def descent_maneuver(self, time: float, descent_rate: float = -3.0) -> ControlInputs:
        """
        Descending maneuver.
        
        Args:
            time: Current simulation time
            descent_rate: Desired descent rate in m/s (negative)
        """
        # Elevator input for descent
        elevator_input = self.trim_elevator + descent_rate * 0.02
        
        return ControlInputs(
            elevator=np.clip(elevator_input, -self.max_elevator, self.max_elevator),
            aileron=self.trim_aileron + np.random.normal(0, 0.01),
            rudder=self.trim_rudder + np.random.normal(0, 0.01),
            throttle=max(0.3, self.cruise_throttle - 0.2)  # Reduce power for descent
        )
    
    def dutch_roll(self, time: float, frequency: float = 0.5, amplitude: float = 0.3) -> ControlInputs:
        """
        Dutch roll oscillation (combined yaw and roll motion).
        
        Args:
            time: Current simulation time
            frequency: Oscillation frequency in Hz
            amplitude: Oscillation amplitude (0 to 1)
        """
        omega = 2 * np.pi * frequency
        
        return ControlInputs(
            elevator=self.trim_elevator,
            aileron=amplitude * 0.2 * np.sin(omega * time),
            rudder=amplitude * 0.3 * np.sin(omega * time + np.pi/4),  # Phase shift
            throttle=self.cruise_throttle
        )
    
    def takeoff_sequence(self, time: float) -> ControlInputs:
        """
        Takeoff sequence: full throttle, gradual elevator input.
        """
        if time < 5.0:
            # Initial acceleration phase
            elevator_input = self.trim_elevator
            throttle_input = 1.0
        elif time < 10.0:
            # Rotation phase
            elevator_input = self.trim_elevator + 0.3 * (time - 5.0) / 5.0
            throttle_input = 1.0
        else:
            # Initial climb
            elevator_input = self.trim_elevator + 0.3
            throttle_input = 0.9
        
        return ControlInputs(
            elevator=np.clip(elevator_input, -self.max_elevator, self.max_elevator),
            aileron=self.trim_aileron + np.random.normal(0, 0.02),
            rudder=self.trim_rudder + np.random.normal(0, 0.02),
            throttle=throttle_input
        )
    
    def landing_approach(self, time: float) -> ControlInputs:
        """
        Landing approach: reduced throttle, gradual descent.
        """
        # Gradual flare near the end
        flare_time = 30.0  # Start flare 30 seconds into approach
        
        if time < flare_time:
            # Approach phase
            elevator_input = self.trim_elevator - 0.1  # Nose down for descent
            throttle_input = 0.4
        else:
            # Flare phase
            flare_progress = (time - flare_time) / 10.0  # 10-second flare
            elevator_input = self.trim_elevator - 0.1 + 0.4 * min(1.0, flare_progress)
            throttle_input = 0.4 - 0.3 * min(1.0, flare_progress)
        
        return ControlInputs(
            elevator=np.clip(elevator_input, -self.max_elevator, self.max_elevator),
            aileron=self.trim_aileron + np.random.normal(0, 0.03),  # More turbulence near ground
            rudder=self.trim_rudder + np.random.normal(0, 0.02),
            throttle=max(0.0, throttle_input)
        )
    
    def turbulence(self, time: float, intensity: float = 0.1) -> ControlInputs:
        """
        Random turbulence-like control inputs.
        
        Args:
            time: Current simulation time
            intensity: Turbulence intensity (0 to 1)
        """
        # Use time-correlated noise for more realistic turbulence
        freq1, freq2, freq3 = 0.1, 0.3, 0.7  # Different frequency components
        
        elevator_turb = intensity * (
            0.3 * np.sin(2 * np.pi * freq1 * time + np.random.random()) +
            0.2 * np.sin(2 * np.pi * freq2 * time + np.random.random()) +
            0.1 * np.sin(2 * np.pi * freq3 * time + np.random.random())
        )
        
        aileron_turb = intensity * (
            0.2 * np.sin(2 * np.pi * freq1 * time + np.random.random() + 1) +
            0.3 * np.sin(2 * np.pi * freq2 * time + np.random.random() + 1) +
            0.1 * np.sin(2 * np.pi * freq3 * time + np.random.random() + 1)
        )
        
        rudder_turb = intensity * (
            0.1 * np.sin(2 * np.pi * freq1 * time + np.random.random() + 2) +
            0.1 * np.sin(2 * np.pi * freq2 * time + np.random.random() + 2) +
            0.2 * np.sin(2 * np.pi * freq3 * time + np.random.random() + 2)
        )
        
        return ControlInputs(
            elevator=np.clip(self.trim_elevator + elevator_turb, -self.max_elevator, self.max_elevator),
            aileron=np.clip(self.trim_aileron + aileron_turb, -self.max_aileron, self.max_aileron),
            rudder=np.clip(self.trim_rudder + rudder_turb, -self.max_rudder, self.max_rudder),
            throttle=np.clip(self.cruise_throttle + intensity * 0.1 * np.random.normal(), 0, 1)
        )


class FlightPlan:
    """
    Manages a sequence of flight maneuvers over time.
    """
    
    def __init__(self):
        self.maneuvers = []
        self.control_system = ControlSystem()
    
    def add_maneuver(self, start_time: float, end_time: float, 
                    maneuver_func: Callable[[float], ControlInputs]):
        """Add a maneuver to the flight plan."""
        self.maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'function': maneuver_func
        })
    
    def get_controls(self, time: float) -> ControlInputs:
        """Get control inputs for the current time."""
        # Find the active maneuver
        for maneuver in self.maneuvers:
            if maneuver['start_time'] <= time <= maneuver['end_time']:
                # Adjust time to be relative to maneuver start
                relative_time = time - maneuver['start_time']
                return maneuver['function'](relative_time)
        
        # Default to straight and level if no maneuver is active
        return self.control_system.straight_and_level(time)


def create_sample_flight_plan() -> FlightPlan:
    """Create a sample flight plan with various maneuvers."""
    plan = FlightPlan()
    cs = plan.control_system
    
    # Define the flight sequence
    plan.add_maneuver(0, 15, cs.takeoff_sequence)
    plan.add_maneuver(15, 30, cs.climb_maneuver)
    plan.add_maneuver(30, 60, cs.straight_and_level)
    plan.add_maneuver(60, 90, lambda t: cs.coordinated_turn(t, 0.1))  # Right turn
    plan.add_maneuver(90, 120, cs.straight_and_level)
    plan.add_maneuver(120, 140, lambda t: cs.coordinated_turn(t, -0.1))  # Left turn
    plan.add_maneuver(140, 170, cs.straight_and_level)
    plan.add_maneuver(170, 200, lambda t: cs.turbulence(t, 0.2))  # Moderate turbulence
    plan.add_maneuver(200, 240, cs.landing_approach)
    
    return plan
