"""
Aircraft Model - Defines the physical properties and aerodynamic characteristics
of a small aircraft (based on Cessna 172 specifications).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class AircraftState:
    """Represents the complete state of the aircraft at a given time."""
    time: float
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for easy data analysis."""
        return {
            'time': self.time,
            'x': self.position[0],
            'y': self.position[1], 
            'z': self.position[2],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'vz': self.velocity[2],
            'roll': self.orientation[0],
            'pitch': self.orientation[1],
            'yaw': self.orientation[2],
            'p': self.angular_velocity[0],
            'q': self.angular_velocity[1],
            'r': self.angular_velocity[2],
            'airspeed': np.linalg.norm(self.velocity),
            'altitude': -self.position[2]  # Negative z is altitude
        }


class AircraftModel:
    """
    Small aircraft model based on Cessna 172 specifications.
    Implements 6-DOF flight dynamics with basic aerodynamics.
    """
    
    def __init__(self):
        # Aircraft physical properties (Cessna 172-like)
        self.mass = 1043.0  # kg
        self.wing_area = 16.2  # m^2
        self.wing_span = 11.0  # m
        self.chord = 1.5  # m (mean aerodynamic chord)
        
        # Moments of inertia (kg⋅m²)
        self.Ixx = 1285.0  # Roll inertia
        self.Iyy = 1824.0  # Pitch inertia
        self.Izz = 2666.0  # Yaw inertia
        self.Ixz = 0.0     # Cross-product (assumed zero for simplicity)
        
        # Aerodynamic coefficients
        self.CL0 = 0.28     # Lift coefficient at zero angle of attack
        self.CLalpha = 5.7  # Lift curve slope (per radian)
        self.CD0 = 0.03     # Parasite drag coefficient
        self.K = 0.04       # Induced drag factor
        
        # Control derivatives
        self.CLde = 0.4     # Elevator lift effectiveness
        self.Cmde = -1.1    # Elevator pitch moment
        self.Cydr = 0.14    # Rudder side force
        self.Cndr = -0.074  # Rudder yaw moment
        self.Clda = 0.1     # Aileron roll moment
        
        # Environmental constants
        self.g = 9.81       # Gravity (m/s²)
        self.rho = 1.225    # Air density at sea level (kg/m³)
        
        # Engine properties
        self.max_thrust = 1200.0  # N (approximate for Cessna 172)
        
    def get_inertia_matrix(self) -> np.ndarray:
        """Return the inertia matrix."""
        return np.array([
            [self.Ixx, 0, -self.Ixz],
            [0, self.Iyy, 0],
            [-self.Ixz, 0, self.Izz]
        ])
    
    def rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Create rotation matrix from Euler angles (3-2-1 sequence).
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians  
            yaw: Yaw angle in radians
            
        Returns:
            3x3 rotation matrix from body to earth frame
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
    
    def calculate_aerodynamic_forces(self, state: AircraftState, 
                                   controls: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aerodynamic forces and moments in body frame.
        
        Args:
            state: Current aircraft state
            controls: Control surface deflections (elevator, aileron, rudder, throttle)
            
        Returns:
            Tuple of (forces, moments) in body frame
        """
        # Extract velocity components in body frame
        # For simplicity, assume velocity is already in body frame
        # (In reality, would need to transform from earth to body frame)
        V = np.linalg.norm(state.velocity)
        
        if V < 0.1:  # Avoid division by zero
            return np.zeros(3), np.zeros(3)
        
        # Dynamic pressure
        q = 0.5 * self.rho * V**2
        
        # Angle of attack (simplified - assume small angles)
        alpha = np.arctan2(state.velocity[2], state.velocity[0]) if state.velocity[0] > 0.1 else 0
        
        # Sideslip angle
        beta = np.arcsin(state.velocity[1] / V) if V > 0.1 else 0
        
        # Lift and drag coefficients
        CL = self.CL0 + self.CLalpha * alpha + self.CLde * controls.get('elevator', 0)
        CD = self.CD0 + self.K * CL**2
        
        # Forces in wind frame
        L = q * self.wing_area * CL  # Lift
        D = q * self.wing_area * CD  # Drag
        Y = q * self.wing_area * self.Cydr * controls.get('rudder', 0)  # Side force
        
        # Transform to body frame (simplified)
        # Assuming small angles: Fx ≈ -D, Fy ≈ Y, Fz ≈ -L
        forces = np.array([-D, Y, -L])
        
        # Add thrust
        thrust = controls.get('throttle', 0) * self.max_thrust
        forces[0] += thrust
        
        # Add weight (transformed to body frame)
        R = self.rotation_matrix(state.orientation[0], state.orientation[1], state.orientation[2])
        weight_earth = np.array([0, 0, self.mass * self.g])
        weight_body = R.T @ weight_earth
        forces += weight_body
        
        # Moments (simplified)
        moments = np.array([
            q * self.wing_area * self.wing_span * self.Clda * controls.get('aileron', 0),  # Roll
            q * self.wing_area * self.chord * self.Cmde * controls.get('elevator', 0),     # Pitch
            q * self.wing_area * self.wing_span * self.Cndr * controls.get('rudder', 0)   # Yaw
        ])
        
        return forces, moments
    
    def angular_velocity_to_euler_rates(self, angular_vel: np.ndarray, 
                                      orientation: np.ndarray) -> np.ndarray:
        """
        Convert body angular velocities to Euler angle rates.
        
        Args:
            angular_vel: [p, q, r] in rad/s
            orientation: [roll, pitch, yaw] in radians
            
        Returns:
            Euler angle rates [roll_rate, pitch_rate, yaw_rate]
        """
        roll, pitch, yaw = orientation
        p, q, r = angular_vel
        
        # Gimbal lock protection - limit pitch to avoid singularities
        pitch_limit = np.pi/2 - 0.1  # 89.4 degrees
        pitch = np.clip(pitch, -pitch_limit, pitch_limit)
        
        # Transformation matrix with protected trigonometric functions
        cos_pitch = np.cos(pitch)
        tan_pitch = np.tan(pitch)
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        
        # Protect against division by zero
        if abs(cos_pitch) < 1e-6:
            cos_pitch = np.sign(cos_pitch) * 1e-6
        sec_pitch = 1.0 / cos_pitch
        
        euler_rates = np.array([
            p + q * sin_roll * tan_pitch + r * cos_roll * tan_pitch,
            q * cos_roll - r * sin_roll,
            q * sin_roll * sec_pitch + r * cos_roll * sec_pitch
        ])
        
        # Ensure finite values
        euler_rates = np.where(np.isfinite(euler_rates), euler_rates, 0.0)
        
        return euler_rates


def create_initial_state(position: Tuple[float, float, float] = (0, 0, -1000),
                        velocity: Tuple[float, float, float] = (50, 0, 0),
                        orientation: Tuple[float, float, float] = (0, 0, 0)) -> AircraftState:
    """
    Create an initial aircraft state.
    
    Args:
        position: Initial position [x, y, z] in meters (z negative for altitude)
        velocity: Initial velocity [vx, vy, vz] in m/s
        orientation: Initial orientation [roll, pitch, yaw] in radians
        
    Returns:
        Initial AircraftState object
    """
    return AircraftState(
        time=0.0,
        position=np.array(position),
        velocity=np.array(velocity),
        orientation=np.array(orientation),
        angular_velocity=np.zeros(3)
    )
