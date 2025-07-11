"""
Flight Simulator - Main simulation engine that integrates aircraft dynamics,
control systems, and numerical integration to simulate flight.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict, Any, Optional
import time as timer

from aircraft_model import AircraftModel, AircraftState, create_initial_state
from control_system import FlightPlan, create_sample_flight_plan, ControlInputs


class FlightSimulator:
    """
    Main flight simulation engine using 6-DOF aircraft dynamics.
    """
    
    def __init__(self, aircraft_model: Optional[AircraftModel] = None,
                 flight_plan: Optional[FlightPlan] = None):
        """
        Initialize the flight simulator.
        
        Args:
            aircraft_model: Aircraft model to use (default: create new one)
            flight_plan: Flight plan with control inputs (default: sample plan)
        """
        self.aircraft = aircraft_model or AircraftModel()
        self.flight_plan = flight_plan or create_sample_flight_plan()
        
        # Simulation parameters
        self.dt = 0.01  # Time step in seconds
        self.max_step = 0.1  # Maximum integration step
        
        # Data storage
        self.flight_data: List[Dict[str, Any]] = []
        self.simulation_time = 0.0
        
    def state_derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate state derivatives for numerical integration.
        
        Args:
            t: Current time
            y: State vector [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
            
        Returns:
            State derivative vector
        """
        try:
            # Validate input
            if len(y) != 12:
                raise ValueError(f"State vector must have 12 elements, got {len(y)}")
            
            # Unpack state vector and ensure proper shape
            position = np.array(y[0:3])
            velocity = np.array(y[3:6])
            orientation = np.array(y[6:9])
            angular_velocity = np.array(y[9:12])
            
            # Validate arrays are finite
            if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity)) and 
                    np.all(np.isfinite(orientation)) and np.all(np.isfinite(angular_velocity))):
                raise ValueError("Non-finite values detected in state vector")
            
            # Create current state object
            current_state = AircraftState(
                time=t,
                position=position,
                velocity=velocity,
                orientation=orientation,
                angular_velocity=angular_velocity
            )
            
            # Get control inputs for current time
            controls = self.flight_plan.get_controls(t)
            
            # Calculate aerodynamic forces and moments
            forces, moments = self.aircraft.calculate_aerodynamic_forces(
                current_state, controls.to_dict()
            )
            
            # Ensure forces and moments are numpy arrays
            forces = np.array(forces)
            moments = np.array(moments)
            
            # Position derivatives (velocity in earth frame)
            R = self.aircraft.rotation_matrix(orientation[0], orientation[1], orientation[2])
            position_dot = R @ velocity
            
            # Velocity derivatives (acceleration in body frame)
            # F = ma, so a = F/m
            velocity_dot = forces / self.aircraft.mass
            
            # Add centrifugal and Coriolis effects
            omega = np.array(angular_velocity)
            v = np.array(velocity)
            velocity_dot += np.cross(omega, v)
            
            # Orientation derivatives (Euler angle rates)
            orientation_dot = self.aircraft.angular_velocity_to_euler_rates(
                angular_velocity, orientation
            )
            
            # Angular velocity derivatives (angular acceleration)
            # τ = Iω̇ + ω × (Iω)
            I = self.aircraft.get_inertia_matrix()
            angular_momentum = I @ np.array(angular_velocity)
            gyroscopic_moment = np.cross(np.array(angular_velocity), angular_momentum)
            
            try:
                angular_velocity_dot = np.linalg.solve(I, np.array(moments) - gyroscopic_moment)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                angular_velocity_dot = np.array(moments) / np.diag(I)
            
            # Combine all derivatives
            y_dot = np.concatenate([
                position_dot,
                velocity_dot,
                orientation_dot,
                angular_velocity_dot
            ])
            
            # Final validation
            if not np.all(np.isfinite(y_dot)):
                raise ValueError("Non-finite values in derivatives")
            
            return y_dot
            
        except Exception as e:
            print(f"Error in state_derivatives at t={t}: {e}")
            # Return zero derivatives to prevent crash
            return np.zeros(12)
    
    def run_simulation(self, duration: float = 5.0,  # Reduced default duration
                      initial_state: Optional[AircraftState] = None,
                      save_interval: float = 0.5,  # Larger save interval
                      real_time: bool = False) -> List[Dict[str, Any]]:
        """
        Run the flight simulation.
        
        Args:
            duration: Simulation duration in seconds
            initial_state: Initial aircraft state (default: create standard initial state)
            save_interval: Data save interval in seconds
            real_time: Whether to run in real time (for visualization)
            
        Returns:
            List of flight data dictionaries
        """
        print(f"Starting flight simulation for {duration} seconds...")
        start_time = timer.time()
        
        # Set initial state
        if initial_state is None:
            initial_state = create_initial_state(
                position=(0, 0, -100),  # Start at 100m altitude
                velocity=(50, 0, 0),    # 50 m/s forward
                orientation=(0, 0.1, 0) # Slight nose-up attitude
            )
        
        # Pack initial state into vector
        y0 = np.concatenate([
            initial_state.position,
            initial_state.velocity,
            initial_state.orientation,
            initial_state.angular_velocity
        ])
        
        # Time points for data saving
        t_eval = np.arange(0, duration + save_interval, save_interval)
        
        print("Integrating equations of motion...")
        
        # Solve the differential equation
        try:
            solution = solve_ivp(
                self.state_derivatives,
                t_span=(0, duration),
                y0=y0,
                t_eval=t_eval,
                method='RK23',  # More robust method
                max_step=0.5,   # Larger max step for faster computation
                rtol=1e-3,      # Relaxed tolerance
                atol=1e-6       # Relaxed tolerance
            )
            
            if not solution.success:
                print(f"Warning: Integration may not have converged. Message: {solution.message}")
                
        except Exception as e:
            print(f"Error during integration: {e}")
            return []
        
        print("Processing simulation results...")
        
        # Process results
        self.flight_data = []
        for i, t in enumerate(solution.t):
            y = solution.y[:, i]
            
            # Create state object
            state = AircraftState(
                time=t,
                position=y[0:3],
                velocity=y[3:6],
                orientation=y[6:9],
                angular_velocity=y[9:12]
            )
            
            # Get control inputs for this time
            controls = self.flight_plan.get_controls(t)
            
            # Store data
            data_point = state.to_dict()
            data_point.update({
                'elevator': controls.elevator,
                'aileron': controls.aileron,
                'rudder': controls.rudder,
                'throttle': controls.throttle
            })
            
            self.flight_data.append(data_point)
            
            # Real-time delay if requested
            if real_time and i > 0:
                elapsed = timer.time() - start_time
                target_time = t
                if elapsed < target_time:
                    timer.sleep(target_time - elapsed)
        
        elapsed_time = timer.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Simulated {len(self.flight_data)} data points")
        
        return self.flight_data
    
    def get_flight_summary(self) -> Dict[str, Any]:
        """Generate a summary of the flight data."""
        if not self.flight_data:
            return {}
        
        # Convert to numpy arrays for analysis
        times = np.array([d['time'] for d in self.flight_data])
        altitudes = np.array([d['altitude'] for d in self.flight_data])
        airspeeds = np.array([d['airspeed'] for d in self.flight_data])
        
        # Calculate statistics
        summary = {
            'flight_duration': times[-1] - times[0],
            'max_altitude': np.max(altitudes),
            'min_altitude': np.min(altitudes),
            'average_altitude': np.mean(altitudes),
            'max_airspeed': np.max(airspeeds),
            'min_airspeed': np.min(airspeeds),
            'average_airspeed': np.mean(airspeeds),
            'total_distance': self._calculate_total_distance(),
            'data_points': len(self.flight_data)
        }
        
        return summary
    
    def _calculate_total_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.flight_data) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.flight_data)):
            prev_pos = np.array([
                self.flight_data[i-1]['x'],
                self.flight_data[i-1]['y'],
                self.flight_data[i-1]['z']
            ])
            curr_pos = np.array([
                self.flight_data[i]['x'],
                self.flight_data[i]['y'],
                self.flight_data[i]['z']
            ])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
        
        return total_distance
    
    def export_to_csv(self, filename: str = "flight_data.csv"):
        """Export flight data to CSV file."""
        import pandas as pd
        
        if not self.flight_data:
            print("No flight data to export")
            return
        
        df = pd.DataFrame(self.flight_data)
        df.to_csv(filename, index=False)
        print(f"Flight data exported to {filename}")


def main():
    """Example usage of the flight simulator."""
    # Create simulator with default settings
    sim = FlightSimulator()
    
    # Run simulation
    flight_data = sim.run_simulation(duration=5.0)  # 5 seconds for faster demo
    
    # Print summary
    summary = sim.get_flight_summary()
    print("\n" + "="*50)
    print("FLIGHT SUMMARY")
    print("="*50)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Export data
    sim.export_to_csv("sample_flight.csv")
    
    print("\nSimulation complete! Use visualization.py to plot the results.")


if __name__ == "__main__":
    main()
