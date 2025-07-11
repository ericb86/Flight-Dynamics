# Flight Dynamics Simulation

A Python-based flight simulation that models the 6-degree-of-freedom motion of a small aircraft, including pitch, roll, yaw dynamics, and 3D trajectory visualization.

## 🎯 Project Overview

This project simulates the flight dynamics of a small aircraft using Newton's laws of motion and Euler angles. It provides real-time visualization of the aircraft's trajectory, orientation, and flight parameters.

## 🚀 Features

- **6-DOF Flight Simulation**: Models translation and rotation in 3D space
- **Control Input System**: Simulates elevator, aileron, rudder, and throttle inputs
- **Real-time Visualization**: Interactive 3D flight path and orientation plots
- **Flight Data Analytics**: Time-series analysis of altitude, speed, and orientation
- **Aerodynamic Modeling**: Basic aerodynamic forces and moments

## 📋 Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Plotly
- Pandas

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flight-dynamics.git
cd flight-dynamics
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

```python
from flight_simulator import FlightSimulator
from visualization import FlightVisualizer

# Create and run simulation
sim = FlightSimulator()
flight_data = sim.run_simulation(duration=60)  # 60 seconds

# Visualize results
viz = FlightVisualizer(flight_data)
viz.plot_3d_trajectory()
viz.plot_flight_parameters()
```

## 📊 Example Output

The simulation generates:
- 3D flight trajectory visualization
- Time-series plots of altitude, speed, and orientation
- Control input analysis
- Performance metrics

## 🔧 Project Structure

```
flight-dynamics/
├── flight_simulator.py     # Core simulation engine
├── aircraft_model.py       # Aircraft physics and aerodynamics
├── control_system.py       # Flight control inputs
├── visualization.py        # Plotting and animation
├── data_analysis.py        # Flight data analysis tools
├── examples/               # Example simulations
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## 🧮 Physics Implementation

The simulation implements:
- **Translational Motion**: F = ma (forces → acceleration → velocity → position)
- **Rotational Motion**: τ = Iα (torques → angular acceleration → angular velocity → orientation)
- **Euler Angles**: Roll, pitch, yaw representation
- **Aerodynamic Forces**: Lift, drag, thrust, weight
- **Control Surfaces**: Elevator, aileron, rudder effects

## 📈 Technologies Used

- **Simulation**: NumPy, SciPy (numerical integration)
- **Visualization**: Matplotlib, Plotly (3D plotting, animation)
- **Data Analysis**: Pandas (time-series analysis)
- **Mathematics**: Linear algebra, differential equations, Euler angles

## 🎓 Learning Outcomes

- Flight dynamics and aerodynamics principles
- Numerical simulation techniques
- 3D visualization and animation
- Time-series data analysis
- Object-oriented programming in scientific computing

## 🔮 Future Enhancements

- [ ] Real aircraft parameters (Cessna 172, Piper Cherokee)
- [ ] Advanced aerodynamic models (stall, ground effect)
- [ ] GUI dashboard with Streamlit
- [ ] PID autopilot system
- [ ] Integration with X-Plane or FlightGear
- [ ] Weather effects simulation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/flight-dynamics
