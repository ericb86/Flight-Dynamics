"""
Streamlit Dashboard for Flight Dynamics Simulation
Interactive web interface for running and visualizing flight simulations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our simulation modules
try:
    from flight_simulator import FlightSimulator
    from aircraft_model import AircraftModel, create_initial_state
    from control_system import FlightPlan, create_sample_flight_plan
    from visualization import FlightVisualizer
except ImportError:
    st.error("Please ensure all simulation modules are in the same directory as this dashboard.")
    st.stop()


def main():
    st.set_page_config(
        page_title="Flight Dynamics Simulation",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("✈️ Flight Dynamics Simulation Dashboard")
    st.markdown("Interactive 6-DOF aircraft simulation with real-time visualization")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Parameters")
    
    # Aircraft parameters
    st.sidebar.subheader("Aircraft Configuration")
    aircraft_type = st.sidebar.selectbox(
        "Aircraft Type",
        ["Light Aircraft (Cessna 172)", "Heavy Aircraft", "Custom"]
    )
    
    if aircraft_type == "Custom":
        mass = st.sidebar.slider("Mass (kg)", 500, 2000, 1043)
        max_thrust = st.sidebar.slider("Max Thrust (N)", 500, 2000, 1200)
    else:
        mass = 1043 if aircraft_type == "Light Aircraft (Cessna 172)" else 1565
        max_thrust = 1200 if aircraft_type == "Light Aircraft (Cessna 172)" else 1800
    
    # Simulation parameters
    st.sidebar.subheader("Flight Parameters")
    duration = st.sidebar.slider("Flight Duration (seconds)", 30, 300, 120)
    initial_altitude = st.sidebar.slider("Initial Altitude (m)", 50, 500, 100)
    initial_speed = st.sidebar.slider("Initial Speed (m/s)", 20, 80, 50)
    
    # Maneuver selection
    st.sidebar.subheader("Flight Maneuvers")
    include_turns = st.sidebar.checkbox("Include Turning Maneuvers", True)
    include_climb = st.sidebar.checkbox("Include Climb/Descent", True)
    turbulence_level = st.sidebar.slider("Turbulence Level", 0.0, 0.5, 0.1)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        run_simulation(mass, max_thrust, duration, initial_altitude, initial_speed,
                      include_turns, include_climb, turbulence_level)


def run_simulation(mass, max_thrust, duration, initial_altitude, initial_speed,
                  include_turns, include_climb, turbulence_level):
    """Run the flight simulation with given parameters."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create aircraft model
    status_text.text("🔧 Configuring aircraft...")
    progress_bar.progress(20)
    
    aircraft = AircraftModel()
    aircraft.mass = mass
    aircraft.max_thrust = max_thrust
    
    # Create flight plan
    status_text.text("📋 Planning flight...")
    progress_bar.progress(40)
    
    plan = create_custom_flight_plan(duration, include_turns, include_climb, turbulence_level)
    
    # Create initial state
    initial_state = create_initial_state(
        position=(0, 0, -initial_altitude),
        velocity=(initial_speed, 0, 0),
        orientation=(0, 0.1, 0)
    )
    
    # Run simulation
    status_text.text("🛫 Running simulation...")
    progress_bar.progress(60)
    
    simulator = FlightSimulator(aircraft_model=aircraft, flight_plan=plan)
    flight_data = simulator.run_simulation(duration=duration, initial_state=initial_state)
    
    # Create visualizations
    status_text.text("📊 Generating visualizations...")
    progress_bar.progress(80)
    
    viz = FlightVisualizer(flight_data)
    
    # Display results
    progress_bar.progress(100)
    status_text.text("✅ Simulation complete!")
    
    display_results(flight_data, viz, simulator)


def create_custom_flight_plan(duration, include_turns, include_climb, turbulence_level):
    """Create a custom flight plan based on user preferences."""
    plan = FlightPlan()
    cs = plan.control_system
    
    current_time = 0
    phase_duration = duration / 4
    
    # Takeoff phase
    plan.add_maneuver(current_time, current_time + 15, cs.takeoff_sequence)
    current_time += 15
    
    # Climb phase (if enabled)
    if include_climb:
        plan.add_maneuver(current_time, current_time + phase_duration, cs.climb_maneuver)
        current_time += phase_duration
    
    # Cruise with optional turns
    if include_turns:
        # Add turning maneuvers
        turn_duration = phase_duration / 2
        plan.add_maneuver(current_time, current_time + turn_duration, 
                         lambda t: cs.coordinated_turn(t, 0.1))
        current_time += turn_duration
        
        plan.add_maneuver(current_time, current_time + turn_duration,
                         lambda t: cs.coordinated_turn(t, -0.1))
        current_time += turn_duration
    else:
        plan.add_maneuver(current_time, current_time + phase_duration, cs.straight_and_level)
        current_time += phase_duration
    
    # Add turbulence if requested
    if turbulence_level > 0:
        plan.add_maneuver(current_time, duration,
                         lambda t: cs.turbulence(t, turbulence_level))
    else:
        plan.add_maneuver(current_time, duration, cs.straight_and_level)
    
    return plan


def display_results(flight_data, viz, simulator):
    """Display simulation results in the dashboard."""
    
    # Summary statistics
    st.header("📊 Flight Summary")
    summary = simulator.get_flight_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Flight Duration", f"{summary['flight_duration']:.1f} s")
        st.metric("Max Altitude", f"{summary['max_altitude']:.1f} m")
    
    with col2:
        st.metric("Max Airspeed", f"{summary['max_airspeed']:.1f} m/s")
        st.metric("Min Airspeed", f"{summary['min_airspeed']:.1f} m/s")
    
    with col3:
        st.metric("Average Altitude", f"{summary['average_altitude']:.1f} m")
        st.metric("Average Airspeed", f"{summary['average_airspeed']:.1f} m/s")
    
    with col4:
        st.metric("Total Distance", f"{summary['total_distance']:.0f} m")
        st.metric("Data Points", f"{summary['data_points']}")
    
    # 3D Flight Path
    st.header("🛩️ 3D Flight Trajectory")
    fig_3d = viz.plot_3d_trajectory(show_orientation=True)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Flight Parameters
    st.header("📈 Flight Parameters Analysis")
    fig_params = viz.plot_flight_parameters()
    st.plotly_chart(fig_params, use_container_width=True)
    
    # Performance Metrics
    st.header("🎯 Performance Metrics")
    fig_perf = viz.plot_performance_metrics()
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Data table
    st.header("📋 Flight Data")
    df = pd.DataFrame(flight_data)
    
    # Show sample of data
    st.subheader("Sample Data (First 10 Points)")
    display_columns = ['time', 'altitude', 'airspeed', 'roll', 'pitch', 'yaw',
                      'elevator', 'aileron', 'rudder', 'throttle']
    
    # Convert angles to degrees for display
    df_display = df.copy()
    df_display['roll'] = np.degrees(df_display['roll'])
    df_display['pitch'] = np.degrees(df_display['pitch'])
    df_display['yaw'] = np.degrees(df_display['yaw'])
    
    st.dataframe(df_display[display_columns].head(10).round(3))
    
    # Download options
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name="flight_simulation_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Save 3D plot as HTML
        html_str = fig_3d.to_html(include_plotlyjs='cdn')
        st.download_button(
            label="🌐 Download 3D Plot (HTML)",
            data=html_str,
            file_name="3d_flight_visualization.html",
            mime="text/html"
        )


def show_info_page():
    """Display information about the simulation."""
    st.header("ℹ️ About Flight Dynamics Simulation")
    
    st.markdown("""
    ### 🎯 Project Overview
    This interactive dashboard demonstrates a **6-degree-of-freedom flight simulation** 
    of a small aircraft using Python. The simulation models realistic flight dynamics 
    including aerodynamic forces, control surface effects, and aircraft motion.
    
    ### 🔬 Physics Implementation
    - **Newton's Laws**: Force = mass × acceleration for translational motion
    - **Rotational Dynamics**: Torque = inertia × angular acceleration
    - **Euler Angles**: Roll, pitch, yaw representation of aircraft orientation
    - **Aerodynamic Forces**: Lift, drag, thrust, and weight calculations
    - **Control Surfaces**: Elevator, aileron, rudder effects on aircraft motion
    
    ### 🛠️ Technologies Used
    - **NumPy & SciPy**: Numerical computation and differential equation solving
    - **Plotly**: Interactive 3D visualizations and data plots
    - **Streamlit**: Web dashboard interface
    - **Pandas**: Data analysis and manipulation
    
    ### 🎮 How to Use
    1. **Configure Aircraft**: Choose aircraft type and adjust parameters
    2. **Set Flight Parameters**: Duration, altitude, speed, and maneuvers
    3. **Run Simulation**: Click the "Run Simulation" button
    4. **Analyze Results**: Explore 3D visualizations and performance metrics
    5. **Download Data**: Export results for further analysis
    
    ### 📚 Educational Value
    This simulation teaches:
    - Aerospace engineering principles
    - Numerical simulation techniques
    - Data visualization and analysis
    - Scientific Python programming
    - Interactive dashboard development
    """)


# Main app logic
def app():
    # Add navigation
    page = st.sidebar.selectbox("📍 Navigation", ["Simulation", "About"])
    
    if page == "Simulation":
        main()
    else:
        show_info_page()


if __name__ == "__main__":
    app()
