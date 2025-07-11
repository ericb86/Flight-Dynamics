"""
Visualization Module - Creates interactive plots and animations of flight data
using matplotlib and plotly for comprehensive flight analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional


class FlightVisualizer:
    """
    Creates various visualizations of flight simulation data.
    """
    
    def __init__(self, flight_data: List[Dict[str, Any]]):
        """
        Initialize visualizer with flight data.
        
        Args:
            flight_data: List of flight data dictionaries from simulation
        """
        self.flight_data = flight_data
        self.df = pd.DataFrame(flight_data)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
    def plot_3d_trajectory(self, show_orientation: bool = True, 
                          animate: bool = False) -> go.Figure:
        """
        Create 3D trajectory plot with optional orientation vectors.
        
        Args:
            show_orientation: Whether to show aircraft orientation vectors
            animate: Whether to create animated plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Main trajectory
        fig.add_trace(go.Scatter3d(
            x=self.df['x'],
            y=self.df['y'], 
            z=self.df['altitude'],
            mode='lines+markers',
            line=dict(color=self.df['airspeed'], colorscale='Viridis', width=4),
            marker=dict(size=2),
            name='Flight Path',
            text=[f'Time: {t:.1f}s<br>Speed: {v:.1f} m/s<br>Alt: {a:.1f}m' 
                  for t, v, a in zip(self.df['time'], self.df['airspeed'], self.df['altitude'])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Start and end points
        fig.add_trace(go.Scatter3d(
            x=[self.df['x'].iloc[0]],
            y=[self.df['y'].iloc[0]],
            z=[self.df['altitude'].iloc[0]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[self.df['x'].iloc[-1]],
            y=[self.df['y'].iloc[-1]],
            z=[self.df['altitude'].iloc[-1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='End',
            showlegend=True
        ))
        
        # Orientation vectors (sample every 50 points to avoid clutter)
        if show_orientation and len(self.df) > 10:
            step = max(1, len(self.df) // 20)  # Show ~20 orientation vectors
            
            for i in range(0, len(self.df), step):
                row = self.df.iloc[i]
                
                # Calculate orientation vectors
                roll, pitch, yaw = row['roll'], row['pitch'], row['yaw']
                
                # Forward vector (nose direction)
                forward = np.array([
                    np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch)
                ]) * 50  # Scale for visibility
                
                # Add orientation vector
                fig.add_trace(go.Scatter3d(
                    x=[row['x'], row['x'] + forward[0]],
                    y=[row['y'], row['y'] + forward[1]],
                    z=[row['altitude'], row['altitude'] + forward[2]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Layout
        fig.update_layout(
            title='3D Flight Trajectory',
            scene=dict(
                xaxis_title='X Position (m)',
                yaxis_title='Y Position (m)',
                zaxis_title='Altitude (m)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_flight_parameters(self) -> go.Figure:
        """
        Create comprehensive flight parameter plots.
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Altitude vs Time', 'Airspeed vs Time',
                'Attitude (Roll, Pitch, Yaw)', 'Control Inputs',
                'Velocity Components', 'Angular Velocities'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Altitude
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['altitude'],
                      mode='lines', name='Altitude', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Airspeed
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['airspeed'],
                      mode='lines', name='Airspeed', line=dict(color='green')),
            row=1, col=2
        )
        
        # Attitude angles (convert to degrees)
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['roll']),
                      mode='lines', name='Roll', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['pitch']),
                      mode='lines', name='Pitch', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['yaw']),
                      mode='lines', name='Yaw', line=dict(color='green')),
            row=2, col=1
        )
        
        # Control inputs
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['elevator'],
                      mode='lines', name='Elevator', line=dict(color='purple')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['aileron'],
                      mode='lines', name='Aileron', line=dict(color='orange')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['rudder'],
                      mode='lines', name='Rudder', line=dict(color='brown')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['throttle'],
                      mode='lines', name='Throttle', line=dict(color='black')),
            row=2, col=2
        )
        
        # Velocity components
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['vx'],
                      mode='lines', name='Vx', line=dict(color='red')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['vy'],
                      mode='lines', name='Vy', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['vz'],
                      mode='lines', name='Vz', line=dict(color='green')),
            row=3, col=1
        )
        
        # Angular velocities (convert to degrees/second)
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['p']),
                      mode='lines', name='Roll Rate', line=dict(color='red')),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['q']),
                      mode='lines', name='Pitch Rate', line=dict(color='blue')),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=np.degrees(self.df['r']),
                      mode='lines', name='Yaw Rate', line=dict(color='green')),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Flight Parameters Analysis',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
        fig.update_yaxes(title_text="Speed (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Angle (°)", row=2, col=1)
        fig.update_yaxes(title_text="Control Input", row=2, col=2)
        fig.update_yaxes(title_text="Velocity (m/s)", row=3, col=1)
        fig.update_yaxes(title_text="Angular Rate (°/s)", row=3, col=2)
        
        return fig
    
    def create_animated_trajectory(self, save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animated matplotlib visualization of the flight.
        
        Args:
            save_path: Optional path to save animation as GIF
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Create 3D subplot
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # Initialize empty plots
        line_3d, = ax1.plot([], [], [], 'b-', linewidth=2)
        point_3d, = ax1.plot([], [], [], 'ro', markersize=8)
        
        line_alt, = ax2.plot([], [], 'b-', linewidth=2)
        line_speed, = ax3.plot([], [], 'g-', linewidth=2)
        line_attitude, = ax4.plot([], [], 'r-', linewidth=2, label='Roll')
        line_pitch, = ax4.plot([], [], 'b-', linewidth=2, label='Pitch')
        line_yaw, = ax4.plot([], [], 'g-', linewidth=2, label='Yaw')
        
        # Set up axes
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D Trajectory')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude vs Time')
        ax2.grid(True)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Airspeed (m/s)')
        ax3.set_title('Airspeed vs Time')
        ax3.grid(True)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (°)')
        ax4.set_title('Aircraft Attitude')
        ax4.legend()
        ax4.grid(True)
        
        # Set axis limits
        ax1.set_xlim(self.df['x'].min(), self.df['x'].max())
        ax1.set_ylim(self.df['y'].min(), self.df['y'].max())
        ax1.set_zlim(self.df['altitude'].min(), self.df['altitude'].max())
        
        ax2.set_xlim(self.df['time'].min(), self.df['time'].max())
        ax2.set_ylim(self.df['altitude'].min(), self.df['altitude'].max())
        
        ax3.set_xlim(self.df['time'].min(), self.df['time'].max())
        ax3.set_ylim(self.df['airspeed'].min(), self.df['airspeed'].max())
        
        ax4.set_xlim(self.df['time'].min(), self.df['time'].max())
        ax4.set_ylim(-30, 30)  # Reasonable attitude range
        
        def animate(frame):
            # Update 3D trajectory
            x_data = self.df['x'][:frame]
            y_data = self.df['y'][:frame]
            z_data = self.df['altitude'][:frame]
            
            line_3d.set_data_3d(x_data, y_data, z_data)
            if frame > 0:
                point_3d.set_data_3d([x_data.iloc[-1]], [y_data.iloc[-1]], [z_data.iloc[-1]])
            
            # Update 2D plots
            time_data = self.df['time'][:frame]
            
            line_alt.set_data(time_data, self.df['altitude'][:frame])
            line_speed.set_data(time_data, self.df['airspeed'][:frame])
            line_attitude.set_data(time_data, np.degrees(self.df['roll'][:frame]))
            line_pitch.set_data(time_data, np.degrees(self.df['pitch'][:frame]))
            line_yaw.set_data(time_data, np.degrees(self.df['yaw'][:frame]))
            
            return line_3d, point_3d, line_alt, line_speed, line_attitude, line_pitch, line_yaw
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=len(self.df),
            interval=50, blit=False, repeat=True
        )
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print("Animation saved!")
        
        return anim
    
    def plot_performance_metrics(self) -> go.Figure:
        """
        Create performance analysis plots.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Altitude Profile', 'Speed Profile',
                'Flight Envelope', 'Control Activity'
            )
        )
        
        # Altitude profile with phases
        colors = []
        for alt in self.df['altitude']:
            if alt < 200:
                colors.append('red')    # Low altitude
            elif alt < 500:
                colors.append('orange') # Medium altitude
            else:
                colors.append('green')  # High altitude
        
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['altitude'],
                      mode='markers+lines',
                      marker=dict(color=colors),
                      name='Altitude'),
            row=1, col=1
        )
        
        # Speed profile
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=self.df['airspeed'],
                      mode='lines', name='Airspeed'),
            row=1, col=2
        )
        
        # Flight envelope (altitude vs speed)
        fig.add_trace(
            go.Scatter(x=self.df['airspeed'], y=self.df['altitude'],
                      mode='markers',
                      marker=dict(color=self.df['time'], colorscale='Viridis'),
                      name='Flight Envelope'),
            row=2, col=1
        )
        
        # Control activity (RMS of control inputs)
        window_size = 20
        control_activity = []
        for i in range(len(self.df)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(self.df), i + window_size // 2)
            
            window_data = self.df.iloc[start_idx:end_idx]
            rms = np.sqrt(np.mean(
                window_data['elevator']**2 +
                window_data['aileron']**2 +
                window_data['rudder']**2
            ))
            control_activity.append(rms)
        
        fig.add_trace(
            go.Scatter(x=self.df['time'], y=control_activity,
                      mode='lines', name='Control Activity'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Flight Performance Analysis',
            height=600
        )
        
        return fig
    
    def show_all_plots(self, save_html: bool = True):
        """
        Display all visualization plots.
        
        Args:
            save_html: Whether to save plots as HTML files
        """
        print("Generating flight visualizations...")
        
        # 3D Trajectory
        fig_3d = self.plot_3d_trajectory()
        fig_3d.show()
        if save_html:
            fig_3d.write_html("3d_trajectory.html")
        
        # Flight Parameters
        fig_params = self.plot_flight_parameters()
        fig_params.show()
        if save_html:
            fig_params.write_html("flight_parameters.html")
        
        # Performance Metrics
        fig_perf = self.plot_performance_metrics()
        fig_perf.show()
        if save_html:
            fig_perf.write_html("performance_metrics.html")
        
        print("All plots generated successfully!")


def main():
    """Example usage of the visualization module."""
    # Load sample data (you would typically load from simulation)
    import json
    
    try:
        # Try to load existing flight data
        with open('sample_flight.json', 'r') as f:
            flight_data = json.load(f)
        
        # Create visualizer
        viz = FlightVisualizer(flight_data)
        
        # Show all plots
        viz.show_all_plots()
        
        # Create animation
        anim = viz.create_animated_trajectory('flight_animation.gif')
        plt.show()
        
    except FileNotFoundError:
        print("No flight data found. Run flight_simulator.py first to generate data.")


if __name__ == "__main__":
    main()
