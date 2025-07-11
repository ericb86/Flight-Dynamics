"""
Setup Script for Flight Dynamics Simulation Project
Automates the installation and configuration process.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True


def create_virtual_environment():
    """Create a virtual environment for the project."""
    venv_path = Path("flight-env")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "flight-env"], 
                      check=True, capture_output=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def get_pip_command():
    """Get the appropriate pip command for the virtual environment."""
    if os.name == 'nt':  # Windows
        return "flight-env\\Scripts\\pip"
    else:  # macOS/Linux
        return "flight-env/bin/pip"


def install_packages():
    """Install required packages."""
    packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "plotly>=5.13.0",
        "pandas>=1.5.0",
        "jupyter>=1.0.0",
        "streamlit>=1.20.0"
    ]
    
    pip_cmd = get_pip_command()
    
    print("📦 Installing required packages...")
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([pip_cmd, "install", package], 
                          check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully")
    return True


def create_launch_scripts():
    """Create convenient launch scripts."""
    
    # Jupyter notebook launcher
    if os.name == 'nt':  # Windows
        jupyter_script = """@echo off
cd /d "%~dp0"
call flight-env\\Scripts\\activate
jupyter notebook Flight_Dynamics_Simulation.ipynb
pause"""
        with open("launch_notebook.bat", "w") as f:
            f.write(jupyter_script)
        
        # Streamlit dashboard launcher
        streamlit_script = """@echo off
cd /d "%~dp0"
call flight-env\\Scripts\\activate
streamlit run dashboard.py
pause"""
        with open("launch_dashboard.bat", "w") as f:
            f.write(streamlit_script)
        
        print("✅ Created launch_notebook.bat and launch_dashboard.bat")
    
    else:  # macOS/Linux
        jupyter_script = """#!/bin/bash
cd "$(dirname "$0")"
source flight-env/bin/activate
jupyter notebook Flight_Dynamics_Simulation.ipynb"""
        with open("launch_notebook.sh", "w") as f:
            f.write(jupyter_script)
        os.chmod("launch_notebook.sh", 0o755)
        
        # Streamlit dashboard launcher
        streamlit_script = """#!/bin/bash
cd "$(dirname "$0")"
source flight-env/bin/activate
streamlit run dashboard.py"""
        with open("launch_dashboard.sh", "w") as f:
            f.write(streamlit_script)
        os.chmod("launch_dashboard.sh", 0o755)
        
        print("✅ Created launch_notebook.sh and launch_dashboard.sh")


def verify_installation():
    """Verify that all components are working correctly."""
    print("🔍 Verifying installation...")
    
    # Test imports
    pip_cmd = get_pip_command()
    
    test_script = """
import numpy as np
import scipy
import matplotlib
import plotly
import pandas as pd
import streamlit

print("All imports successful!")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Streamlit: {streamlit.__version__}")
"""
    
    try:
        if os.name == 'nt':  # Windows
            python_cmd = "flight-env\\Scripts\\python"
        else:  # macOS/Linux
            python_cmd = "flight-env/bin/python"
        
        result = subprocess.run([python_cmd, "-c", test_script], 
                               capture_output=True, text=True, check=True)
        print("✅ Installation verification successful!")
        print(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation verification failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def print_usage_instructions():
    """Print instructions for using the project."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📚 HOW TO USE:")
    
    if os.name == 'nt':  # Windows
        print("• Double-click 'launch_notebook.bat' to open Jupyter notebook")
        print("• Double-click 'launch_dashboard.bat' to open Streamlit dashboard")
        print("• Or manually activate environment: flight-env\\Scripts\\activate")
    else:  # macOS/Linux
        print("• Run './launch_notebook.sh' to open Jupyter notebook")
        print("• Run './launch_dashboard.sh' to open Streamlit dashboard")
        print("• Or manually activate environment: source flight-env/bin/activate")
    
    print("\n🛩️ PROJECT COMPONENTS:")
    print("• Flight_Dynamics_Simulation.ipynb - Complete tutorial notebook")
    print("• flight_simulator.py - Core simulation engine")
    print("• aircraft_model.py - Aircraft physics and parameters")
    print("• control_system.py - Flight control inputs")
    print("• visualization.py - Data plotting and analysis")
    print("• dashboard.py - Interactive Streamlit web app")
    print("• examples.py - Usage examples and demos")
    
    print("\n🎯 PORTFOLIO TIPS:")
    print("• Run the Jupyter notebook for the complete tutorial")
    print("• Use the Streamlit dashboard for interactive demos")
    print("• Export visualizations as HTML for web portfolios")
    print("• Customize aircraft parameters for different scenarios")
    
    print("\n📞 SUPPORT:")
    print("• Check README.md for detailed documentation")
    print("• All code is well-commented for learning")
    print("• Modify parameters to explore different flight scenarios")


def main():
    """Main setup function."""
    print("🛩️ Flight Dynamics Simulation - Setup Script")
    print("="*60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing packages", install_packages),
        ("Creating launch scripts", create_launch_scripts),
        ("Verifying installation", verify_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    # Success!
    print_usage_instructions()


if __name__ == "__main__":
    main()
