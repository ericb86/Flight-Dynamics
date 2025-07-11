@echo off
cd /d "%~dp0"
call flight-env\Scripts\activate
jupyter notebook Flight_Dynamics_Simulation.ipynb
pause