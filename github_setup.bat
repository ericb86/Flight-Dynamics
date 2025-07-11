@echo off
echo ===============================================
echo      Flight Dynamics - GitHub Setup Guide
echo ===============================================
echo.
echo This will help you put your project on GitHub!
echo.
echo STEP 1: Clean up the project
echo Removing debug files and cache...

if exist "debug_math.py" del "debug_math.py"
if exist "test_quick.py" del "test_quick.py" 
if exist "simple_simulator.py" del "simple_simulator.py"
if exist "sample_flight.csv" del "sample_flight.csv"

echo.
echo STEP 2: What you need to do next:
echo.
echo 1. Go to https://github.com
echo 2. Click "New repository" (green button)
echo 3. Name it: flight-dynamics-simulation
echo 4. Make it PUBLIC (so employers can see it)
echo 5. Do NOT check "Add README" (we already have one)
echo 6. Click "Create repository"
echo.
echo STEP 3: After creating the repo, run these commands:
echo.
echo git init
echo git add .
echo git commit -m "Initial commit: Flight dynamics simulation"
echo git branch -M main
echo git remote add origin https://github.com/YOUR_USERNAME/flight-dynamics-simulation.git
echo git push -u origin main
echo.
echo ===============================================
echo Replace YOUR_USERNAME with your actual GitHub username!
echo ===============================================
pause
