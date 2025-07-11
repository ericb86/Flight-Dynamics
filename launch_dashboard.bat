@echo off
cd /d "%~dp0"
call flight-env\Scripts\activate
streamlit run dashboard.py
pause