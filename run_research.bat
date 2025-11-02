@echo off
echo.
echo Running Indonesian Public University Tuition Fees Prediction Research...
echo.

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run the main research script
python src/main.py

echo.
echo Research execution completed!
echo.
pause