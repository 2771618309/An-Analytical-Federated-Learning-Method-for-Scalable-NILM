@echo off
chcp 65001 >nul
title STM32 Federated Learning Monitor
cd /d "%~dp0"
echo ================================
echo STM32 Federated Learning Monitor
echo ================================
echo.
set /p PYTHON_PATH="Please enter the full path to your Python interpreter (e.g., D:\software\anaconda\envs\pytorch\python.exe): "
echo.
echo This platform requires the following libraries:
echo - streamlit
echo - pyserial
echo - pandas
echo - numpy
echo - openpyxl
echo - plotly
echo.
set /p INSTALL_DEPS="Do you want to install/update these dependencies? (y/n): "
if /i "%INSTALL_DEPS%"=="y" (
    echo.
    echo Installing dependencies...
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    echo.
    echo Dependencies installation completed!
    echo.
)
echo Starting monitoring platform...
"%PYTHON_PATH%" -m streamlit run stm32_dashboard.py
pause
