@echo off
chcp 65001 >nul
title STM32 Federated Learning Monitor
cd /d "%~dp0"
D:\software\anaconda\envs\pytorch\python.exe -m streamlit run stm32_dashboard.py
pause
