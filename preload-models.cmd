@echo off
cd /d %~dp0
call "%~dp0runtime.cmd" python -s download_models.py
pause
