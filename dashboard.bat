@echo off
REM Dashboarda abiarazi
REM Nabigatzailean irekiko da (http://localhost:8501). Ixteko itxi cmd hau.

cd /d "%~dp0"

REM Python aukeratu: .venv miniconda3 edo anaconda3 badaude haiek bestela python edo py
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto :prest
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PY=%USERPROFILE%\miniconda3\python.exe"
    goto :prest
)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PY=%USERPROFILE%\anaconda3\python.exe"
    goto :prest
)
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=py"

:prest
REM Liburutegiak falta badira instalatu
"%PY%" -c "import streamlit, altair, pandas, numpy" >nul 2>nul
if errorlevel 1 (
    echo Liburutegiak instalatzen, itxaron...
    "%PY%" -m pip install streamlit altair pandas numpy
)

"%PY%" -m streamlit run dashboard\app.py
pause
