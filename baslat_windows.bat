@echo off
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo Sanal ortam bulunamadi, sistem python kullaniliyor.
    set "PY=python"
)

echo ==========================================
echo   Beyin BT Siniflandirma - Streamlit
echo ==========================================
echo Python: %PY%
echo.

"%PY%" -m streamlit run src/arayuz/arayuz_app.py

pause
