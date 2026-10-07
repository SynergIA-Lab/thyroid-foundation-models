@echo off
REM Script para crear y activar un entorno virtual con Anaconda (Windows)

set ENV_NAME=thyroid_env
cd /d "%~dp0"

echo === Creando entorno virtual (conda) a partir de environment.yml ===
call conda env create -f environment.yml

echo.
echo === Entorno virtual '%ENV_NAME%' creado correctamente! ===
echo Para activarlo en el futuro, ejecuta:
echo conda activate %ENV_NAME%
pause
