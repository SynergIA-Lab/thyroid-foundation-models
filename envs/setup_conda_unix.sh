#!/bin/bash
# Script para crear y activar un entorno virtual con Anaconda (macOS/Ubuntu)

ENV_NAME="thyroid_env"
cd "$(dirname "$0")"

echo "=== Creando entorno virtual (conda) a partir de environment.yml ==="
conda env create -f environment.yml

echo "=== Entorno virtual '$ENV_NAME' creado correctamente! ==="
echo "Para activarlo en el futuro, ejecuta:"
echo "conda activate $ENV_NAME"
