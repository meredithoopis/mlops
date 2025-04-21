#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./dags ./logs ./plugins ./config
cp car_dag.py ./dags/ 

echo "AIRFLOW_UID=$(id -u)" > .env

docker compose up --detach

