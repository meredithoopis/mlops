#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./dags ./logs ./plugins ./config

if [ ! -f ./dags/car_dag.py ]; then
    cp car_dag.py ./dags/
    echo "Copied car_dag.py to ./dags/"
else
    echo "car_dag.py already exists in ./dags/, skipping copy."
fi

if [ ! -f .env ]; then
    echo "AIRFLOW_UID=$(id -u)" > .env
    echo ".env file created with AIRFLOW_UID."
else
    echo ".env file already exists, skipping."
fi

docker compose up 
