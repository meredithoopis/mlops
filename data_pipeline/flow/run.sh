#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./dags ./logs ./plugins ./config

FILES=("car_dag.py" "crawl_dag.py")

for FILE in "${FILES[@]}"; do
    if [ ! -f "./dags/$FILE" ]; then
        cp "$FILE" ./dags/
        echo "Copied $FILE to ./dags/"
    else
        echo "$FILE already exists in ./dags/, skipping copy."
    fi
done



if [ ! -f .env ]; then
    echo "AIRFLOW_UID=$(id -u)" > .env
    echo ".env file created with AIRFLOW_UID."
else
    echo ".env file already exists, skipping."
fi

docker compose up 
