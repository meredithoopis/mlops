#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./dags ./logs ./plugins ./config ./mlruns
FILES=("car_dag.py" "crawl_dag.py" "model_dag.py")

for FILE in "${FILES[@]}"; do
    if [ ! -f "./dags/$FILE" ]; then
        cp "$FILE" ./dags/
        echo "Copied $FILE to ./dags/"
    else
        echo "$FILE already exists in ./dags/, skipping copy."
    fi
done



# if [ ! -f .env ]; then
#     echo "AIRFLOW_UID=$(id -u)" > .env
#     echo ".env file created with AIRFLOW_UID."
# else
#     echo ".env file already exists, skipping."
# fi

if [ ! -f .env ]; then
    echo "AIRFLOW_UID=$(id -u)" > .env
    echo ".env file created with AIRFLOW_UID."
else
    if ! grep -q "^AIRFLOW_UID=" .env; then
        echo "AIRFLOW_UID=$(id -u)" >> .env
        echo "Appended AIRFLOW_UID to existing .env file."
    else
        echo "AIRFLOW_UID already set in .env, skipping."
    fi
fi

docker compose up 
