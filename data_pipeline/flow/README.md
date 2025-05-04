# Pipeline to process data
A Python script to automate the preprocessing of the training data. 

## ⚙️ How-to
Running the script will create containers for Airflow which contains two DAG files, one for processing training data and another for automate crawling data. 
```bash
chmod +x run.sh  
./run.sh
```


Current problem: 
Database airflow (Save logs, info): docker compose exec postgres psql -U airflow -d airflow 
Saving images and labels: From Pgadmin, create a db named carrrr, give permissions for airflow user (password airflow)
Remember: Add delete db script before running the script  
To-do: In crawl_dag, find ways to connect to envs(API, db) outside of the file 