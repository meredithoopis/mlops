# Pipeline to process training data 
A Python script to automate the preprocessing of the training data. 

## ⚙️ How-to
Running the script 
```bash
chmod +x run.sh  
./run.sh
```


Current problem: 
Database airflow (Save logs, info): docker compose exec postgres psql -U airflow -d airflow 
Saving images and labels: From Pgadmin, create a db named carrrr, give permissions for airflow user (password airflow)
docker-compose.yaml: I should remove redis image 
Remember: Add delete db script before running the script  
To-do: Try querying the db 