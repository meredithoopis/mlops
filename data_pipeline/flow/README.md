Run as follows:  

```bash
chmod +x run.sh  
./run.sh
```

Current problem: 
Database airflow (Save logs, info): docker compose exec postgres psql -U airflow -d airflow 
Saving images and labels: From Pgadmin, create a db named carrrr, give permissions for airflow user (password airflow)
docker-compose.yaml: I should remove redis image 
Remember: Add delete db script before running the script  