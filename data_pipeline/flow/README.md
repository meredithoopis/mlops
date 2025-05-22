# Pipeline to process data
A Python script to automate the preprocessing of the training data, ctawl new data, and training the model. 

## ⚙️ How-to
Create an .env file in this folder and put all your API Key as well as necessary configurations (as illustrated in the crawl_test folder). 
Running the following script will create containers for Airflow which contains 3 DAG files, one for processing training data, one for crawling the data, and another for training the model. 
```bash
chmod +x run.sh  
./run.sh
```

