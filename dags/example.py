import logging

from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DAG_ID = "mlops"

logging.basicConfig(filename="my_first_dag.log", level=logging.INFO)
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


args = {
    "owner": "Ruslan Sabirov",
    "email": ["frankdiyk@gmail.com"],
}

dag = DAG(
    dag_id=DAG_ID,
    default_args=args,
    max_active_runs=1,
    concurrency=3,
    schedule_interval="0 4 * * *",
    start_date=days_ago(1),
    tags=["mlops"],
)

def download_data() -> None:
    import pandas as pd
    
    from sklearn import datasets
    
    wine_data = datasets.load_wine()
    
    X = pd.DataFrame(wine_data['data'], columns = wine_data['feature_names'])
    y = wine_data['target']
    
    X.to_csv('features.csv')

task_download_data = PythonOperator(task_id="task_download_data",
                                 python_callable=download_data,
                                 dag=dag)

task_download_data  
