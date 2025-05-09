import logging

from typing import Dict, Any

from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


DAG_ID = "train_multi_models"

models = dict(zip(["RandomForest", "LinearRegression", "HistGB"], 
                  [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))


logging.basicConfig(filename="my_forth_dag.log", level=logging.INFO)
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

args = {
    "owner": "Ruslan Sabirov",
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
    import io
    import pandas as pd
    
    from sklearn import datasets
    
    # Получим датасет California housing
    housing = datasets.fetch_california_housing(as_frame=True)
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    # Сохраняем данные в буффер
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохранить файл в формате pkl на S3
    BUCKET = Variable.get("BUCKET")
    s3_hook = S3Hook("s3_connection")
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="2025/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info("Data downloaded.")
    

def train_model(**kwargs) -> Dict[str, Any]:
    import pandas as pd
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

    # Чтение параметра
    ti = kwargs["ti"]
    model_name  = kwargs["model_name"]

    # Использовать созданный ранее S3 connection
    s3_hook = S3Hook("s3_connection")
    BUCKET = Variable.get("BUCKET")
    file = s3_hook.download_file(key=f"2025/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Обучить модель
    model = models[model_name]
    model.fit(X_train_fitted, y_train)
    y_pred = model.predict(X_test_fitted)

    metrics = {}
    metrics[f"{model_name}_r_squared"] = r2_score(y_test, y_pred)
    metrics[f"{model_name}_RMSE"] = mean_squared_error(y_test, y_pred)**0.5
    metrics[f"{model_name}_MAE"] = median_absolute_error(y_test, y_pred)

    return metrics
    

def save_results(**kwargs) -> None:
    import io
    import json

    ti = kwargs["ti"]
    models_metrics = ti.xcom_pull(
        task_ids=[f"task_train_model_{model_name}" for model_name in models.keys()]
    )
    result = {}
    for model_metrics in models_metrics:
        result.update(model_metrics)

    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(result).encode())
    filebuffer.seek(0)

    BUCKET = Variable.get("BUCKET")
    s3_hook = S3Hook("s3_connection")
    s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"2025/multi_model/metrics/metrics.json",
            bucket_name=BUCKET,
            replace=True,
        )
    

task_download_data = PythonOperator(task_id="task_download_data",
                                 python_callable=download_data,
                                 dag=dag)
training_model_tasks = [PythonOperator(task_id=f"task_train_model_{model_name}",
                                 python_callable=train_model,
                                 dag=dag, provide_context=True, op_kwargs={"model_name": model_name}) for model_name in models.keys()]

task_save_results = PythonOperator(task_id="task_save_results",
                                 python_callable=save_results,
                                 dag=dag, provide_context=True)

task_download_data >> training_model_tasks >> task_save_results
