import os
import logging

from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

DAG_ID = "train_dag_with_mlflow"


logging.basicConfig(filename="my_third_dag.log", level=logging.INFO)
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

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

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
    

def train_model() -> None:
    import mlflow
    import pandas as pd

    from mlflow.models import infer_signature
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    configure_mlflow()
    
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
    mlflow.set_experiment(experiment_name="MedHouseExp")
    with mlflow.start_run(run_name="my_third_run", experiment_id = "135293466297753618"):
        # Обучить модель
        model = LinearRegression()
        model.fit(X_train_fitted, y_train)
        y_pred = model.predict(X_test_fitted)
    
        # Получить описание данных
        signature = infer_signature(X_test_fitted, y_pred)
        # Сохранить модель в артифактори
        model_info = mlflow.sklearn.log_model(model, "MedHouseExp_airflow", signature=signature)
        # Сохранить метрики модели
        mlflow.evaluate(
            model_info.model_uri,
            data=X_test_fitted,
            targets=y_test.values,
            model_type="regressor",
            evaluators=["default"],
        )

    

task_download_data = PythonOperator(task_id="task_download_data",
                                 python_callable=download_data,
                                 dag=dag)

task_train_model = PythonOperator(task_id="task_train_model",
                                 python_callable=train_model,
                                 dag=dag, provide_context=True)

task_download_data >> task_train_model 
