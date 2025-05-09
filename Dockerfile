FROM apache/airflow:2.9.3-python3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt