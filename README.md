1. На хосте создать airflow/logs от пользователя airflow, либо выдать права, т.к. при использовании docker compose папка создается, но без прав на использование.
2. В минио создать AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY и добавить эти значения в dockerfile
3. Для запуска airflow дагов, нужно добавить конекшены для s3 с учетными данными и extra { "endpoint_url":"адрес минио", "region_name":"ввести регион"}
4. В aiflow vars добавить варсы с ключами и адресами mlflow и s3 AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_ENDPOINT_URL, BUCKET, MLFLOW_TRACKING_URI
5. В dockerfile в команд mlflow-service изменить default-artifact-root с вагим созданным бакутом из минио
