# training_service теперь работает как клиент

Этот Python-процесс больше не поднимает HTTP-сервер и не требует статический IP.
Он сам опрашивает backend, забирает queued-задачи обучения, скачивает датасет с backend, обучает модель локально и загружает `best.pt` + мобильный артефакт обратно на backend.

## Схема

1. Админ в .NET AdminApp нажимает «Запустить обучение».
2. Backend сохраняет zip-датасет у себя и создаёт задачу `queued` в БД.
3. Python-клиент с компьютера для обучения делает исходящий запрос к backend: `GET /api/training-client/jobs/next`.
4. Клиент скачивает датасет, обучает YOLO, экспортирует mobile model.
5. Клиент загружает артефакты в backend: `POST /api/training-client/jobs/{jobId}/artifacts`.
6. Backend сохраняет файлы модели у себя и создаёт новую версию модели.
7. Админ выбирает, какую версию опубликовать для мобильных пользователей.
8. Мобильное приложение скачивает только опубликованную модель с backend: `/api/mobile-models/latest/download`.

## Запуск

```bash
pip install -r requirements.txt

# Windows PowerShell пример:
$env:PRODUCTS_DATE_BACKEND_URL="http://YOUR_BACKEND_HOST:5099"
$env:TRAINING_CLIENT_API_KEY="super_secret_training_key_2026"
$env:TRAINING_CLIENT_ID="gpu-pc-1"
python app.py worker
```

Один проход без бесконечного цикла:

```bash
python app.py once
```

## Переменные окружения

```env
PRODUCTS_DATE_BACKEND_URL=http://127.0.0.1:5099
TRAINING_CLIENT_API_KEY=super_secret_training_key_2026
TRAINING_CLIENT_ID=gpu-pc-1
TRAINING_CLIENT_DATA=./data
TRAINING_CLIENT_POLL_SECONDS=15
TRAINING_CLIENT_REQUEST_TIMEOUT=120
```

`TRAINING_CLIENT_API_KEY` должен совпадать с `TrainingService:ApiKey` в `ProductsDateAPI/appsettings.json`.
