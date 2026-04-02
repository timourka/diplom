# training_service

Отдельный HTTP-сервис для обучения модели на другой машине.

## Что исправлено

- список job доступен через `GET /jobs`;
- `device=auto` автоматически выбирает GPU, если `torch.cuda.is_available() == True`, иначе падает на `cpu` без ошибки `Invalid CUDA device=0`;
- дефолтные параметры обучения оставлены близкими к исходному `train_expiry_all.py`.

## Запуск

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Переменные окружения

```env
TRAINING_SERVICE_API_KEY=your_secret_key
TRAINING_SERVICE_DATA=./data
```
