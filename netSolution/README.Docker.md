# Запуск backend через Docker

Эти файлы поднимают только backend `ProductsDateAPI` и PostgreSQL. Админка WinForms в контейнер не собирается.

## 1. Подготовить сервер

На сервере должны быть установлены Docker и Docker Compose v2.

## 2. Настроить переменные

В корне `netSolution`:

```bash
cp .env.example .env
nano .env
```

Обязательно поменяй:

- `POSTGRES_PASSWORD`
- `JWT_SECRET`
- `TRAINING_CLIENT_API_KEY`

`TRAINING_CLIENT_API_KEY` должен быть таким же, как переменная `TRAINING_CLIENT_API_KEY` у Python-клиента обучения.

## 3. Запустить

```bash
docker compose up -d --build
```

API будет доступен на порту из `.env`, по умолчанию:

```text
http://SERVER_IP:5099
```

Swagger, если `SWAGGER_ENABLED=true`:

```text
http://SERVER_IP:5099/swagger
```

## 4. Проверить логи

```bash
docker compose logs -f api
```

```bash
docker compose logs -f postgres
```

## 5. Миграции базы

По умолчанию в `.env` стоит:

```text
APPLY_MIGRATIONS=true
```

Тогда API при старте сам выполнит `db.Database.Migrate()` и применит EF Core миграции.

После первого успешного запуска можно поставить:

```text
APPLY_MIGRATIONS=false
```

и перезапустить:

```bash
docker compose up -d
```

## 6. Где хранятся данные

Docker volumes:

- `postgres_data` — база PostgreSQL
- `api_storage` — модели, датасеты задач обучения, артефакты
- `api_uploads` — загруженные датасеты/видео из error reports
- `api_temp` — временные файлы упаковки датасета

Важно: не удаляй volumes, если не хочешь потерять данные.

## 7. Seed dataset

Если нужен базовый датасет для обучения, положи его рядом с `docker-compose.yml` в папку:

```text
dataset/
```

Ожидаемая структура:

```text
dataset/images/*.jpg|*.jpeg|*.png
dataset/labels/*.txt
```

Контейнер видит эту папку как `/app/dataset`.

## 8. Python-клиент обучения

На компьютере с GPU укажи адрес backend:

```bash
export PRODUCTS_DATE_BACKEND_URL="http://SERVER_IP:5099"
export TRAINING_CLIENT_API_KEY="тот_же_ключ_что_в_.env"
export TRAINING_CLIENT_ID="gpu-pc-1"
python app.py worker
```

## 9. Полезные команды

Остановить:

```bash
docker compose down
```

Остановить и удалить данные БД/моделей:

```bash
docker compose down -v
```

Пересобрать только API:

```bash
docker compose build api
docker compose up -d api
```

## Backup / restore backend data

Backend has admin endpoints for exporting and importing a full ZIP backup.
The ZIP contains:

- `manifest.json` — backup metadata;
- `database.json` — users, products, stored products, error reports, video samples, model versions, training jobs;
- `storage/training/**` — trained model files and training job datasets/artifacts;
- `uploads/**` — uploaded error-report datasets.

Get an admin JWT token first through `/api/Auth/login`, then export:

```bash
curl -L \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -o productsdate_backup.zip \
  "http://SERVER_IP:5099/api/admin/backup/export"
```

Import into an empty database:

```bash
curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "BackupZip=@productsdate_backup.zip" \
  "http://SERVER_IP:5099/api/admin/backup/import"
```

Import with replacement of existing data and files:

```bash
curl -X POST \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "BackupZip=@productsdate_backup.zip" \
  -F "ReplaceExisting=true" \
  "http://SERVER_IP:5099/api/admin/backup/import"
```

`ReplaceExisting=true` deletes current database rows plus current `/app/storage/training` and `/app/uploads` files before restoring from the ZIP.
