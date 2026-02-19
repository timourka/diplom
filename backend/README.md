# Flask + Postgres + Swagger (CRUD)

## Запуск (локально)
```bash
cd backend
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
python run.py
```

Swagger UI: http://localhost:5000/docs  
Health: http://localhost:5000/health  

## Запуск через Docker Compose (опционально)
```bash
docker compose up --build
```
Swagger UI: http://localhost:5000/docs  
