from flask import Flask
from sqlalchemy_utils import database_exists, create_database

from app.config import Config
from app.extensions import db, migrate
from app.api import api, register_namespaces

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)

    api.init_app(app)
    register_namespaces(api)

    # Создание БД и таблиц (аналог EF "create database")
    with app.app_context():
        uri = app.config["SQLALCHEMY_DATABASE_URI"]
        if not database_exists(uri):
            create_database(uri)
        db.create_all()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
