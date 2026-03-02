from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import User

ns = Namespace("users", description="Users CRUD")

user_model = ns.model("User", {
    "id": fields.Integer(readOnly=True),
    "email": fields.String(required=True),
    "passwordHash": fields.String(required=True),
    "isBlocked": fields.Boolean(required=False),
    "settings": fields.Raw(required=False),
})

user_create = ns.model("UserCreate", {
    "email": fields.String(required=True),
    "passwordHash": fields.String(required=True),
    "isBlocked": fields.Boolean(required=False),
    "settings": fields.Raw(required=False),
})

@ns.route("")
class UsersList(Resource):
    @ns.marshal_list_with(user_model)
    def get(self):
        return User.query.all()

    @ns.expect(user_create, validate=True)
    @ns.marshal_with(user_model, code=201)
    def post(self):
        data = request.json
        u = User(
            email=data["email"],
            password_hash=data["passwordHash"],
            is_blocked=bool(data.get("isBlocked", False)),
            settings=data.get("settings"),
        )
        db.session.add(u)
        db.session.commit()
        return u, 201

@ns.route("/<int:user_id>")
class UsersItem(Resource):
    @ns.marshal_with(user_model)
    def get(self, user_id: int):
        return User.query.get_or_404(user_id)

    @ns.expect(user_model, validate=True)
    @ns.marshal_with(user_model)
    def put(self, user_id: int):
        u = User.query.get_or_404(user_id)
        data = request.json
        if "email" in data:
            u.email = data["email"]
        if "passwordHash" in data:
            u.password_hash = data["passwordHash"]
        if "isBlocked" in data:
            u.is_blocked = bool(data["isBlocked"])
        if "settings" in data:
            u.settings = data["settings"]
        db.session.commit()
        return u

    def delete(self, user_id: int):
        u = User.query.get_or_404(user_id)
        db.session.delete(u)
        db.session.commit()
        return {"deleted": True}
