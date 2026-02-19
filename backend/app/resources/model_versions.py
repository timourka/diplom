from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import ModelVersion

ns = Namespace("model-versions", description="Model versions CRUD")

mv_model = ns.model("ModelVersion", {
    "id": fields.Integer(readOnly=True),
    "trained_at": fields.String(readOnly=True),
    "metrics": fields.String(required=False),
})

mv_create = ns.model("ModelVersionCreate", {
    "metrics": fields.String(required=False),
})

@ns.route("")
class MVList(Resource):
    @ns.marshal_list_with(mv_model)
    def get(self):
        return ModelVersion.query.all()

    @ns.expect(mv_create, validate=True)
    @ns.marshal_with(mv_model, code=201)
    def post(self):
        data = request.json or {}
        mv = ModelVersion(metrics=data.get("metrics"))
        db.session.add(mv)
        db.session.commit()
        return mv, 201

@ns.route("/<int:mv_id>")
class MVItem(Resource):
    @ns.marshal_with(mv_model)
    def get(self, mv_id: int):
        return ModelVersion.query.get_or_404(mv_id)

    @ns.expect(mv_model, validate=True)
    @ns.marshal_with(mv_model)
    def put(self, mv_id: int):
        mv = ModelVersion.query.get_or_404(mv_id)
        data = request.json or {}
        if "metrics" in data:
            mv.metrics = data["metrics"]
        db.session.commit()
        return mv

    def delete(self, mv_id: int):
        mv = ModelVersion.query.get_or_404(mv_id)
        db.session.delete(mv)
        db.session.commit()
        return {"deleted": True}
