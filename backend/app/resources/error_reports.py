from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import ErrorReport

ns = Namespace("error-reports", description="Error reports CRUD")

report_model = ns.model("ErrorReport", {
    "id": fields.Integer(readOnly=True),
    "user_id": fields.Integer(required=True),
    "video_id": fields.Integer(required=True),
    "model_version_id": fields.Integer(required=False),
    "comment": fields.String(required=False),
    "created_at": fields.String(readOnly=True),
    "resolved": fields.Boolean(required=False),
})

report_create = ns.model("ErrorReportCreate", {
    "user_id": fields.Integer(required=True),
    "video_id": fields.Integer(required=True),
    "model_version_id": fields.Integer(required=False),
    "comment": fields.String(required=False),
    "resolved": fields.Boolean(required=False),
})

@ns.route("")
class ReportList(Resource):
    @ns.marshal_list_with(report_model)
    def get(self):
        return ErrorReport.query.order_by(ErrorReport.created_at.desc()).all()

    @ns.expect(report_create, validate=True)
    @ns.marshal_with(report_model, code=201)
    def post(self):
        data = request.json
        r = ErrorReport(
            user_id=data["user_id"],
            video_id=data["video_id"],
            model_version_id=data.get("model_version_id"),
            comment=data.get("comment"),
            resolved=bool(data.get("resolved", False)),
        )
        db.session.add(r)
        db.session.commit()
        return r, 201

@ns.route("/<int:report_id>")
class ReportItem(Resource):
    @ns.marshal_with(report_model)
    def get(self, report_id: int):
        return ErrorReport.query.get_or_404(report_id)

    @ns.expect(report_model, validate=True)
    @ns.marshal_with(report_model)
    def put(self, report_id: int):
        r = ErrorReport.query.get_or_404(report_id)
        data = request.json
        if "user_id" in data:
            r.user_id = data["user_id"]
        if "video_id" in data:
            r.video_id = data["video_id"]
        if "model_version_id" in data:
            r.model_version_id = data["model_version_id"]
        if "comment" in data:
            r.comment = data["comment"]
        if "resolved" in data:
            r.resolved = bool(data["resolved"])
        db.session.commit()
        return r

    def delete(self, report_id: int):
        r = ErrorReport.query.get_or_404(report_id)
        db.session.delete(r)
        db.session.commit()
        return {"deleted": True}
