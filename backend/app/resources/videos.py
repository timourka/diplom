from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import VideoSample

ns = Namespace("videos", description="Video samples CRUD")

video_model = ns.model("VideoSample", {
    "id": fields.Integer(readOnly=True),
    "video_path": fields.String(required=True),
    "source": fields.String(required=False, description="camera/file/test"),
})

video_create = ns.model("VideoSampleCreate", {
    "video_path": fields.String(required=True),
    "source": fields.String(required=False),
})

@ns.route("")
class VideoList(Resource):
    @ns.marshal_list_with(video_model)
    def get(self):
        return VideoSample.query.all()

    @ns.expect(video_create, validate=True)
    @ns.marshal_with(video_model, code=201)
    def post(self):
        data = request.json
        v = VideoSample(video_path=data["video_path"], source=data.get("source"))
        db.session.add(v)
        db.session.commit()
        return v, 201

@ns.route("/<int:video_id>")
class VideoItem(Resource):
    @ns.marshal_with(video_model)
    def get(self, video_id: int):
        return VideoSample.query.get_or_404(video_id)

    @ns.expect(video_model, validate=True)
    @ns.marshal_with(video_model)
    def put(self, video_id: int):
        v = VideoSample.query.get_or_404(video_id)
        data = request.json
        if "video_path" in data:
            v.video_path = data["video_path"]
        if "source" in data:
            v.source = data["source"]
        db.session.commit()
        return v

    def delete(self, video_id: int):
        v = VideoSample.query.get_or_404(video_id)
        db.session.delete(v)
        db.session.commit()
        return {"deleted": True}
