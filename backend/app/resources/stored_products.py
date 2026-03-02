from datetime import date
from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import StoredProduct

ns = Namespace("stored-products", description="Stored products CRUD")

stored_model = ns.model("StoredProduct", {
    "id": fields.Integer(readOnly=True),
    "user_id": fields.Integer(required=True),
    "product_id": fields.Integer(required=True),
    "manufacture_at": fields.String(required=False, description="YYYY-MM-DD"),
    "expiry_at": fields.String(required=False, description="YYYY-MM-DD"),
    "created_at": fields.String(readOnly=True),
})

stored_create = ns.model("StoredProductCreate", {
    "user_id": fields.Integer(required=True),
    "product_id": fields.Integer(required=True),
    "manufacture_at": fields.String(required=False),
    "expiry_at": fields.String(required=False),
})

def _parse_date(s: str | None):
    if not s:
        return None
    return date.fromisoformat(s)

@ns.route("")
class StoredList(Resource):
    @ns.marshal_list_with(stored_model)
    def get(self):
        user_id = request.args.get("user_id", type=int)
        q = StoredProduct.query
        if user_id is not None:
            q = q.filter_by(user_id=user_id)
        return q.all()

    @ns.expect(stored_create, validate=True)
    @ns.marshal_with(stored_model, code=201)
    def post(self):
        data = request.json
        sp = StoredProduct(
            user_id=data["user_id"],
            product_id=data["product_id"],
            manufacture_at=_parse_date(data.get("manufacture_at")),
            expiry_at=_parse_date(data.get("expiry_at")),
        )
        db.session.add(sp)
        db.session.commit()
        return sp, 201

@ns.route("/<int:item_id>")
class StoredItem(Resource):
    @ns.marshal_with(stored_model)
    def get(self, item_id: int):
        return StoredProduct.query.get_or_404(item_id)

    @ns.expect(stored_model, validate=True)
    @ns.marshal_with(stored_model)
    def put(self, item_id: int):
        sp = StoredProduct.query.get_or_404(item_id)
        data = request.json
        if "user_id" in data:
            sp.user_id = data["user_id"]
        if "product_id" in data:
            sp.product_id = data["product_id"]
        if "manufacture_at" in data:
            sp.manufacture_at = _parse_date(data.get("manufacture_at"))
        if "expiry_at" in data:
            sp.expiry_at = _parse_date(data.get("expiry_at"))
        db.session.commit()
        return sp

    def delete(self, item_id: int):
        sp = StoredProduct.query.get_or_404(item_id)
        db.session.delete(sp)
        db.session.commit()
        return {"deleted": True}
