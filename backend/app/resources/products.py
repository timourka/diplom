from flask import request
from flask_restx import Namespace, Resource, fields
from app.extensions import db
from app.models import Product

ns = Namespace("products", description="Products CRUD")

product_model = ns.model("Product", {
    "id": fields.Integer(readOnly=True),
    "name": fields.String(required=True),
    "manufacturer": fields.String(required=False),
    "barcode": fields.String(required=False),
})

product_create = ns.model("ProductCreate", {
    "name": fields.String(required=True),
    "manufacturer": fields.String(required=False),
    "barcode": fields.String(required=False),
})

@ns.route("")
class ProductList(Resource):
    @ns.marshal_list_with(product_model)
    def get(self):
        return Product.query.all()

    @ns.expect(product_create, validate=True)
    @ns.marshal_with(product_model, code=201)
    def post(self):
        data = request.json
        p = Product(
            name=data["name"],
            manufacturer=data.get("manufacturer"),
            barcode=data.get("barcode"),
        )
        db.session.add(p)
        db.session.commit()
        return p, 201

@ns.route("/<int:product_id>")
class ProductItem(Resource):
    @ns.marshal_with(product_model)
    def get(self, product_id: int):
        return Product.query.get_or_404(product_id)

    @ns.expect(product_model, validate=True)
    @ns.marshal_with(product_model)
    def put(self, product_id: int):
        p = Product.query.get_or_404(product_id)
        data = request.json
        if "name" in data:
            p.name = data["name"]
        if "manufacturer" in data:
            p.manufacturer = data["manufacturer"]
        if "barcode" in data:
            p.barcode = data["barcode"]
        db.session.commit()
        return p

    def delete(self, product_id: int):
        p = Product.query.get_or_404(product_id)
        db.session.delete(p)
        db.session.commit()
        return {"deleted": True}
