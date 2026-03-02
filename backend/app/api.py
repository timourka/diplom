from flask_restx import Api

api = Api(
    title="Diplom API",
    version="0.1",
    description="CRUD API (Flask + Postgres + Swagger)",
    doc="/docs",
)

def register_namespaces(api: Api):
    from app.resources.users import ns as users_ns
    from app.resources.products import ns as products_ns
    from app.resources.stored_products import ns as stored_products_ns
    from app.resources.videos import ns as videos_ns
    from app.resources.model_versions import ns as model_versions_ns
    from app.resources.error_reports import ns as error_reports_ns

    api.add_namespace(users_ns, path="/users")
    api.add_namespace(products_ns, path="/products")
    api.add_namespace(stored_products_ns, path="/stored-products")
    api.add_namespace(videos_ns, path="/videos")
    api.add_namespace(model_versions_ns, path="/model-versions")
    api.add_namespace(error_reports_ns, path="/error-reports")
