from datetime import datetime
from app.extensions import db

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_blocked = db.Column(db.Boolean, nullable=False, default=False)
    settings = db.Column(db.JSON, nullable=True)

    stored_products = db.relationship("StoredProduct", back_populates="user", cascade="all, delete-orphan")
    error_reports = db.relationship("ErrorReport", back_populates="user", cascade="all, delete-orphan")


class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    manufacturer = db.Column(db.String(255), nullable=True)
    barcode = db.Column(db.String(128), nullable=True, index=True)

    stored_products = db.relationship("StoredProduct", back_populates="product", cascade="all, delete-orphan")


class StoredProduct(db.Model):
    __tablename__ = "stored_products"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)

    manufacture_at = db.Column(db.Date, nullable=True)
    expiry_at = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="stored_products")
    product = db.relationship("Product", back_populates="stored_products")


class VideoSample(db.Model):
    __tablename__ = "video_samples"

    id = db.Column(db.Integer, primary_key=True)
    video_path = db.Column(db.String(1024), nullable=False)
    source = db.Column(db.String(32), nullable=True)  # camera / file / test

    error_reports = db.relationship("ErrorReport", back_populates="video", cascade="all, delete-orphan")


class ModelVersion(db.Model):
    __tablename__ = "model_versions"

    id = db.Column(db.Integer, primary_key=True)
    trained_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    metrics = db.Column(db.Text, nullable=True)

    error_reports = db.relationship("ErrorReport", back_populates="model_version", cascade="all, delete-orphan")


class ErrorReport(db.Model):
    __tablename__ = "error_reports"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = db.Column(db.Integer, db.ForeignKey("video_samples.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True)

    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    resolved = db.Column(db.Boolean, nullable=False, default=False)

    user = db.relationship("User", back_populates="error_reports")
    video = db.relationship("VideoSample", back_populates="error_reports")
    model_version = db.relationship("ModelVersion", back_populates="error_reports")
