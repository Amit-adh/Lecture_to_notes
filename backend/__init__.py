from flask import Flask, request, Blueprint, jsonify
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder='..frontend/templates', static_folder='static')
    db.init_app(app)
    
    return app