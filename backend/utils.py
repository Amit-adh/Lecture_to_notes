from flask import request, jsonify
import jwt

from functools import wraps

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        pass