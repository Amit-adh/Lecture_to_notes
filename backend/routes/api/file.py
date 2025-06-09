from flask import Blueprint, request, jsonify

file_bp = Blueprint('file', __name__)

@file_bp.route('/file', methods=['POST'])
def file():
    file = request.files['file']
    print(file.content_type)
    