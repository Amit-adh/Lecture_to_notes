from flask import Blueprint, request

auth = Blueprint("auth", __name__)

@auth.route("/login", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        pass
        # username = 
