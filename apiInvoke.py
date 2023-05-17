from flask import Flask
from flask_restful import Api
from Original_code_get import Get

server = Flask(__name__)
api = Api(server)
api.add_resource(Get, "/Original_code_get")
