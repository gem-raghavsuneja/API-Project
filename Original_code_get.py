# import requests
from flask import request, make_response
from flask_restful import Resource
from Original_code import predict_elements
import json


class Get(Resource):
    @classmethod
    def get(cls):
        try:
            score, element_name, test = predict_elements()
            if score == None or element_name is None or test is None:
                msg = {"msg": "Data Not available for the given id"}
                return make_response(msg, 403)
            # return {"status": "Successful",
            #         "Data": {"Score : ": score, "Element Name : ": element_name, "Test : ": test}}
            dict1 = {"response":"Helloo"}

            dict_respone = {"score":score, "element_name": element_name}
            re =json.dumps(dict_respone)
            return re
        except Exception as e:
            return {"msg": "Exception Occured : " + e}
