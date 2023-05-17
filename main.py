# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask

# app = Flask(__name__)
from apiInvoke import server


@server.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    server.run(debug=False, host="127.0.0.1", port=50000)
