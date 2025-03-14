from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    data = request.json  # Get JSON data from the request body
    return jsonify({"message": "Hello, World!", "received_data": data})

@app.route('/about', methods=['POST'])
def about():
    data = request.json  # Get JSON data from the request body
    return jsonify({"message": "About Page", "received_data": data})
