from flask import Flask, jsonify, request
import predictor_api
import requests
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# to run this FLASK_ENV=development FLASK_APP=flask1.py flask run
@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		data = json.loads(request.data)
		url = data[0]["payload"]["url"]
		image = requests.get(url)
		img = Image.open(BytesIO(image.content))
		print(type(img))
		print(type(image))
		print(url)
		print(image)
		print(img)
		result = predictor_api.make_prediction(url)
	return jsonify({'result': result})






