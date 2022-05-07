from flask import Flask, jsonify, request
from deploy.torch_utils import get_top_10

# from app.torch_utils import transform_image, get_prediction  # production

app = Flask(__name__)

@app.route('/')
def hello():
    result = 'Hello there'
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')

        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})

        try:
            img_bytes = file.read()
            top_10 = get_top_10(img_bytes)
            result = {'top10': top_10}
            return jsonify(result)
        except:
            return jsonify({'error': 'error during prediction'})
