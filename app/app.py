from flask import Flask, request, jsonify
from predictor import load_model, get_prediction
from io import BytesIO

app = Flask(__name__)
model = load_model()

@app.route("/")
def index():
    return "🚀 Image Classifier API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    try:
        prediction = get_prediction(BytesIO(image.read()), model)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
