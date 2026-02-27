from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return "Spam Classifier API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data["message"]

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Ham"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)