from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('iris_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
