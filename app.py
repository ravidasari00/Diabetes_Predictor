from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        # Prepare input data for prediction
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        
        # Make predictions and get probabilities
        my_prediction = classifier.predict(data)[0]
        probabilities = classifier.predict_proba(data)
        probability = round(max(probabilities[0]) * 100, 2)  # Assuming binary classification

        return render_template('result.html', prediction=my_prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
