from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the model
model_path = 'RF_Model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
    print(model)  # Print model summary

app = Flask(__name__)

@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form and convert to float
        float_features = [float(x) for x in request.form.values()]
        print(float_features)  # Print input features

        final_features = [np.array(float_features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Glass' if prediction[0] == 1 else 'Not Glass'
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        # Handle error and display message on webpage
        return render_template('index.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
