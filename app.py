<<<<<<< HEAD
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the model
model_path = 'RF_Model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form and convert to float
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Glass' if prediction[0] == 1 else 'Not Glass'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the model
model_path = 'RF_Model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form and convert to float
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Glass' if prediction[0] == 1 else 'Not Glass'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> cf0b68084926920e9b9cc7cff13ca65cdfc9647a
