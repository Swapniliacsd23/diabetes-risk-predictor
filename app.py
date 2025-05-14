from flask import Flask, render_template, request, url_for
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    graph_url = None

    if request.method == 'POST':
        try:
            # Safely retrieve form values
            pregnancies = float(request.form.get('Pregnancies', 0))
            glucose = float(request.form.get('Glucose', 0))
            bloodpressure = float(request.form.get('BloodPressure', 0))
            skinthickness = float(request.form.get('SkinThickness', 0))
            insulin = float(request.form.get('Insulin', 0))
            bmi = float(request.form.get('BMI', 0))
            dpf = float(request.form.get('DiabetesPedigreeFunction', 0))
            age = float(request.form.get('Age', 0))

            # Scale the input and predict
            input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            result = model.predict(input_scaled)

            prediction = 'High Risk of Diabetes' if result[0] == 1 else 'Low Risk of Diabetes'

            # Create graph and save
            plt.figure(figsize=(5, 5))
            plt.scatter([glucose], [age], c='red', s=100)
            plt.xlabel('Glucose')
            plt.ylabel('Age')
            plt.title('Glucose vs Age')
            plt.grid(True)

            graph_path = os.path.join('static', 'graph.png')
            plt.savefig(graph_path)
            plt.close()

            graph_url = url_for('static', filename='graph.png')

        except Exception as e:
            prediction = f"Error in processing input: {str(e)}"

    return render_template('index.html', prediction=prediction, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)

