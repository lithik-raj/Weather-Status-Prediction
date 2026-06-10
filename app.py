from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        rainfall = float(request.form['rainfall'])

        prediction = model.predict(
            [[temperature, humidity, windspeed, rainfall]]
        )[0]

        # Read dataset
        df = pd.read_csv('data.csv')

        counts = df['WeatherCondition'].value_counts()

        # Highlight predicted weather
        colors = []

        for weather in counts.index:
            if weather == prediction:
                colors.append('green')
            else:
                colors.append('gray')

        plt.figure(figsize=(8, 5))

        plt.bar(
            counts.index,
            counts.values,
            color=colors
        )

        plt.title(
            f'Weather Distribution (Prediction: {prediction})'
        )

        plt.xlabel('Weather Condition')
        plt.ylabel('Count')

        plt.tight_layout()

        chart_path = 'static/chart.png'

        plt.savefig(chart_path)
        plt.close()

        return render_template(
            'result.html',
            prediction=prediction,
            chart=chart_path
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)