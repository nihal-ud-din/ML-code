from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    return ""

if __name__ == '__main__':
    app.run(debug=True, port=5001)