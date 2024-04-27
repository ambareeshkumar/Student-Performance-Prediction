from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            writing_score = float(request.form.get('writing_score')),
            reading_score = float(request.form.get('reading_score'))
        )
        pred_df = data.convert_data_into_df()
        logging.info(f'Prediction Dataframe in {pred_df}')

        logging.info(f"Initializing PredictPipeline to predict Final result")
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        logging.info(f"predicted the final result{result}")
        result_formatted = "{:.1f}".format(result[0])
        return render_template('home.html',result = result_formatted)

if __name__ == '__main__':
    application.run(host = "0.0.0.0")

