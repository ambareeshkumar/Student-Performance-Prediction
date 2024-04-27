import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading Model in predict pipleine")
            model_path = 'src/components/artifacts/model.pkl'
            preprocessor_path = 'src/components/artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("Loading Model in predict pipleine completed")

            logging.info("Transforming features in predict pipleine")
            # logging.info("features", features)
            data_scaled = preprocessor.transform(features)
            logging.info("Transforming features in predict pipleine completed")

            logging.info("Predicting in predict pipleine")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.error(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def convert_data_into_df(self):
        try:
            logging.info("Converting UI data into Dataframe")
            df = pd.DataFrame({
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]
            })
            logging.info("Converting UI data into Dataframe completed")
            return df
        except Exception as e:
            logging.error(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
            raise CustomException(e,sys)