import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self, config:DataTransformerConfig):
        self.config = config

    def get_data_transformer(self):
        try:
            numerical_features = ["writing_score","reading_score"]
            categorical_features = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"
                                    ]

            logging.info("Entered numerical pipeline")
            numerical_pipeline = Pipeline(
                steps = [
                    ("Impute", SimpleImputer(strategy = "median")),
                    ("Scaling", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical Pipeline Completed")

            logging.info("Entered categorical pipeline")
            categorical_pipeline = Pipeline(
                steps = [
                    ("Impute", SimpleImputer(strategy = "most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("Scaling", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical Pipeline Completed")

            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
            raise CustomException(e,sys)

    def initiate_data_transform(self,train_path,test_path):
        try:
            logging.info(f"Importing train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Imported train and test data successfully")

            logging.info(f"Obtaining preprocessor")
            preprocessor_obj = self.get_data_transformer()

            target_col = 'math_score'

            logging.info(f"Splitting Independent and dependent columns for train and test")
            input_feature_train = train_df.drop(target_col, axis = 1)
            target_feature_train = train_df[target_col]

            input_feature_test = test_df.drop(target_col, axis = 1)
            target_feature_test = test_df[target_col]

            preprocess_train = preprocessor_obj.fit_transform(input_feature_train)
            preprocess_test = preprocessor_obj.transform(input_feature_test)

            train_arr = np.c_[preprocess_train, np.array(target_feature_train)]
            test_arr = np.c_[preprocess_test, np.array(target_feature_test)]

            save_object(
                file_path = self.config.preprocessor_obj_file_path,
                object = preprocessor_obj
            )

            logging.info("Data Transformation completed")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
            raise CustomException(e,sys)