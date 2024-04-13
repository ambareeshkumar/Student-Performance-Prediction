import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
AdaBoostRegressor,
GradientBoostingRegressor,
RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    modelTrainer_path = os.path.join('artifacts','model.py')

class ModelTrainer:

    try:
        def __init__(self, config:ModelTrainerConfig):
            self.config = config

        def initiate_model_trainer(self,train_array,test_array):

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'XGBRegressor': XGBRegressor(),
                'catBoostRegressor': CatBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, test_models=models)
            logging.info(f"Model Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best Model Score: {best_model_score}")

            best_model_name = max(model_report,  key=lambda x: model_report[x])
            logging.info(f"Best Model {best_model_name} Score: {best_model_score}")

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("Model Score is poor, try again...")
                raise CustomException(f"Model score is poor",sys)

            save_object(
                file_path=ModelTrainerConfig.modelTrainer_path,
                object = best_model
            )
            logging.info("Saved Model File Successfully.")

            return best_model_score

    except Exception as e:
        logging.info(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
        raise CustomException(e,sys)

