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
    modelTrainer_path = os.path.join('artifacts','model.pkl')

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
                'CatBoostRegressor': CatBoostRegressor()
            }

            param_grid = {
                'LinearRegression': {'fit_intercept': [True, False]},
                'DecisionTreeRegressor': {'max_depth': [None, 5, 10, 20]},
                'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
                'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
                'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0],
                                              'max_depth': [3, 5, 10]},
                'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0],
                                 'max_depth': [3, 5, 10]},
                'CatBoostRegressor': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, test_models=models, params = param_grid)
            logging.info(f"Model Report: {model_report}")

            model_test_report = {}

            for key, val in model_report.items():
                if key.split('_')[1] == 'TestScore':
                    model_test_report[key] = val

            print(f"Model Test Report: {model_test_report}")
            best_model_score = max(model_test_report.values())
            logging.info(f"Best Model Score: {best_model_score}")

            model_test_report = max(model_report,  key=lambda x: model_report[x])
            best_model_name = model_test_report.split('_')[0]
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
        logging.error(f"Error occurred in the file {__file__} on line number {sys._getframe().f_lineno} error message {str(e)}")
        raise CustomException(e,sys)

