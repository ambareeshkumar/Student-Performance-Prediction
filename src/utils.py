import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object,file_obj)

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, test_models, params):
    """
    Evaluates the performance of a list of models on the test set.
    """
    try:
        report = {}
        for ind in range(len(test_models)):

            ModelName = list(test_models.keys())[ind]
            logging.info(f"Evaluating Model {ModelName}")
            Model = test_models[ModelName]
            param = params[ModelName]

            logging.info(f"Performing GridSearchCv with {ModelName}")
            gird_search = GridSearchCV(Model, param, cv = 3)
            gird_search.fit(X_train, y_train)

            logging.info(f"Best Parameters for {ModelName}: {gird_search.best_params_}")
            Model.set_params(**gird_search.best_params_)
            logging.info(f"Best Score for {ModelName}: {gird_search.best_score_}")

            logging.info(f"Fitting X_train and Y_train for {ModelName}")
            Model.fit(X_train, y_train)

            logging.info(f"Predicitng Y_train and Y_test for {ModelName}")
            y_train_pred = Model.predict(X_train)
            y_test_pred = Model.predict(X_test)

            train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)
            report[f'{ModelName}_TrainScore'] = train_score
            report[f'{ModelName}_TestScore'] = test_score

        return report

    except Exception as e:
        logging.error(f"Exception occurred : {e}")
        raise CustomException(e,sys)