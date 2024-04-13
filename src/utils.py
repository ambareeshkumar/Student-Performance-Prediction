import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(object,file_obj)

    except Exception as e:
        logging.info(f"Error saving object to {file_path}: {e}")
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, test_models):
    """
    Evaluates the performance of a list of models on the test set.
    """

    try:
        report = {}

        for ModelName, Model in test_models.items():
            Model.fit(X_train, y_train)
            y_train_pred = Model.predict(X_train)
            y_test_pred = Model.predict(X_test)
            train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)
            report[ModelName] = test_score

            return report

    except Exception as e:
        raise CustomException(e,sys)