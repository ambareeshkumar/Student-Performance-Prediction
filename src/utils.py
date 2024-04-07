import os
import sys
import pandas as pd
import numpy as np
import dill

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