import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


from src.components.data_transformation import DataTransformer
from src.components.data_transformation import DataTransformerConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_save_path: str = os.path.join('artifacts',"raw_data.csv")
    train_save_path: str = os.path.join('artifacts',"train.csv")
    test_save_path: str = os.path.join('artifacts',"test.csv")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            df = pd.read_csv('../../Data/student_performance.csv')
            logging.info("Successfully Converted CSV to DataFrame")
            os.makedirs(os.path.dirname(self.config.train_save_path),exist_ok=True)

            df.to_csv(self.config.raw_save_path, index=False, header=True)
            logging.info("Successfully Saved Raw Data")

            logging.info("Train Test Split Initiated")
            train_set,test_set = train_test_split(df, test_size=0.2,random_state=14)

            train_set.to_csv(self.config.train_save_path, index=False, header=True)
            test_set.to_csv(self.config.test_save_path, index=False, header=True)
            logging.info("Successfully Saved Train and Test Data")

            logging.info("Data Ingestion Completed")

            return (self.config.train_save_path,
                    self.config.test_save_path,
            )
        except Exception as e:
            logging.error("Data Ingestion Failed")
            raise CustomException(e,sys)

if __name__ == "__main__":

    logging.info("Running Data Ingestion")
    Ingestion = DataIngestion(DataIngestionConfig)
    train_data, test_data = Ingestion.initiate_data_ingestion()

    logging.info(f"Running Data Transformation")
    Transformation = DataTransformer(DataTransformerConfig)
    train_arr, test_arr, _ = Transformation.initiate_data_transform(train_data, test_data)

    logging.info(f"Running Model Training")
    ModelTrainer = ModelTrainer(ModelTrainerConfig)
    ModelTrainer.initiate_model_trainer(train_arr, test_arr)
