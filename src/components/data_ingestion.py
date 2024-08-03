import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# import the transformation file
from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig

# import the model trainer component
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # reading the dataset(can be from anywhere)
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Read the dataset as df")

            # creating a directory to store the test and train and raw data
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            # converting the raw data to a csv file
            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)

            logging.info("train_test_split intitiated")
            # splitting the data into train and test set
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            # converting the train_set to csv file and saving it to the train file
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            # converting the test_set to csv file and saving it to the test file
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("ingestion of the data is completed")
            # returning the two data ingestion/splitted data components to the next step i.e. data transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            # raising a custom exception
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_tranformation = DataTransformation()
    train_arr, test_arr, _ = data_tranformation.initiate_data_transformation(
        train_data, test_data)
    Modeltrainer = ModelTrainer()
    print(Modeltrainer.initiate_model_trainer(train_arr, test_arr))
