import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # used to handle missing values
from sklearn.pipeline import Pipeline

# import the custom exceptions
from src.exception import CustomException
from src.logger import logging

# import the save_object function from utils
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this functoin is respomsible for data transformation

        '''
        try:
            logging.info("Entered Transformation")
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            # Creating a pipeline which is performing 2 steps:handling the missing values and scaling the
            # numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # similary create the categrical pipeline
            # apply imputer pipeline to fill missing values with the mode
            # applying onehotencoding basically converting into numerical values
            # scaling those numerical values
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(
                "numerical columns standard scaling has been completed")
            logging.info("categorical columns encoding has been completed")

            # combine the numerical and categorical pipeline together
            preprocessor = ColumnTransformer(
                [
                    # we are passing the "pipeline_name we want",the actual pipeline,columns we are applying transformation
                    ("numerical_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data completed")

            logging.info("otaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_columns_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(
                [target_columns_name], axis=1)

            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(
                [target_columns_name], axis=1)

            target_feature_test_df = test_df[target_columns_name]

            logging.info(
                "applying preprocessing object on the train and test dataset")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("preprocessing completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
