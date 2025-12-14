import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import os
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transformer_obj(slef):
        # this function gives the transformed data in standardization
        try:
            numerical_features=[
                "writing_score","reading_score"
            ]
            categirical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler()),
                ]
            )
            logging.info("numerical features standariaed")
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=True)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categirical features encoded")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categirical_features)
                ]
            )

            return preprocessor

           
        except Exception as e:
            logging.error(f" An error occurred during data ingestion: {e}")
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df=pd.read_csv("/home/rguktrkvalley/AIML_Projects/ML_PROJECTS/ML_PROJECT1/src/components/artifact/train.csv")
            test_df=pd.read_csv("/home/rguktrkvalley/AIML_Projects/ML_PROJECTS/ML_PROJECT1/src/components/artifact/test.csv")

            logging.info("Train and test data read")
            logging.info("obtaining the preprocessign data")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numerical_features = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing completed. Saving the preprocessor object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
                


