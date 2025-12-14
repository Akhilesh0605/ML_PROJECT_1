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
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Inputing Train test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "RandomForrest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "CatBoosting Regressor":CatBoostRegressor(),
                "Adaboosting Regressor":AdaBoostRegressor(),
                "XGBoosting Regressor":XGBRegressor(),
                "KNearestNeighbours ":KNeighborsRegressor()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            #getting best model
            best_model_score=max(sorted(model_report.values()))

            #getting best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model found")

            logging.info("Model training completed and results stored")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                object=best_model
            )

            Y_pred=best_model.predict(X_test)
            return r2_score(y_test,Y_pred)
        except Exception as e:
            f"There occured an error  {e} in the {sys}"
            raise CustomException(e,sys)