import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # All Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor()
            }

            # Parameter Grids
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 5]
                },
                "Decision Tree": {
                    "max_depth": [5, 8],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "Linear Regression": {},  # No tuning
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "CatBoost": {
                    "iterations": [200, 500],
                    "learning_rate": [0.05, 0.1],
                    "depth": [4, 6]
                },
                "AdaBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                }
            }

            model_report = {}

            # Loop models & perform GridSearch
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                if params.get(model_name):
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params[model_name],
                        cv=3,
                        scoring='r2',
                        n_jobs=-1
                    )
                    grid.fit(X_train, y_train)

                    best_model = grid.best_estimator_
                    logging.info(f"{model_name} Best Params: {grid.best_params_}")

                else:
                    model.fit(X_train, y_train)
                    best_model = model

                # Evaluate
                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                logging.info(f"{model_name} R2 Score: {score}")

                model_report[model_name] = score
                models[model_name] = best_model  # replace with tuned model

            # Select Best Model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No good model found (R2 < 0.60)")

            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_model_score}")

            # Save Model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
