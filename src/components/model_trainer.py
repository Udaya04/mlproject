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
from src.utils import save_object, evaluate_models

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

            # Models dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor()
            }

            # Parameter grids for GridSearchCV
            params = {
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_split": [2, 5]},
                "Decision Tree": {"max_depth": [5, 8], "min_samples_split": [2, 5]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                "Linear Regression": {},
                "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                "CatBoost": {"iterations": [200, 500], "learning_rate": [0.05, 0.1], "depth": [4, 6]},
                "AdaBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
            }

            model_report = {}

            # Loop through all models and do GridSearchCV if parameters exist
            for model_name, model in models.items():
                logging.info(f"Training and tuning model: {model_name}")

                if params.get(model_name):
                    grid = GridSearchCV(model, param_grid=params[model_name], cv=3, n_jobs=-1, scoring='r2')
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    score = r2_score(y_test, best_model.predict(X_test))
                    logging.info(f"{model_name} best params: {grid.best_params_}, R2: {score}")
                else:
                    # No params for Linear Regression
                    model.fit(X_train, y_train)
                    best_model = model
                    score = r2_score(y_test, best_model.predict(X_test))

                model_report[model_name] = score
                models[model_name] = best_model  # save best model

            # Get best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (score < 0.6)")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)






