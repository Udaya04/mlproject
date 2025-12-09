import os
import pandas as pd
import numpy as np
import sys
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    
    try:
        report = {}

        for model_name, model in models.items():
            if param and model_name in param and param[model_name]:
                # Apply GridSearchCV if parameters are provided
                grid = GridSearchCV(estimator=model,
                                    param_grid=param[model_name],
                                    scoring='r2',
                                    cv=3,
                                    n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
            else:
                # No params provided, train model as-is
                model.fit(X_train, y_train)
                best_model = model
                y_pred = best_model.predict(X_test)

            score = r2_score(y_test, y_pred)
            report[model_name] = score
            models[model_name] = best_model  # update the best model

        return report

    except Exception as e:
        raise CustomException(e, sys)



