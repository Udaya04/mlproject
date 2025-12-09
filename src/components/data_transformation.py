import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl").replace("\\", "/")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_datatranformation_object(self):
        """
        
        :param self: -This function responsible for Data tranformation
        """
        try:

            numerical_columns=['math score','writing score','reading score']
            categorical_columns=[
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'

            ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # handling the missing values
                    ('scaler',StandardScaler())
                ]
            )
            catgorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(drop="first")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical Columns Standardscaling completed')
            logging.info('numerical columns encoding completed')
            ## Combine both pipelines
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('catogorical_pipeline',catgorical_pipeline,categorical_columns)
                ]
            )
            logging.info("preprocessor Pipeline completed")
            return preprocessor

        except Exception  as e:
            raise CustomException(e,sys)
    


    def initiate_data_transformation(self,train_path,test_path):

        try:
            # Read Data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and Test data')

            target_column='Average_Score'

            X_train=train_df.drop(columns=[target_column])
            y_train=train_df[target_column]

            X_test=test_df.drop(columns=[target_column])
            y_test=test_df[target_column]

            preprocessor_obj=self.get_datatranformation_object()
            logging.info('Applying Preprocessing On train and test')

            X_train_scaled=preprocessor_obj.fit_transform(X_train)
            X_test_scaled=preprocessor_obj.transform(X_test)

            # np.c_ stacks arrays column-wise, meaning it adds columns side by side.
            train_arr=np.c_[
                X_train_scaled,np.array(y_train)
            ]
            test_arr=np.c_[X_test_scaled,np.array(y_test)]

            ## Save the preprocessor
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

            


        except Exception as e:
            logging.error(f"Data Transformation failed: {e}")
            raise CustomException(e, sys)

    









