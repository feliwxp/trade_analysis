import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
import datetime as dt


class Datapipeline:
    def __init__(self):
        self.TARGET_VARIABLE = "target"

        # Build a preprocessing step for numeric features
        self.numeric_features = [
            'IV', 'RV', 'IV-RV'
        ]
        numeric_pipeline = Pipeline([("StandardScaling", StandardScaler())])

        # Build a preprocessing step for ordinal features
        self.ordinal_features = []
        ordinal_pipeline = Pipeline(
            [("ordinal_encoder", OrdinalEncoder()),
             ("min_max_scaler", MinMaxScaler())]
        )
        # Build a preprocessing step for categorical features
        self.categorical_features = []
        categorical_pipeline = Pipeline(
            [("onehot", OneHotEncoder(drop="first", handle_unknown='ignore'))])

        # Create columntransformer pipeline

        self.preprocessor = ColumnTransformer(
            [
                ("num", numeric_pipeline, self.numeric_features),
                ("ord", ordinal_pipeline, self.ordinal_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            verbose_feature_names_out=True,
            remainder="passthrough",
        )

    def get_labels(self):
        """
        This functions returns the labels of the dataframe based on the pipeline.

        :return: list, data labels
        """

        # Get the labels for the dataframe
        if self.categorical_features:
            cat_features = self.preprocessor.named_transformers_[
                "cat"
            ].get_feature_names_out()
            labels = np.concatenate(
                [self.numeric_features, self.ordinal_features, cat_features]
            )
        else:
            labels = np.concatenate(
                [self.numeric_features, self.ordinal_features]
            )
        return labels

    def preprocessing(self, df):
        """
        This functions takes in a DataFrame and returns a processed Dataframe.

        :param df: pd.DataFrame, data
        :return: pd.DataFrame, processed data
        """

        return df

    def transform_train_data(self, train_data_path):
        """
        This functions takes in to the train data path and returns X_train and y_train with respect with the target variable.

        :param train_data_path: pd.DataFrame, train data
        :return: tuple, (X_train, y_train)
        """
        # Read train csv file
        train = pd.read_csv(train_data_path)

        # Preprocess train data
        train = self.preprocessing(train)

        # Drop target variable in train data
        train_transformed = train.drop(self.TARGET_VARIABLE, axis=1)

        # Fit and transform train data
        train_transformed = self.preprocessor.fit_transform(train_transformed)

        # Convert to numpy array
        X_train = train_transformed
        y_train = train[self.TARGET_VARIABLE].to_numpy()

        return X_train, y_train

    def transform_test_data(self, test_data_path):
        """
        This functions takes in to the test data path and returns X_test and y_test with respect with the target variable.

        :param test_data_path: pd.DataFrame, test data
        :return: tuple, (X_train, y_train)
        """

        # Read test csv file
        test = pd.read_csv(test_data_path)

        # Preprocess test data
        test = self.preprocessing(test)

        # Drop target variable in test data
        test_transformed = test.drop(self.TARGET_VARIABLE, axis=1)

        # Transform train data
        test_transformed = self.preprocessor.transform(test_transformed)

        # Convert to numpy array
        X_test = test_transformed
        y_test = test[self.TARGET_VARIABLE].to_numpy()
        return X_test, y_test
