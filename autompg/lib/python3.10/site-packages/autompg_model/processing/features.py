from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

    
class AutoMPGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns=None,n_quantiles=5):
        self.drop_columns = drop_columns if drop_columns else ['origin', 'car_name']
        self.n_quantiles = n_quantiles
        self.quantile_transformer = QuantileTransformer(random_state=0,n_quantiles=self.n_quantiles)

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            raise ValueError("AutoMPGPreprocessor.fit() received a tuple; expected DataFrame")
        self.feature_columns_ = [col for col in X.columns if col not in self.drop_columns]
        self.quantile_transformer.fit(X[self.feature_columns_])
        return self

    def transform(self, X):
        return self.quantile_transformer.transform(X[self.feature_columns_])



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values: 
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variable:str):

        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")

        self.variable = variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        for i in X.index:
            if X.loc[i, self.variable] > self.upper_bound:
                X.loc[i, self.variable]= self.upper_bound.astype(float)
            if X.loc[i, self.variable] < self.lower_bound:
                X.loc[i, self.variable]= self.lower_bound.astype(float)

        return X

