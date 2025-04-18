
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from autompg_model.config.core import config
from autompg_model.processing.features import  OutlierHandler, AutoMPGPreprocessor

def test_autompg_preprocessor_transforms_data_correctly(sample_input_preprocess_data):
    # Given
    
    preprocessor = AutoMPGPreprocessor(drop_columns=['origin', 'car_name'])
    X = sample_input_preprocess_data.drop(columns=["mpg"])
    y = sample_input_preprocess_data["mpg"]
    # When
    subject = preprocessor.fit(X, y).transform(X)

    # Then
    assert isinstance(subject, np.ndarray)
    assert subject.shape[0] == X.shape[0]
    assert subject.shape[1] == X.shape[1] - 2  # dropped 2 columns

def test_outlier_handler(sample_input_outlier_data):
    # Given
    df = sample_input_outlier_data  # ✅ Don't unpack
    handler = OutlierHandler(variable='horsepower')
    original_outlier = df['horsepower'].iloc[-1]

    # When
    subject = handler.fit(df).transform(df)

    # Then
    transformed_outlier = subject['horsepower'].iloc[-1]
    assert transformed_outlier < original_outlier
    assert transformed_outlier <= handler.upper_bound
