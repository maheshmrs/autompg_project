import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from autompg_model.config.core import config
from autompg_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    """
    Loads the full dataset, splits it into training and test sets, 
    and returns the test set (features and labels).
    """
    # Load full dataset
    data = load_dataset(file_name=config.app_config_.training_data_file)

    # Split into features and target
    X = data[config.model_config_.features]
    y = data[config.model_config_.target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state
    )

    return X_test, y_test  # ✅ Return a tuple
# def sample_input_data():
#     data = load_dataset(file_name = config.app_config_.training_data_file)

#     # divide train and test
#     X_train, X_test, y_train, y_test = train_test_split(
        
#         data[config.model_config_.features],     # predictors
#         data[config.model_config_.target],       # target
#         test_size = config.model_config_.test_size,
#         random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
#     )

#     return X_test, y_test

@pytest.fixture
def sample_input_preprocess_data():
    df = pd.DataFrame({
        'mpg': [18.0, 15.0, 36.0, 20.0, 22.0],
        'cylinders': [8, 8, 4, 6, 6],
        'displacement': [307.0, 350.0, 97.0, 140.0, 122.0],
        'horsepower': [130.0, 165.0, 75.0, 90.0, 88.0],
        'origin': [1, 1, 3, 1, 1],
        'car_name': ['chevy', 'ford', 'honda', 'mazda', 'toyota']
    })
    return df

@pytest.fixture
def sample_input_outlier_data():
    return pd.DataFrame({
        'horsepower': [50.0, 55.0, 60.0, 500.0]
    })
