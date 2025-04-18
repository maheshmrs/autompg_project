import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import TransformedTargetRegressor


from autompg_model.config.core import config

from autompg_model.processing.features import OutlierHandler,AutoMPGPreprocessor

autompg_pipe = Pipeline([

    ######### Imputation ###########
     
    ######## Handle outliers ########
    ('handle_outliers_cylinders', OutlierHandler(variable = config.model_config_.cylinders_var)),
    ('handle_outliers_displacement', OutlierHandler(variable = config.model_config_.displacement_var)),
    ('handle_outliers_horsepower', OutlierHandler(variable = config.model_config_.horsepower_var)),
    ('handle_outliers_weight', OutlierHandler(variable = config.model_config_.weight_var)),
    ('handle_outliers_acceleration', OutlierHandler(variable = config.model_config_.acceleration_var)),
    ('handle_outliers_model_year', OutlierHandler(variable = config.model_config_.model_year_var)),
    ('autompg_preprocessor', AutoMPGPreprocessor(drop_columns=['origin', 'car_name'])),
    ######## One-hot encoding ########

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config_.n_estimators, 
                                       max_depth = config.model_config_.max_depth,
                                      random_state = config.model_config_.random_state))
    
    ])
