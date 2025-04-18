import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from autompg_model import __version__ as _version
from autompg_model.config.core import config
from autompg_model.processing.data_manager import load_pipeline
from autompg_model.processing.data_manager import pre_pipeline_preparation
from autompg_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
autompg_pipe = load_pipeline(file_name = pipeline_file_name)
#print("autompg_pipe =",autompg_pipe)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = autompg_pipe.predict(validated_data)
        results = {"predictions": np.floor(predictions), "version": _version, "errors": errors}
        
    print(results)
    return results



if __name__ == "__main__":

    data_in = {'cylinders': [8], 'displacement': [307.0], 'horsepower': [130.0], 'weight': [3504.0], 'acceleration': [12.0], 'model_year': [70]}
    print(data_in)
    make_prediction(input_data = data_in)