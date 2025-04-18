from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]


class DataInputSchema(BaseModel):
    cylinders: Optional[int]
    displacement: Optional[float]
    horsepower: Optional[float]
    weight: Optional[float]
    acceleration: Optional[float]
    model_year: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
