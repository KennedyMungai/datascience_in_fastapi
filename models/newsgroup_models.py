"""The models file for the news group"""
import os
from typing import List, Optional, Tuple

import joblib
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


class PredictionInput(BaseModel):
    """The template for the Prediction Input data

    Args:
        BaseModel (Pydantic): The parent class fir models
    """
    text: str


class PredictionOutput(BaseModel):
    """The template for the Prediction Output data

    Args:
        BaseModel (Pydantic): The parent class fir models
    """
    category: str


class NewsgroupsModel:
    """The model for the Newsgroup
    """
    model: Optional[Pipeline]
    targets: Optional[List[str]]

    def load_model(self):
        """Loads the model from the file"""
        model_file = os.path.join(os.path.dirname(
            __file__), "newsgroup_model.joblib")
        loaded_model: Tuple[Pipeline, List[str]] = joblib.load(model_file)
        model, targets = loaded_model
        self.model = model
        self.targets = targets

    def predict(self, _input: PredictionInput) -> PredictionOutput:
        """Runs the prediction

        Args:
            input (PredictionInput): The input data

        Returns:
            PredictionOutput: The output data
        """
        if not self.model or not self.targets:
            raise RuntimeError("Model is not loaded")

        prediction = predict(self.model, input.text)
        category = self.targets[prediction]

        return PredictionOutput(category=category)
