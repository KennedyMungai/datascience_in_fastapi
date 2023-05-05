"""The entrypoint to the application"""
from fastapi import FastAPI
from models.newsgroup_models import NewsgroupsModel, PredictionOutput
from fastapi import Depends


app = FastAPI()
newsgroups_model = NewsgroupsModel()


@app.get("/", name="Home", description="The root endpoint of the application", tags=["Home"])
async def home():
    """The root endpoint of the application

    Returns:
        dict[str, str]: A statement to show that the api works
    """
    return {"Hello": "World"}


@app.post(
    "/prediction",
    name="Prediction",
    description="Predict the newsgroups of a given text",
    tags=["Prediction"]
)
async def prediction(
    output: PredictionOutput = Depends(newsgroups_model.predict),
):
    """Predict the newsgroups of a given text

    Args:
        output (PredictionOutput): The output of the prediction

    Returns:
        PredictionOutput: The output of the prediction
    """
    return output
