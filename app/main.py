"""The entrypoint to the application"""
from fastapi import FastAPI, Depends, status
from models.newsgroup_models import NewsgroupsModel, PredictionOutput
import joblib
from sklearn.pipeline import Pipeline


app = FastAPI()
newsgroups_model = NewsgroupsModel()

memory = joblib.Memory(location="cache.joblib")


@memory.cache(ignore=["model"])
def predict(model: Pipeline, text: str) -> int:
    """This method caches the trained model

    Args:
        model (Pipeline): The model of the data
        text (str): The text search?

    Returns:
        int: Prediction accuracy
    """
    prediction = model.predict([text])
    return prediction[0]


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


@app.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cache():
    """The endpoint for clearing the cache

    Returns:
        None: Returns nothing
    """
    memory.clear()
    return None


@app.on_event("startup")
async def startup():
    """The startup event

    Returns:
        None: None

    """
    newsgroups_model.load_model()
