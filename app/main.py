"""The entrypoint to the application"""
from fastapi import FastAPI


app = FastAPI()


@app.get("/", name="Home", description="The root endpoint of the application", tags=["Home"])
async def home():
    """The root endpoint of the application

    Returns:
        dict[str, str]: A statement to show that the api works
    """
    return {"Hello": "World"}
