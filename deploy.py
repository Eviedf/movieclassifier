import joblib
import uvicorn
from fastapi import FastAPI
from os.path import dirname, join, realpath
import re
import string

# following: https://www.freecodecamp.org/news/how-to-deploy-an-nlp-model-with-fastapi/

# intitialize app instance
app = FastAPI(
    title="Movie Model API",
    description="A simple API that uses a onevsrest logistic regression model to predict labels of the movie's plots",
    version="0.1",
)

# load the movie model
with open(
    join(dirname(realpath(__file__)), "models/movie_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "models/binarizer.pkl"), "rb"
) as f:
    binarizer = joblib.load(f)

def process_text(text):
    text = str(text).lower()
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", " ", text
    )
    text = " ".join(text.split())
    return text


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict-review")
def predict_labels(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """    
    # perform prediction
    prediction = model.predict([review])
    probas = model.predict_proba([review])

    label_predictions = binarizer.inverse_transform(prediction)
    print(prediction)
    
    return label_predictions