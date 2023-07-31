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
    description="A simple API that uses a onevsrest random forest model to predict labels of the movie's plots",
    version="0.1",
)

# load the movie model
with open(
    join(dirname(realpath(__file__)), "models/movie_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

# load the multilabel binarizer
with open(
    join(dirname(realpath(__file__)), "models/binarizer.pkl"), "rb"
) as f:
    binarizer = joblib.load(f)

def process_text(text):
    """ 
    Make text lowercase and remove punctuation
    :param text: string to process
    :return: processed string
    """
    text = str(text).lower()
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", " ", text
    )
    text = " ".join(text.split())
    return text


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict-genre")
def predict_labels(plot: str):
    """
    A simple function that receives a movie plot and predicts the genre of the movie
    :param rplot:
    :return: prediction, probabilities
    """    
    # perform prediction
    processed_review = process_text(plot)
    prediction = model.predict([processed_review])
    probas = list(model.predict_proba([processed_review])[0][:])
    classes = binarizer.classes_.tolist()

    prob_dict = dict()
    for i in range(len(probas)):
        predclass = classes[i]
        prob = probas[i]
        prob_dict[predclass] = prob

    label_predictions = binarizer.inverse_transform(prediction)

    result = {"prediction": label_predictions, "probs": str(prob_dict)}
    return result