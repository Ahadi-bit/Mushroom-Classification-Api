import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


origins = ['*']


class MushroomFeatureIN(BaseModel):
    capSurface: int
    bruises: int
    odor: int
    gillAttachment: int
    gillSpacing: int
    gillSize: int
    gillColor: int
    stalkShape: int
    stalkRoot: int
    stalkSurfaceAboveRing: int
    stalkSurfaceBelowRing: int
    stalkColorAboveRing: int
    stalkColorBelowRing: int
    ringNumber: int
    population: int
    habitat: int


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


classifier_pickle = open("pickle/classification-model.pkl", "rb")
classifier = pickle.load(classifier_pickle)

correlation_pickle = open("pickle/correlation.pkl", "rb")
correlation = pickle.load(correlation_pickle)



@app.get("/")
def hello():
    return {"welcome":"Welcome to the mushroom classification api!"}


@app.post('/predict')
def predict(incoming_data: MushroomFeatureIN):
    data = incoming_data.dict()
    data = pd.DataFrame(data, index=[0])
    temp = data.values
    temp = temp.reshape(1, -1)
    prediction = classifier.predict(temp)
    confidence = classifier.predict_proba(temp)
    not_edible = round(confidence[0][0]*100)
    edible = round(confidence[0][1]*100)

    print(confidence[0][1])
    if prediction[0] == 0:
        prediction = "Edible Mushroom"
    else:
        prediction = "Poisonous Mushroom"

    return prediction, not_edible, edible


@app.get('/correlation')
def report():

    return correlation


data = pd.read_csv('data/mushrooms.csv')

@app.get("/balance/{feature}")
def balance(feature: str):
    balance = {}

    if feature in data:
        balance = data.reset_index().groupby(feature).apply(lambda x : x['class'].value_counts().to_dict()).to_dict()
    else:
        balance = {"error":"Item does not exist"}
          
    return balance


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# cd