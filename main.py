import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

origins = ['*']



# Base model for API
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

description = """
Mushroom Classification API helps you do awesome stuff. 🚀

## Predict

Classifies mushroom as posionous or edible.

## Correlation

Pulls correlation of mushroom features based on target variables

## Balance
Returns the distribution of mushroom features and there count on edible and poisonous

## Population
Pulls the population distribution accross the data.

"""

app = FastAPI(
    title="Mushroom Classification API",
    description=description,
    version="0.0.1"
)

#cors middle cors needed to create get and post request locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## Machine Learning Model
classifier_pickle = open("pickle/classification-model.pkl", "rb")
classifier = pickle.load(classifier_pickle)

## Needed for the correlation chart
correlation_pickle = open("pickle/correlation.pkl", "rb")
correlation = pickle.load(correlation_pickle)


# Entry Point of the api
@app.get("/")
def entry():
    return {"welcome":"Welcome to the mushroom classification api!"}

# prediction api
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

# correlation api
@app.get('/correlation')
def report():

    return correlation

## Data
data = pd.read_csv('data/mushrooms.csv')

## Balance API
@app.get("/balance/{feature}")
def balance(feature: str):
    balance = {}

    if feature in data:
        balance = data.reset_index().groupby(feature).apply(lambda x : x['class'].value_counts().to_dict()).to_dict()
    else:
        balance = {"error":"Item does not exist"}
          
    return balance

## Population API
@app.get("/population")
def population():
    populations = data['population'].value_counts().to_dict()    
    
    pop_labels = {'v':'Several', 'y':'Solitary', 's':'Scattered', 'n':'Numerous', 'a':'Abundant', 'c':'Clustered'}

    label_population = dict((pop_labels[key], value) for (key, value) in populations.items())
    
    return label_population

## Enables the ability to run locally 
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
