import uvicorn
from fastapi import FastAPI
from values import Values
import numpy as np
import pandas as pd
import pickle

app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.post('/predict')
def predict_rent(data: Values):
    data = data.dict()
    print(data)
    BHK = data["BHK"]
    Size = data["Size"]
    Area = encoders["Area Type"].transform([data["Area"]])[0]
    City = encoders["City"].transform([data["City"]])[0]
    Furnishing = encoders["Furnishing Status"].transform([data["Furnishing"]])[0]
    Bathroom = data["Bathroom"]

    prediction = model.predict([[BHK,Size,Area,City,Furnishing,Bathroom]])
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)