

import uvicorn
from fastapi import FastAPI
from values import Values
import numpy as np
import pandas as pd
import pickle
from fastapi import HTTPException

app = FastAPI()
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)
encoders = pickle.load(open("encoders.pkl", "rb"))

# @app.post('/predict')
# def predict_rent(data: Values):
#     data = data.dict()
#     print(data)
#     BHK = data["BHK"]
#     Size = data["Size"]
#     Area = encoders["Area Type"].transform([data["Area"]])[0]
#     City = encoders["City"].transform([data["City"]])[0]
#     Furnishing = encoders["Furnishing Status"].transform([data["Furnishing"]])[0]
#     Bathroom = data["Bathroom"]

#     prediction = model.predict([[BHK,Size,Area,City,Furnishing,Bathroom]])
#     return prediction

@app.post('/predict')
def predict_rent(data: Values):
    """
    Validate types and categorical values before calling the model.
    Returns a JSON-serializable dict with the prediction or raises HTTPException(400).
    """
    payload = data.dict()
    # Basic presence checks
    required = ["BHK", "Size", "Area", "City", "Furnishing", "Bathroom"]
    missing = [k for k in required if k not in payload or payload[k] is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    # Type checks / conversions
    try:
        BHK = int(payload["BHK"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="BHK must be an integer")

    try:
        Size = float(payload["Size"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Size must be a number")

    try:
        Bathroom = int(payload["Bathroom"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Bathroom must be an integer")

    # Validate categorical values against saved encoders
    area_val = payload["Area"]
    city_val = payload["City"]
    furnishing_val = payload["Furnishing"]

    try:
        Area = encoders["Area Type"].transform([area_val])[0]
    except Exception:
        allowed = list(getattr(encoders["Area Type"], "classes_", []))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Area '{area_val}'. Allowed: {allowed}"
        )

    try:
        City = encoders["City"].transform([city_val])[0]
    except Exception:
        allowed = list(getattr(encoders["City"], "classes_", []))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid City '{city_val}'. Allowed: {allowed}"
        )

    try:
        Furnishing = encoders["Furnishing Status"].transform([furnishing_val])[0]
    except Exception:
        allowed = list(getattr(encoders["Furnishing Status"], "classes_", []))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Furnishing '{furnishing_val}'. Allowed: {allowed}"
        )

    # Model prediction with safe error handling
    try:
        pred = model.predict([[BHK, Size, Area, City, Furnishing, Bathroom]])
        # Ensure scalar numeric result that can be JSON serialized
        value = float(np.squeeze(pred))
        return {"prediction": round(value, 2)}
    except Exception as e:
        # Unexpected error in prediction => 500, but surface a friendly message
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)