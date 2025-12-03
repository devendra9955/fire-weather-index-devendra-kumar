# fast api call for our project
from fastapi import FastAPI, Request
from pydantic import BaseModel, ValidationError
from fastapi.responses import JSONResponse
import joblib
import pandas as pd


app = FastAPI(
    title="Fire Weather Index Predictor API",
    description="Predict Fire Weather Index (FWI) from weather data.",
    version="1.1"
)

model = joblib.load("Linear_regression_models.pkl")
scaler = joblib.load("Scalers.pkl")

FEATURES = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']


class FWIInput(BaseModel):
    Temperature: float
    RH: float
    Ws: float
    Rain: float
    FFMC: float
    DMC: float
    DC: float
    ISI: float
    BUI: float



@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    errors = []
    for e in exc.errors():
        field = e.get("loc", ["unknown"])[-1]
        msg = e.get("msg", "Invalid value")
        errors.append(f"Field '{field}': {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid or missing input data",
            "details": errors,
            "tip": "Ensure all numeric fields are provided correctly: "
                   "Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI",
        },
    )



@app.get("/")
def home():
    return {
        "message": "Welcome to Fire Weather Index Prediction API ",
        "usage": "Send a POST request to /predict with weather data to get FWI prediction."
    }



@app.post("/predict")
def predict_fwi(data: FWIInput):
    try:

        input_df = pd.DataFrame([{
            "Temperature": data.Temperature,
            "RH": data.RH,
            "Ws": data.Ws,
            "Rain": data.Rain,
            "FFMC": data.FFMC,
            "DMC": data.DMC,
            "DC": data.DC,
            "ISI": data.ISI,
            "BUI": data.BUI
        }])

        scaled_data = scaler.transform(input_df)

        predicted_fwi = model.predict(scaled_data)[0]

        return {
            "Predicted_FWI": round(float(predicted_fwi), 2),
            "Status": "Success"
        }

    except Exception as e:
    
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e)
            },
        )
