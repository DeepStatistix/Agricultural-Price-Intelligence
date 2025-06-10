import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os

app = FastAPI()

# Setup for rendering the HTML page
templates = Jinja2Templates(directory="templates")

# Base directory where models are stored
MODEL_BASE_PATH = "../models" 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/forecast/{market}/{variety_grade}/{model_name}")
async def get_forecast(market: str, variety_grade: str, model_name: str):
    """
    Loads a specific model and returns a forecast.
    Example URL: /forecast/market_A/variety_X_grade_A/prophet
    """
    try:
        # Construct the path to the model file
        model_path = os.path.join(MODEL_BASE_PATH, market, variety_grade, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Load the specified model from the .pkl file
        with open(model_path, 'rb') as pkl_file:
            model = pickle.load(pkl_file)

        # --- Forecasting Logic ---
        # This part is highly dependent on your model type.
        # You will need to adapt this logic for each of your six model types.
        # Example for a statsmodels/Prophet-like model:
        
        forecast_steps = 7 # e.g., forecast for the next 7 days
        if hasattr(model, 'forecast'):
            forecast_values = model.forecast(steps=forecast_steps)
            last_date = model.model.endog.index[-1]
            # Adjust frequency ('D' for daily, 'W' for weekly) as needed
            forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]

        elif hasattr(model, 'predict'): # For scikit-learn compatible models
            # This requires creating future feature data, which is more complex.
            # This is a placeholder for your feature engineering logic.
            # You would need to create a future DataFrame with the same features
            # your model was trained on (e.g., month, year, lag features).
            # For now, we'll return a placeholder.
            raise HTTPException(status_code=501, detail="Forecasting for this model type is not implemented yet.")
        
        else:
            raise HTTPException(status_code=501, detail="Unknown model type.")

        # Format the response
        response_data = [
            {"date": date.strftime('%Y-%m-%d'), "forecast": value}
            for date, value in zip(forecast_dates, forecast_values.tolist())
        ]
        return response_data

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model file not found at {model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
