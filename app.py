from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Modeli yükle
with open("iris_gnb_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    label_encoder = saved_data["label_encoder"]
    scaler = saved_data["scaler"]

# Kullanıcıdan alınacak veriler
class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("iindex.html", {"request": request})

@app.post("/predict")
async def predict(features: IrisFeatures):
    # Input verisini dataframe’e çevir
    input_data = pd.DataFrame([features.dict()])

    # Scale et
    input_scaled = scaler.transform(input_data)

    # Tahmin et
    pred_encoded = model.predict(input_scaled)

    # Encode edilmiş tahmini tekrar orijinal label’e çevir
    pred_species = label_encoder.inverse_transform(pred_encoded)

    return {"predicted_species": pred_species[0]}