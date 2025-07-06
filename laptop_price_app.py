# laptop_price_app.py
# author: Joe Tolsma (tolsmajdx@gmail.com)

# package import 
# ------------------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from laptop_predictor_files.utils.preprocessing import FeaturizeCPU, FeaturizeStorage, FeaturizeScreen
from typing import Dict

# create and configure application
# ------------------------------------------------------------------------------
app = FastAPI()

# define input schema
class LaptopInput(BaseModel):
    Manufacturer: str
    ModelName: str
    Category: str
    ScreenSize: float
    ScreenSpec: str
    CPU: str
    RAM: int
    Storage: str
    GPU: str
    OperatingSystem: str
    OperatingSystemVersion: str
    Weight: float

# set input path for stored model objects
MODEL_PATH = Path("laptop_predictor_files/model")

# load model objects for prediction
# ------------------------------------------------------------------------------
model = joblib.load(MODEL_PATH / "laptop_price_model.pkl")
model_features = joblib.load(MODEL_PATH / "laptop_price_model_features.pkl")
model_encoders = joblib.load(MODEL_PATH / "laptop_feature_encoding_maps.pkl")
model_replace_strings = joblib.load(MODEL_PATH / "laptop_feature_replace_strings.pkl")

# define the prediction endpoint
# ------------------------------------------------------------------------------
@app.post("/predict")

def predict_price(input_data: LaptopInput):

    input_data = input_data.model_dump()

    # apply user-defined functions from model fitting to parse CPU, storage, screen, and GPU features
    cpu_features = FeaturizeCPU(input_data["CPU"])
    storage_features = FeaturizeStorage(input_data["Storage"])
    screen_features = FeaturizeScreen(input_data["ScreenSpec"])
    gpu_str_split = input_data["GPU"].split(" ",maxsplit = 1)
    gpu_features = {"GPUBrand":gpu_str_split[0],"GPUProduct":gpu_str_split[1]}
        
    # Check feature values to be encoded; if not in the encoding dictionaries, replace with the replacement string for each feature
    cpu_features["CPUBrand"] = model_replace_strings["cpu_brand"] if cpu_features["CPUBrand"] not in model_encoders["cpu_brand"].keys() else cpu_features["CPUBrand"]
    cpu_features["CPUProduct"] = model_replace_strings["cpu_prod"] if cpu_features["CPUProduct"] not in model_encoders["cpu_prod"].keys() else cpu_features["CPUProduct"]

    gpu_features["GPUBrand"] = model_replace_strings["gpu_brand"] if gpu_features["GPUBrand"] not in model_encoders["gpu_brand"].keys() else gpu_features["GPUBrand"]
    gpu_features["GPUProduct"] = model_replace_strings["gpu_prod"] if gpu_features["GPUProduct"] not in model_encoders["gpu_prod"].keys() else gpu_features["GPUProduct"]

    input_data["Manufacturer"] = model_replace_strings["manufs"] if input_data["Manufacturer"] not in model_encoders["manufs"].keys() else input_data["Manufacturer"]
    input_data["Category"] = model_replace_strings["category"] if input_data["Category"] not in model_encoders["category"].keys() else input_data["Category"]
    input_data["OperatingSystem"] = model_replace_strings["os"] if input_data["OperatingSystem"] not in model_encoders["os"].keys() else input_data["OperatingSystem"]
    input_data["OperatingSystemVersion"] = model_replace_strings["os_version"] if input_data["OperatingSystemVersion"] not in model_encoders["os_version"].keys() else input_data["OperatingSystemVersion"]
    
    # execute encoding step for all categorical features
    X = pd.DataFrame(
        data = [[
        model_encoders["manufs"][input_data["Manufacturer"]], # encode manufacturer
        model_encoders["category"][input_data["Category"]], # encode category
        model_encoders["cpu_brand"][cpu_features["CPUBrand"]], #encode CPU brand
        model_encoders["cpu_prod"][cpu_features["CPUProduct"]], # encode CPU product
        model_encoders["gpu_brand"][gpu_features["GPUBrand"]], # encode GPU brand
        model_encoders["gpu_prod"][gpu_features["GPUProduct"]], # encode GPU product
        model_encoders["os"][input_data["OperatingSystem"]], # encode operating system
        model_encoders["os_version"][input_data["OperatingSystemVersion"]] # encode operating system version
        ]],
        columns = ["manufacturer","category","cpu_brand","cpu_product","gpu_brand","gpu_product","operating_system","operating_system_version"]
    )
    
    # prepare screen-related model features
    screen_features["Screen Resolution"] = screen_features["Diag Resolution"] / input_data["ScreenSize"]
    screen_features = pd.DataFrame([screen_features]).drop("Diag Resolution",axis = 1)
    screen_features.columns = screen_features.columns.str.lower().str.replace(" ","_")

    # prepare storage-related model features
    storage_features = pd.DataFrame([storage_features])
    storage_features.columns = storage_features.columns.str.lower().str.replace(" ","_")

    # prepare CPU-related model features
    cpu_features_numeric = {k:v for k,v in cpu_features.items() if k in ["CPUGhz","CPU Core Count"]}
    cpu_features_numeric = (pd.DataFrame([cpu_features_numeric])
        .rename({"CPUGhz":"cpu_ghz","CPU Core Count":"cpu_core_count"},axis = 1)
        .astype("float")
    )
    
    # store remaining (numeric) model features as a separate dataframe
    input_numeric_dict = {
        "ram":int(input_data["RAM"]),
        "weight":float(input_data["Weight"])
        }

    input_numeric = pd.DataFrame([input_numeric_dict])

    # combine all feature dataframes into a single model input dataframe, then set column order to match the column arrangement that the model expects
    X = pd.concat([X,screen_features,storage_features,cpu_features_numeric,input_numeric],axis = 1)
    X = X[model_features]
    
    # generate price prediction -- need to reverse log transform (applied during training) using exponent back-transform
    prediction = np.expm1(model.predict(X)[0])
    
    return {"predicted_price": np.round(prediction,2)}
