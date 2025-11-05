# app/server.py
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)



# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    
    import numpy as np
    print("✅ Predict endpoint hit!")

    
    input_array = np.array([
        [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
        for s in req.samples
    ])

    # Run prediction with MLflow model
    preds = model.predict(input_array)

    # Ensure integer class IDs
    preds = [int(p) for p in preds]

    # Map class IDs to human-readable labels
    labels = [IRIS_LABELS[p] for p in preds]

    # Return structured response
    return PredictResponse(class_id=preds, class_label=labels)
    
# TODO Add endpoint to get the current model serving version
from fastapi.responses import JSONResponse

@app.get(
    "/model_version",
    tags=["model"],
    summary="Get current model serving version",
    description="Returns the MLflow model name, version, and stage currently served."
)
def get_model_version():
    return JSONResponse(content={
        "model_name": MODEL_NAME,
        "model_version_or_stage": MODEL_VERSION, 
        "model_uri": MODEL_URI
    })
    
    
    
class UpdateModelRequest(BaseModel):
    model_version: str = Field(None, description="MLflow model version to serve")
  
    

@app.post(
    "/update_model",
    tags=["model"],
    summary="Update the served MLflow model",
    description="Provide a model version or stage to update the currently served model."
)
def update_model(req: UpdateModelRequest):
    global MODEL_VERSION, MODEL_URI, model

    if not req.model_version and not req.model_stage:
        return {"error": "You must provide either model_version or model_stage"}

    if req.model_version:
        MODEL_VERSION = req.model_version
        MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    

    # Reload the model in memory
    model = mlflow.pyfunc.load_model(MODEL_URI)

    return {
        "status": "success",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_uri": MODEL_URI
    }


# TODO Add endpoint to update the serving version
# TODO Predict using the correct served version
