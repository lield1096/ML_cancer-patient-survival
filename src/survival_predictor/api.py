"""REST API serving the trained model.

Run locally:  uvicorn survival_predictor.api:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .predict import SurvivalModel

logger = logging.getLogger(__name__)

MAX_BATCH = 1000


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120, examples=[55])
    tumor_size: float = Field(..., ge=0, le=500, description="Tumor size in mm", examples=[30])
    regional_nodes_examined: int = Field(..., ge=0, examples=[14])
    regional_nodes_positive: int = Field(..., ge=0, examples=[3])

    @model_validator(mode="after")
    def positive_not_more_than_examined(self) -> PatientFeatures:
        if self.regional_nodes_positive > self.regional_nodes_examined:
            raise ValueError("regional_nodes_positive cannot exceed regional_nodes_examined")
        return self


class PredictionResponse(BaseModel):
    probability_dead: float
    predicted_status: Literal["Alive", "Dead"]


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "models/model.joblib")
    try:
        app.state.model = SurvivalModel.load(path)
        logger.info("Loaded model '%s' from %s", app.state.model.model_name, path)
    except FileNotFoundError:
        logger.error("No model at %s - /predict will return 503 until one is trained", path)
        app.state.model = None
    yield


app = FastAPI(
    title="Cancer Survival Predictor",
    version=__version__,
    description="Predicts survival status from clinical features. Research demo only - "
    "not for clinical use.",
    lifespan=lifespan,
)


def _model(request: Request) -> SurvivalModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": request.app.state.model is not None}


@app.get("/model")
def model_info(request: Request) -> dict:
    model = _model(request)
    return {
        "model_name": model.model_name,
        "features": model.features,
        "threshold": model.threshold,
        "test_metrics": model.test_metrics,
        "metadata": model.metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    pred = _model(request).predict([patient.model_dump()])[0]
    return PredictionResponse(**vars(pred))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    preds = _model(request).predict([p.model_dump() for p in batch.patients])
    return BatchResponse(predictions=[PredictionResponse(**vars(p)) for p in preds])
