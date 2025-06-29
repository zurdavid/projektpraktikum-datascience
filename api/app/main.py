import logging
import tomllib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import datastore
from app.model import load_model
from app.schemas import (
    FraudPrediction,
    FraudPredictionRequest,
)
from app.version import __version__ as VERSION

log = logging.getLogger("uvicorn.app")


def create_app(config_path="config.toml") -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    config_file = Path(config_path)
    config = tomllib.loads(config_file.read_text("utf-8"))

    app = FastAPI(
        title="Fraud Detection API",
        description="Detecting fraud in self checkout transactions",
        version=VERSION,
    )
    app.state.data_store = datastore.load_datastore()
    app.state.model = load_model(config)
    register_routes(app)

    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def register_routes(app: FastAPI):
    @app.get("/")
    async def healthcheck():
        """Health check endpoint to verify if the API is running."""
        return {"status": "healthy", "version": VERSION}

    @app.post("/fraud-prediction")
    async def fraud_prediction(request: FraudPredictionRequest) -> FraudPrediction:
        stores_df = app.state.data_store.stores_df()
        products_df = app.state.data_store.products_df()
        return app.state.model.detect_fraud(request, stores_df, products_df)


app = create_app()
