import time
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import requests
from pydantic import BaseModel, Field, field_validator


class TestConfig(BaseModel):
    base_url: str = "localhost"
    max_response_time: float = 1.5  # seconds
    timeout: float = 2.0  # seconds


# Output
class Explanation(BaseModel):
    human_readable_reason: Optional[str] = None
    offending_products: Optional[List[str]] = None


class FraudPrediction(BaseModel):
    version: str
    is_fraud: bool
    fraud_proba: Optional[float] = Field(None, ge=0, le=1)
    estimated_damage: Optional[float] = None
    explanation: Optional[Explanation] = None

    @field_validator("version")
    @classmethod
    def validate_semantic_version(cls, v):
        """Validate semantic versioning format."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must follow semantic versioning (x.y.z)")
        return v


# Input
class TransactionLine(BaseModel):
    id: int
    product_id: UUID
    timestamp: datetime
    pieces_or_weight: float
    sales_price: float
    was_voided: bool
    camera_product_similar: bool
    camera_certainty: float


class TransactionHeader(BaseModel):
    store_id: UUID
    cash_desk: int
    transaction_start: datetime
    transaction_end: datetime
    total_amount: float
    payment_medium: str
    customer_feedback: Optional[int] = None


class FraudPredictionRequest(BaseModel):
    transaction_header: TransactionHeader
    transaction_lines: List[TransactionLine]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class APIClient:
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        try:
            response = self.session.get(
                self.config.base_url, timeout=self.config.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def predict_fraud(
        self, request_data: FraudPredictionRequest
    ) -> tuple[FraudPrediction, float]:
        """Make fraud prediction request and measure response time"""

        url = f"{self.config.base_url}/fraud-prediction"

        start_time = time.time()
        response = self.session.post(
            url, data=request_data.model_dump_json(), timeout=self.config.timeout
        )
        response_time = time.time() - start_time

        response.raise_for_status()
        prediction = FraudPrediction.model_validate(response.json())
        return prediction, response_time


def main():
    data = """
    {
      "transaction_header": {
        "store_id": "9aaa4004-eacd-425d-89ac-5e3d64dc9426",
        "cash_desk": 1,
        "transaction_start": "2023-09-13T17:29:51",
        "transaction_end": "2023-09-13T17:31:56",
        "total_amount": 29.90,
        "payment_medium": "CREDIT_CARD",
        "customer_feedback": 4
      },
      "transaction_lines": [
        {
          "id": 1,
          "product_id": "87654321-4321-8765-cbaa-210987654321",
          "timestamp": "2023-09-13T17:30:00",
          "pieces_or_weight": 2.0,
          "sales_price": 15.50,
          "was_voided": false,
          "camera_product_similar": true,
          "camera_certainty": 0.95
        },
        {
          "id": 2,
          "product_id": "87654321-4321-8765-cbaa-210987654321",
          "timestamp": "2023-09-13T17:30:00",
          "pieces_or_weight": 2.0,
          "sales_price": 15.50,
          "was_voided": false,
          "camera_product_similar": true,
          "camera_certainty": 0.95
        }
      ]
    }"""
    request_data = FraudPredictionRequest.model_validate_json(data)

    client = APIClient(TestConfig(base_url="http://localhost:8080"))
    print("Check if API is healthy...")
    if not client.health_check():
        print("Health check failed.")
        return

    print("Health check succeeded.")

    res = client.predict_fraud(request_data)
    print("Response time:", res[1])
    print("Prediction:", res[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
