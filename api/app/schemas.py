"""
schemas.py

Module for defining the data schemas used in fraud detection requests and responses.

Classes:
    Explanation: Schema for providing human-readable explanations of fraud predictions.
    FraudPrediction: Schema for the fraud prediction response, including version, fraud status, probability, estimated damage, and explanation.
    TransactionLine: Schema for individual transaction lines with details like product ID, timestamp, sales price, etc.
    TransactionHeader: Schema for the transaction header containing store ID, cash desk number, transaction times, total amount, payment medium, and customer feedback.
    FraudPredictionRequest: Schema for the request containing transaction header and lines.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


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
    product_id: Optional[UUID] = None
    timestamp: datetime
    pieces_or_weight: float
    sales_price: float
    was_voided: bool
    camera_product_similar: Optional[bool] = None
    camera_certainty: Optional[float] = None


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

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
