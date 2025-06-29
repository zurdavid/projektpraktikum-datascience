import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def get(client, path):
    with open(path, "r") as f:
        payload = json.load(f)
    response = client.post("/fraud-prediction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "0.1.2"
    return data


def test_fraud_case(client):
    path = "./tests/json/fraud-1be63d50-7cff-4ed7-922e-68929b99036f.json"
    data = get(client, path)
    assert data["is_fraud"] is True
    assert data["fraud_proba"] > 0.0
    assert data["estimated_damage"] > 0.0
    assert data["explanation"]["human_readable_reason"].startswith(
        "Das Modell hat einen möglichen Betrugsfall erkannt:"
    )


def test_non_fraud_case(client):
    path = "./tests/json/non_fraud-7ba24ec9-37fc-4736-a04e-2d1898b2ebef.json"
    data = get(client, path)
    assert data["is_fraud"] is False
    assert data["fraud_proba"] >= 0.0
    assert data["estimated_damage"] >= 0.0
    assert data["explanation"] is None


def test_unscanned_products(client):
    path = "./tests/json/unscanned_products-77578667-8427-46de-8d08-4008d551d770.json"
    data = get(client, path)
    assert data["is_fraud"] is True
    assert data["fraud_proba"] == 1.0
    assert data["estimated_damage"] > 0.0
    assert (
        data["explanation"]["human_readable_reason"]
        == "Kamera hat nicht gescannte Produkte erkannt"
    )
    assert data["explanation"]["offending_products"] == [
        "413afc83-1903-4984-ab10-4fc02e4182fc"
    ]


def test_unscanned_missing_products(client):
    path = "./tests/json/unscanned_products_product_id_missing-b5450bf4-a819-4e95-8de7-9b29220e7281.json"
    data = get(client, path)
    assert data["is_fraud"] is True
    assert data["fraud_proba"] == 1.0
    assert data["estimated_damage"] is None
    assert (
        data["explanation"]["human_readable_reason"]
        == "Kamera hat nicht gescannte Produkte erkannt"
    )
    assert data["explanation"]["offending_products"] == ["unbekannt"]


def test_excluded_discount_fraud():
    app = create_app("./tests/test_config.toml")
    client = TestClient(app)
    path = (
        "./tests/json/excluded_discount_fraud-2835d96f-f2c4-4ec9-befc-dba05873a5d6.json"
    )
    data = get(client, path)
    assert data["is_fraud"] is True
    assert data["fraud_proba"] == 1.0
    assert data["estimated_damage"] > 0.0
    assert (
        data["explanation"]["human_readable_reason"]
        == "Rabatt auf von Rabatt ausgenommene Produkte angewendet"
    )
    assert data["explanation"]["offending_products"] == [
        "a5dd1a6b-48cf-41cd-a489-570a044f5c6b (BEVERAGES)"
    ]


def test_transaction_without_lines(client):
    path = "./tests/json/no_lines-cce5317f-e4b6-4890-b776-fd9838730ae2.json"
    data = get(client, path)
    assert data["is_fraud"] is True
    assert (
        data["explanation"]["human_readable_reason"]
        == "Keine Transaktionszeilen vorhanden"
    )


def test_camera_values_missing(client):
    path = (
        "./tests/json/camera_values_missing-b19f3361-59e3-4644-9299-f3484fdfd007.json"
    )
    data = get(client, path)
    assert data["is_fraud"] is False
