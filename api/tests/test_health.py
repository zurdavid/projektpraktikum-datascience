import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_healthy(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"version": "0.1.2", "status": "healthy"}
