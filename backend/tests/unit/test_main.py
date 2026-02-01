import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
from app.database import get_db, get_redis_client

client = TestClient(app)

# Mocks
def mock_get_db():
    try:
        db = MagicMock()
        db.execute.return_value = None
        yield db
    finally:
        pass

def mock_get_redis():
    r = MagicMock()
    r.ping.return_value = True
    return r

# Override dependencies
app.dependency_overrides[get_db] = mock_get_db
app.dependency_overrides[get_redis_client] = mock_get_redis

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Welcome to IES_EV API"

def test_health_check_success():
    # With mocks in place, should return 200 and healthy
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["redis"] == "connected"

@patch("app.services.ai_service.ai_service.generate_response")
def test_chat_ai_success(mock_generate):
    # Mock AI response
    mock_response_data = {
        "choices": [
            {
                "message": {
                    "content": "Hello, this is a test response."
                }
            }
        ]
    }
    mock_generate.return_value = mock_response_data
    
    payload = {"message": "Hello AI"}
    response = client.post("/api/ai/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello, this is a test response."
    assert "raw" in data

@patch("app.services.ai_service.ai_service.generate_response")
def test_chat_ai_failure(mock_generate):
    # Mock exception
    from fastapi import HTTPException
    mock_generate.side_effect = HTTPException(status_code=503, detail="AI Service Error")
    
    payload = {"message": "Fail please"}
    response = client.post("/api/ai/chat", json=payload)
    
    assert response.status_code == 503
    assert response.json()["detail"] == "AI Service Error"
