from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

from App.main import app

client = TestClient(app)

@pytest.mark.system
@patch("App.main.ask_aichatbot_payroll_question")
def test_full_system_chat(mock_ask):
    mock_ask.return_value = (
        "This is the system test answer", 
        ["https://example.com/image.png"]
    )

    payload = {
        "model": "payroll-rag", 
        "messages" : [
            {
                "role": "user", "content": "What is payroll?"
            }
        ], 
        "stream": False
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data

    assert data["choices"][0]["message"]["content"]

    content = data["choices"][0]["message"]["content"]
    assert "This is the system test answer" in content

    assert "![image](https://example.com/image.png)" in content

    mock_ask.assert_called_once()

@pytest.mark.system
@patch("App.main.ask_aichatbot_payroll_question")
def test_streaming_response(mock_ask):
    mock_ask.return_value= (
        "Streaming answer",
        []
    )
    payload = {
        "model": "payroll-rag", 
        "messages" : [
            {
                "role": "user", "content": "What is payroll?"
            }
        ], 
        "stream": True
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
