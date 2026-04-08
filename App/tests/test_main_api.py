import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from fastapi.responses import StreamingResponse

from App.main import app

client = TestClient(app)

'''Unit Tests'''

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_unit_success(mock_llm):
    mock_llm.return_value = ("Unit Answer", [])

    response = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    })

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "Unit Answer"

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_unit_with_images(mock_llm):
    mock_llm.return_value = ("Answer", ["img1.png"])

    response = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "Show images"}
        ]
    })

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    assert "Answer" in content
    assert "![image](img1.png)" in content
def test_chat_unit_no_user_message():
    response = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "assistant", "content": "Hi"}
        ]
    })

    assert response.status_code == 200
    assert response.json()["error"] == "No user message found"

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_unit_streaming(mock_llm):
    mock_llm.return_value = ("Hello World", [])

    response = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": True
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "data:" in body
    assert "[DONE]" in body


@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_unit_llm_failure(mock_llm):
    mock_llm.side_effect = Exception("LLM crashed")

    response = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    })

    assert response.status_code == 200
    assert "error" in response.json()


''' Integration Test '''

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "payroll-rag"

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_success(mock_llm):
    mock_llm.return_value = ("Test answer", [])

    response = client.post("/v1/chat/completions",
                            json={
            "model": "payroll-rag",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": False
        }
    ) 
    assert response.status_code == 200
    data = response.json()
    assert  "choices" in data
    assert data["choices"][0]["message"]["content"] == "Test answer"


@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_with_images(mock_llm):
    mock_llm.return_value = ("Answer", ["img1.png", "img2.png"])

    response = client.post("/v1/chat/completions",
                            json={

            "messages": [
                {"role": "user", "content": "Show images"}
            ],
         
        }
    ) 
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    assert "Answer" in content
    assert "![image](img1.png)" in content
    assert "![image](img2.png)" in content


def test_chat_no_user_message():
    response = client.post("/v1/chat/completions",
                            json={

            "messages": [
                {"role": "assistant", "content": "Hi"}
            ],
          
        }
    ) 
    assert response.status_code == 200
    assert response.json()["error" ] == "No user message found"

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_streaming(mock_llm):
    mock_llm.return_value = ("Hello World", [])
    
    response = client.post("/v1/chat/completions",
                            json={
         
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": True
        }
    ) 
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text

    assert "data:" in body
    assert "[DONE]" in body

@patch("App.main.ask_aichatbot_payroll_question")
def test_chat_llm_failure(mock_llm):
    mock_llm.side_effect = Exception("LLM crashed")
   

    response = client.post("/v1/chat/completions",
                            json={
         
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
           
        }
    ) 

    assert response.status_code == 200
    assert "error" in response.json()
    

    
    