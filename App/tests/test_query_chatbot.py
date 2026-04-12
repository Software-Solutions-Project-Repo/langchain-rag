import pytest
from unittest.mock import patch, MagicMock
from App.query_data import query_rag
from App.LLM import ask_aichatbot_payroll_question 


'''Unit Tests'''
def test_empty_question():
    result = ask_aichatbot_payroll_question("", [])
    assert result == "No question provided"


@patch("App.LLM.chain")
@patch("App.LLM.query_rag")

def test_successful_response(mock_query_rag, mock_chain):
    mock_query_rag.return_value = [
        {"content": "Payroll info", "similarity": 0.9}
    ]

    mock_output = MagicMock()
    mock_output.content = "This is the answer"

    mock_chain.invoke.return_value = mock_output

    answer, images = ask_aichatbot_payroll_question("What is payroll?", [])

    assert answer == "This is the answer"
    assert isinstance(images,list)

    mock_chain.invoke.assert_called_once()

@patch("App.LLM.chain")
@patch("App.LLM.query_rag")

def test_chat_history(mock_query_rag, mock_chain):
    mock_query_rag.return_value = []

    mock_output = MagicMock()
    mock_output.content = "OK"

    mock_chain.invoke.return_value = mock_output
    chat_history = ["User: hi"]

    ask_aichatbot_payroll_question("Hello", chat_history)

    args, kwargs= mock_chain.invoke.call_args

    assert mock_chain.invoke.called
    
    combined = str(args) + str(kwargs)

    assert "hi" in combined or "User" in combined

@patch("App.LLM.chain")
@patch("App.LLM.query_rag")

def test_llm_failure(mock_query_rag, mock_chain):
    mock_query_rag.return_value = []

    mock_chain.invoke.side_effect = Exception("LLM failed")

    result = ask_aichatbot_payroll_question("test", [])

    assert result == "Cannot Service your request. Sorry"


'''Integration Test'''

@patch("App.LLM.chain")  # Mock ONLY the LLM
@patch("App.LLM.query_rag")  # Mock the RAG data
def test_full_integration(mock_query_rag, mock_chain):

    
    mock_query_rag.return_value = [
        {
            "content": "Payroll is processed monthly",
            "similarity": 0.9
        },
        {
            "question": "What is payroll?",
            "answer": "Payroll is salary processing",
            "similarity": 0.95
        },
        {
            "error_code": "ERR001",
            "question": "DB error",
            "answer": "Check connection",
            "similarity": 0.85
        }
    ]


    mock_output = MagicMock()
    mock_output.content = "Final answer from LLM"

    mock_chain.invoke.return_value = mock_output

   
    answer, images = ask_aichatbot_payroll_question(
        "What is payroll?",
        ["User: What is payroll?", "Assistant: I don't know"]
    )

    assert answer == "Final answer from LLM"
    assert isinstance(images, list)

    mock_chain.invoke.assert_called_once()


    args, kwargs = mock_chain.invoke.call_args

    assert "context" in kwargs or "context" in args[0]
    assert "question" in kwargs or "question" in args[0]