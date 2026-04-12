import unittest 
from unittest.mock import patch, MagicMock
from App.query_data import query_rag


'''Unit Tests'''

class QueryRagTests(unittest.TestCase):
    @patch("App.query_data.get_embedding_function")
    @patch("App.query_data.create_client")

    def test_query_rag_success(self, mock_create_client, mock_get_embedding):

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_get_embedding.return_value = mock_embedder

        mock_response = MagicMock()
        mock_response.execute.return_value.data = [{"doc": "doc1"}]

        mock_question = MagicMock()
        mock_question.execute.return_value.data = [{"qa": "qa1"}]

        mock_error = MagicMock()
        mock_error.execute.return_value.data = [{"error": "err1"}]

        mock_client =MagicMock()
        mock_client.rpc.side_effect = [
            mock_response, 
            mock_question, 
            mock_error
        ]


        mock_create_client.return_value = mock_client

        result = query_rag("test")

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["doc"], "doc1")

    @patch("App.query_data.get_embedding_function")
    @patch("App.query_data.create_client")
    def test_query_rag_no_results(self, mock_create_client, mock_get_embedding):


        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_get_embedding.return_value = mock_embedder

        empty = MagicMock()
        empty.execute.return_value.data = []

        mock_client = MagicMock()
        mock_client.rpc.return_value = empty 
        mock_create_client.return_value = mock_client 

        result = query_rag("nothing")

        self.assertIsNone(result)


    @patch("App.query_data.get_embedding_function")
    @patch("App.query_data.create_client")
    def test_query_rag_embedding_failure(self, mock_create_client, mock_get_embedding):
        mock_embedder = MagicMock()
        #mock_embedder.embed_query.side_effect = Exception("fail")
        mock_embedder.embed_query.return_value = [0.1,0.2]
        mock_get_embedding.return_value = mock_embedder

        mock_rpc = MagicMock()
        mock_execute = MagicMock()
        mock_execute.data = []

        mock_rpc.execute.return_value =mock_execute

        mock_client =MagicMock()
        mock_client.rpc.return_value = mock_rpc
        mock_create_client.return_value = mock_client 

        result = query_rag("bad")

        self.assertIsNone(result,list)


'''Integration Test'''


@patch("App.query_data.get_embedding_function")
@patch("App.query_data.create_client")
def test_query_rag_integration(mock_create_client, mock_get_embedding):

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_get_embedding.return_value = mock_embedder

    mock_doc_response = MagicMock()
    mock_doc_response.execute.return_value.data = [{"doc": "doc1"}]

    mock_qa_response = MagicMock()
    mock_qa_response.execute.return_value.data = [{"qa": "qa1"}]

    mock_error_response = MagicMock()
    mock_error_response.execute.return_value.data = [{"error": "err1"}]

    mock_client = MagicMock()
    mock_client.rpc.side_effect = [
        mock_doc_response,
        mock_qa_response,
        mock_error_response
    ]

    mock_create_client.return_value = mock_client

   
    result = query_rag("test query")

    assert isinstance(result, list)
    assert len(result) == 3

    assert result[0]["doc"] == "doc1"
    assert result[1]["qa"] == "qa1"
    assert result[2]["error"] == "err1"


    mock_embedder.embed_query.assert_called_once_with("test query")


    assert mock_client.rpc.call_count == 3