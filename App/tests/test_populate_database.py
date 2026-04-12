import os
import pytest 
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from App.populate_database import main

def mock_args(**kwargs):
    return SimpleNamespace(**kwargs)

'''Unit Test'''


@patch("App.populate_database.SupabaseVectorStore.from_documents")
@patch("App.populate_database.RecursiveCharacterTextSplitter")
@patch("App.populate_database.PyPDFLoader")
@patch("App.populate_database.create_client")
@patch("App.populate_database.os.path.exists", return_value=True)
@patch("App.populate_database.load_dotenv")
def test_successful_upload_unit(
    mock_load_dotenv,
    mock_exists,
    mock_create_client,
    mock_loader,
    mock_splitter,
    mock_from_documents,
):

    mock_doc = MagicMock()
    mock_doc.page_content = "hello world"
    mock_loader.return_value.load.return_value = [mock_doc]

    mock_chunk = MagicMock()
    mock_chunk.page_content = "chunk text"
    mock_chunk.metadata = {}

    splitter_instance = MagicMock()
    splitter_instance.split_documents.return_value = [mock_chunk]
    mock_splitter.return_value = splitter_instance

    with patch("App.populate_database.HuggingFaceEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()

        with patch("App.populate_database.argparse.ArgumentParser.parse_args") as mock_args_fn:
            mock_args_fn.return_value = mock_args(
                pdf="fake.pdf",
                table="documents",
                chunk_size=500,
                chunk_overlap=50,
                model="model"
            )

            main()

    mock_from_documents.assert_called_once()






'''Integration Tests'''
@patch("App.populate_database.os.path.exists", return_value = False )
@patch("App.populate_database.print")
def test_file_not_found(mock_print, mock_exists):
    with patch("App.populate_database.argparse.ArgumentParser.parse_args") as mock_args_fn:
        mock_args_fn.return_value =  mock_args(
            pdf = "fake.pdf", 
            table = "documents", 
            chunk_size = 500,
            chunk_overlap = 50, 
            model = "model"
        )

        main()
    mock_print.assert_any_call("File not found: fake.pdf")

@patch("App.populate_database.SupabaseVectorStore.from_documents")
@patch("App.populate_database.RecursiveCharacterTextSplitter")
@patch("App.populate_database.PyPDFLoader")

@patch("App.populate_database.create_client")
@patch("App.populate_database.os.path.exists", return_value = True)
@patch("App.populate_database.load_dotenv")

def test_no_chunks(
    mock_load_dotenv, 
    mock_exists, 
    mock_create_client, 
    mock_loader, 
    mock_splitter,
    mock_from_documents,
):
    mock_loader.return_value.load.return_value = [MagicMock(page_content="text")]

    splitter_instance = MagicMock()
    splitter_instance.split_documents.return_value = []
    mock_splitter.return_value = splitter_instance

    with patch("App.populate_database.argparse.ArgumentParser.parse_args") as mock_args_fn:
        mock_args_fn.return_value =  mock_args(
            pdf = "fake.pdf", 
            table = "documents", 
            chunk_size = 500,
            chunk_overlap = 50, 
            model = "model"
        )

        main()
    mock_from_documents.assert_not_called()

@patch("App.populate_database.SupabaseVectorStore.from_documents")
@patch("App.populate_database.RecursiveCharacterTextSplitter")
@patch("App.populate_database.PyPDFLoader")
@patch("App.populate_database.HuggingFaceEmbeddings")
@patch("App.populate_database.create_client")
@patch("App.populate_database.os.path.exists", return_value = True)
@patch("App.populate_database.load_dotenv")



def test_successful_upload(
    mock_load_dotenv, 
    mock_exists, 
    mock_create_client, 
    mock_embeddings,
    mock_loader, 
    mock_splitter,
    mock_from_documents,
):  
    
    mock_loader.return_value.load.return_value = [
        MagicMock(page_content="hello world")
    ]

    mock_chunk = MagicMock()
    mock_chunk.page_content = "chunk text"
    mock_chunk.metadata = {}

    splitter_instance = MagicMock()
    splitter_instance.split_documents.return_value = [mock_chunk]
    mock_splitter.return_value = splitter_instance

    mock_embeddings.return_value = MagicMock()

    with patch("App.populate_database.argparse.ArgumentParser.parse_args") as mock_args_fn:
        mock_args_fn.return_value =  mock_args(
            pdf = "fake.pdf", 
            table = "documents", 
            chunk_size = 500,
            chunk_overlap = 50, 
            model = "model"
        )

        main()
    mock_from_documents.assert_called_once()

@patch("App.populate_database.SupabaseVectorStore.from_documents", side_effect=Exception("DB error"))
@patch("App.populate_database.RecursiveCharacterTextSplitter")
@patch("App.populate_database.PyPDFLoader")
@patch("App.populate_database.HuggingFaceEmbeddings")
@patch("App.populate_database.create_client")
@patch("App.populate_database.os.path.exists", return_value = True)
@patch("App.populate_database.load_dotenv")
@patch("App.populate_database.print")


def test_upload_exception(
    mock_print,
    mock_load_dotenv, 
    mock_exists, 
    mock_create_client, 
    mock_embeddings,
    mock_loader, 
    mock_splitter,
    mock_from_documents,

):  

    mock_loader.return_value.load.return_value = [MagicMock(page_content="text")]

    mock_chunk = MagicMock()
    mock_chunk.page_content = "chunk"
    mock_chunk.metadata = {}

    splitter_instance = MagicMock()
    splitter_instance.split_documents.return_value = [mock_chunk]
    mock_splitter.return_value = splitter_instance

    mock_embeddings.return_value = MagicMock()

    with patch("App.populate_database.argparse.ArgumentParser.parse_args") as mock_args_fn:
        mock_args_fn.return_value =  mock_args(
            pdf = "fake.pdf", 
            table = "documents", 
            chunk_size = 500,
            chunk_overlap = 50, 
            model = "model"
        )

        main()
    assert any(
        "Error inserting documents" in str(call.args)
        for call in mock_print.call_args_list
    )