from unittest.mock import MagicMock, patch

import pytest
import requests

from sftk.clean_data import ScientificNameEntry, ScientificNameProcessing


@pytest.fixture
def valid_scientific_name():
    return "Kathetostoma giganteum"


@pytest.fixture
def invalid_scientific_name():
    return "Kathetostoma giganteu"


@pytest.fixture
def sp_name():
    return "Triglidae sp"


@pytest.fixture
def common_name():
    return "Giant stargazer"


@pytest.fixture
def processor(valid_scientific_name, common_name):
    return ScientificNameProcessing(
        scientific_name_to_check=valid_scientific_name, common_name=common_name
    )


def test_clean_name_valid(processor, valid_scientific_name):
    cleaned_name = processor.clean_name()
    assert cleaned_name == valid_scientific_name


def test_clean_name_with_sp(processor, sp_name):
    processor.scientific_name_to_check = sp_name
    cleaned_name = processor.clean_name()
    assert cleaned_name == "Triglidae"
    assert processor.higher_taxon == " sp"


def test_clean_name_invalid_type(processor):
    processor.scientific_name_to_check = 1234  # Invalid input (int)
    cleaned_name = processor.clean_name()
    assert cleaned_name is None


@patch("requests.get")
def test_query_api_success(mock_get, processor):
    # Mock the response from the API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"AphiaID": 123456, "valid_name": "Kathetostoma giganteum", "rank": "Species"}
    ]
    mock_get.return_value = mock_response

    entry = processor.query_api()

    assert isinstance(entry, ScientificNameEntry)
    assert entry.aphia_id == 123456
    assert entry.scientific_name == "Kathetostoma giganteum"
    assert entry.scientific_names_match is True
    assert entry.taxon_rank == "Species"


@patch("requests.get")
def test_query_api_no_results(mock_get, processor, valid_scientific_name):
    # Mock API response with no results
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    entry = processor.query_api()

    assert isinstance(entry, ScientificNameEntry)
    assert entry.scientific_name_to_check == valid_scientific_name
    assert entry.scientific_name is None
    assert entry.scientific_names_match is False


@patch("requests.get")
def test_query_api_error(mock_get, processor, valid_scientific_name):
    # Simulate an error in the request (e.g., connection error)
    mock_get.side_effect = requests.exceptions.RequestException("Connection error")

    entry = processor.query_api()

    assert isinstance(entry, ScientificNameEntry)
    assert entry.scientific_name_to_check == valid_scientific_name
    assert entry.scientific_name is None


@patch("requests.get")
def test_query_api_invalid_json(mock_get, processor, valid_scientific_name):
    # Mock an invalid JSON response from the API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_response

    entry = processor.query_api()

    assert isinstance(entry, ScientificNameEntry)
    assert entry.scientific_name_to_check == valid_scientific_name
    assert entry.scientific_name is None


def test_failed_to_process(processor, valid_scientific_name, caplog):
    # Test if the failed_to_process method logs the correct warning and returns the expected value
    result = processor.failed_to_process("Test error")

    assert "Test error" in caplog.text  # Ensure the error message is logged
    assert isinstance(result, ScientificNameEntry)
    assert result.scientific_name_to_check == valid_scientific_name
    assert result.scientific_name is None
