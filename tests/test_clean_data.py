import unittest
from unittest.mock import patch, MagicMock
from sftk.clean_data import ScientificNameProcessing, ScientificNameEntry 
import requests

class TestScientificNameProcessing(unittest.TestCase):

    def setUp(self):
        # Set up any common test data
        self.valid_scientific_name = "Kathetostoma giganteum"
        self.invalid_scientific_name = "Kathetostoma giganteu"
        self.sp_name = "Triglidae sp"
        self.common_name = "Giant stargazer"
        self.processor = ScientificNameProcessing(
            scientific_name_to_check=self.valid_scientific_name,
            common_name=self.common_name
        )

    def test_clean_name_valid(self):
        cleaned_name = self.processor.clean_name()
        self.assertEqual(cleaned_name, self.valid_scientific_name)

    def test_clean_name_with_sp(self):
        self.processor.scientific_name_to_check = self.sp_name
        cleaned_name = self.processor.clean_name()
        self.assertEqual(cleaned_name, "Triglidae")
        self.assertEqual(self.processor.higher_taxon, " sp")

    def test_clean_name_invalid_type(self):
        self.processor.scientific_name_to_check = 1234  # Invalid input (int)
        cleaned_name = self.processor.clean_name()
        self.assertIsNone(cleaned_name) 
        
    
    @patch("requests.get")
    def test_query_api_success(self, mock_get):
        # Mock the response from the API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "AphiaID": 123456,
            "valid_name": "Kathetostoma giganteum",
            "rank": "Species"
        }]
        mock_get.return_value = mock_response

        entry = self.processor.query_api()

        self.assertIsInstance(entry, ScientificNameEntry)
        self.assertEqual(entry.aphia_id, 123456)
        self.assertEqual(entry.scientific_name, "Kathetostoma giganteum")
        self.assertEqual(entry.scientific_names_match, True)
        self.assertEqual(entry.taxon_rank, "Species")

    @patch("requests.get")
    def test_query_api_no_results(self, mock_get):
        # Mock API response with no results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        entry = self.processor.query_api()

        self.assertIsInstance(entry, ScientificNameEntry)
        self.assertEqual(entry.scientific_name_to_check, self.valid_scientific_name)
        self.assertEqual(entry.scientific_name, None)
        self.assertEqual(entry.scientific_names_match, False)

    @patch("requests.get")
    def test_query_api_error(self, mock_get):
        # Simulate an error in the request (e.g., connection error)
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        entry = self.processor.query_api()

        self.assertIsInstance(entry, ScientificNameEntry)
        self.assertEqual(entry.scientific_name_to_check, self.valid_scientific_name)
        self.assertEqual(entry.scientific_name, None)

    @patch("requests.get")
    def test_query_api_invalid_json(self, mock_get):
        # Mock an invalid JSON response from the API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        entry = self.processor.query_api()

        self.assertIsInstance(entry, ScientificNameEntry)
        self.assertEqual(entry.scientific_name_to_check, self.valid_scientific_name)
        self.assertEqual(entry.scientific_name, None)

    def test_failed_to_process(self):
        # Test if the failed_to_process method logs the correct warning and returns the expected value
        with self.assertLogs(level='WARNING') as log:
            result = self.processor.failed_to_process("Test error")
            self.assertIn("Test error", log.output[0])  # Ensure the error message is logged

        self.assertIsInstance(result, ScientificNameEntry)
        self.assertEqual(result.scientific_name_to_check, self.valid_scientific_name)
        self.assertEqual(result.scientific_name, None)