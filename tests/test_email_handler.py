import imaplib
import sys
import os
from unittest import TestCase
from unittest.mock import patch, MagicMock
from sftk.email_handler import EmailHandler, Email

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir)

from mock_data.mock_email_data import MOCK_EMAIL_DATA

class TestEmailHandler(TestCase):
    # Fake imap connection
    @patch("imaplib.IMAP4_SSL")
    def setUp(self, mock_imap):
        """Set up a mock IMAP connection before each test."""
        self.mock_imap = mock_imap
        self.mock_mail = MagicMock()
        mock_imap.return_value = self.mock_mail

        # Fake IMAP login response
        self.mock_mail.login.return_value = ("OK", [])
        self.mock_mail.select.return_value = ("OK", [])

        # Fake list of mailboxes
        self.mock_mail.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Archive"'])

        # Initialize EmailHandler with mock
        self.handler = EmailHandler(email="test@example.com", password="password")

    def tearDown(self):
        """Clean up after each test."""
        self.handler.close()

    def test_login_failure(self):
        """Test that login failure raises an exception."""
        self.mock_mail.login.return_value = ("NO", [b"Authentication failed"])

        with self.assertRaises(imaplib.IMAP4.error):
            self.handler.login(email="wrong@example.com", password="wrongpassword", mailbox="INBOX")

    def test_search_emails(self):
        """Test searching for emails."""
        self.mock_mail.search.return_value = ("OK", [b"1 2 3"])

        result = self.handler.search(None, "UNSEEN")
        self.assertEqual(result, [b"1", b"2", b"3"])

    def test_search_no_results(self):
        """Test searching for emails when none match."""
        self.mock_mail.search.return_value = ("OK", [b""])

        result = self.handler.search(None, "UNSEEN")
        self.assertEqual(result, [], f"Empty list should be returned when no emails match the search criteria. Got: {result}")

    def test_set_mailbox_not_found(self):
        """Test selecting a non-existent mailbox."""
        self.mock_mail.list.return_value = ("OK", [b'(\\HasNoChildren) "/" "INBOX"'])

        with self.assertRaises(ValueError):
            self.handler.set_mailbox("NotARealMailbox")

    def test_get_email_content_fetch_failure(self):
        """Test email retrieval when fetch fails."""
        self.mock_mail.fetch.return_value = ("NO", [])

        result = self.handler.get_email_content("1")
        self.assertIsNone(result, "None should be returned when the email cannot be fetched.")

    def test_expunge_success(self):
        """Test that expunge is called correctly."""
        self.mock_mail.expunge.return_value = ("OK", [])

        self.handler.expunge()
        self.mock_mail.expunge.assert_called_once()

    def test_move_to_archive(self):
        """Test moving an email to the archive."""
        self.mock_mail.copy.return_value = ("OK", [])
        self.mock_mail.store.return_value = ("OK", [])

        self.handler.move_to_archive("1")

        self.mock_mail.copy.assert_called_once_with("1", '"Archive"')
        self.mock_mail.store.assert_called_once_with("1", "+FLAGS", r"(\Deleted)")

    def test_get_email_content(self):
        """Test retrieving an email's content.

        Three different cases are tested:
        - An email with no JSON data
        - An email with JSON data
        - An email with broken JSON data
        """
        no_json_email = MOCK_EMAIL_DATA["email_no_json"]
        self.mock_mail.fetch.return_value = no_json_email

        email_obj = self.handler.get_email_content("1")
        self.assertEqual(email_obj.subject, "Test Subject", f"The email subject should be 'Test Subject', Got: {email_obj.subject}")
        self.assertEqual(email_obj.body, "This is a test email", f"The email body should be 'This is a test email', Got: {email_obj.body}")

        # Try and parse the json data - should return none if the email has no json data
        json_data = EmailHandler.extract_json_from_body(email_obj.body)
        self.assertIsNone(json_data, "No JSON data should be found in the email body, Got: {json_data}")

        json_email = MOCK_EMAIL_DATA["email_with_json"]
        self.mock_mail.fetch.return_value = json_email
        expected_result = {"key": "value"}
        email_obj = self.handler.get_email_content("1")
        json_data = EmailHandler.extract_json_from_body(email_obj.body)
        self.assertEqual(json_data, expected_result, f"The extracted JSON data should be {expected_result}, Got: {json_data}")

        # Broken json should raise an exception and return None
        broken_json_email = MOCK_EMAIL_DATA["email_broken_json"]
        self.mock_mail.fetch.return_value = broken_json_email
        email_obj = self.handler.get_email_content("1")

        json_data = EmailHandler.extract_json_from_body(email_obj.body)
        self.assertIsNone(json_data, "No JSON data should be found in the email body, Got: {json_data}")
