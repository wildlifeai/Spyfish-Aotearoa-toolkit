import logging
from collections import defaultdict
from sftk.email_handler import EmailHandler
from sftk.common import KEYWORDS, KEYWORD_LOOKUP
from sftk import log_config
from sftk.utils import flatten_list

class DataIngression:
    """
    Base class for Data Ingression processes
    Empty for now
    May in future provide functionality such as:
    - Logging ingression processes and providing summaries

    Could be standardised to provide input -> process -> output functionality
    This would allow us to standardise the way data is ingressed into the system
    """
    def __init__(self):
        pass

    def run(self):
        pass

    def cleanup(self):
        pass

class EmailIngression(DataIngression):
    """
    JSON  data ingressed from emails
    NOTE: Once an email is opened imap will mark it as read
    This means if an email FAILS to be processed it will be marked as read and not picked up again
    This is why summary logging is important for record keeping
    The original email_to_sharepoint_copy.py script does not handle this
    Nor does it actually use the move_to_archive method to archive processed emails
    """
    def __init__(self):
        super().__init__()
        # Email handler with default values
        self.email_handler = EmailHandler()

    def run(self):
        super().run()
        keyword_emails = self.get_inbox()

        data = defaultdict(list)
        for keyword, email_ids in keyword_emails.items():
            for id in email_ids:
                datum = self.extract_json_from_email_id(id)
                if datum:
                    data[keyword].append(datum)
                    logging.info(f"Extracted data from email with ID: {id}")
                else:
                    logging.warning(f"No data extracted from email with ID: {id}")

        all_ids = flatten_list(list(keyword_emails.values()))
        self.cleanup(all_ids)

        return data

    def cleanup(self, email_ids: list):
        """
        Cleans up by moving emails to the archive folder

        Args:
            email_ids (list): A list of email IDs to move to the archive folder.

        Returns:
            None
        """
        super().cleanup()
        for id in email_ids:
            try:
                self.email_handler.move_to_archive(id)
                logging.info(f"Moved email with ID: {id} to the archive folder.")
            except Exception as e:
                logging.error(f"Failed to move email with ID: {id} to the archive folder: {e}")

        self.email_handler.expunge()
        self.email_handler.close()

    def get_inbox(self, keywords: list = KEYWORDS) -> dict[str, list]:
        """
        Get all emails from the inbox.

        Args:
            keywords (list): A list of keywords to search for in the email subject.

        Returns:
            dict: A dictionary containing the keyword and the list of email IDs that contain the keyword.
        """
        keyword_emails = {}
        for k in keywords:
            email_ids = self.email_handler.search(None, "UNSEEN", "SUBJECT", f'{k}')
            if email_ids:
                keyword_emails[k] = email_ids
                logging.info(f"Found {len(email_ids)} emails with the keyword ''.")

        return keyword_emails

    def extract_json_from_email_id(self, id: str):
        """
        Takes in an email ID then extracts and parses the JSON data from the email body

        Args:
            id (str): The email ID

        Returns:
            dict | None: The extracted JSON data or None if no data was extracted
        """
        message = self.email_handler.get_email_content(id)
        # Exception raised when the email is empty
        if message is None:
            return None
        body = message['body']
        datum = self.email_handler.extract_json_from_body(body)
        return datum
