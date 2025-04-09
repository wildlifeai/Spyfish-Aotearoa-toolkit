import os
from dotenv import load_dotenv

def load_env_wrapper() -> None:
    """
    Wrapper for loading environment variables from a .env file.

    Guard clause added to prevent loading environment variables when running in a GitHub Actions environment.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        return

    load_dotenv()


load_env_wrapper()

# Email configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
EMAIL_ARCHIVE_FOLDER_CANDIDATES = [
    "[Gmail]/All Mail",  # Gmail
    "Archive",  # Common IMAP
    "Archived",  # Alternative naming
    "INBOX.Archive",  # Some email systems
]
EMAIL_ARCHIVE_FOLDER = "[Gmail]/All Mail"

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_SHAREPOINT_CSVS = os.getenv("S3_SHARPOINT_CSVS")
S3_SHAREPOINT_MOVIE_CSV = os.getenv("S3_SHAREPOINT_MOVIE_CSV")
S3_SHAREPOINT_SURVEY_CSV = os.getenv("S3_SHAREPOINT_SURVEY_CSV")
S3_SHAREPOINT_SITE_CSV = os.getenv("S3_SHAREPOINT_SITE_CSV")
S3_SHAREPOINT_SPECIES_CSV = os.getenv("S3_SHAREPOINT_SPECIES_CSV")

# Keywords to monitor in email
KEYWORDS = [
    "survey",
    "site",
    "movie",
    "species",
]
KEYWORD_LOOKUP = {
    "survey": "SurveyID",
    "site": "SiteID",
    "movie": "DropID",
    "species": "ScientificName",
}
