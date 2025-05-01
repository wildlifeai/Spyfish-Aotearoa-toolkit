import os

from dotenv import load_dotenv


def load_env_wrapper() -> None:
    """
    Wrapper for loading environment variables from a .env file.

    Guard clause added to prevent loading environment variables when running in
    a GitHub Actions environment.
    """
    if os.getenv("GITHUB_ACTIONS") == "true":
        return

    load_dotenv(override=True)


load_env_wrapper()


# General settings
DEV_MODE = os.getenv("DEV_MODE")

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

# S3 Sharepoint Files
S3_SHAREPOINT_PATH = os.getenv("S3_SHAREPOINT_PATH")
S3_SHAREPOINT_DEFINITIONS_CSV = os.getenv("S3_SHAREPOINT_DEFINITIONS_CSV")
#  TODO check if they are both used: MOVIE_CSV or DEPLOYMENT_CSV
S3_SHAREPOINT_DEPLOYMENT_CSV = os.getenv("S3_SHAREPOINT_DEPLOYMENT_CSV")
S3_SHAREPOINT_MOVIE_CSV = os.getenv("S3_SHAREPOINT_DEPLOYMENT_CSV")
S3_SHAREPOINT_RESERVES_CSV = os.getenv("S3_SHAREPOINT_RESERVES_CSV")
S3_SHAREPOINT_SITE_CSV = os.getenv("S3_SHAREPOINT_SITE_CSV")
S3_SHAREPOINT_SPECIES_CSV = os.getenv("S3_SHAREPOINT_SPECIES_CSV")
S3_SHAREPOINT_SURVEY_CSV = os.getenv("S3_SHAREPOINT_SURVEY_CSV")
S3_SHAREPOINT_TEST_CSV = os.getenv("S3_SHAREPOINT_TEST_CSV")

# S3 KSO Files
S3_KSO_PATH = os.getenv("S3_KSO_PATH")
S3_KSO_ANNOTATIONS_CSV = os.getenv("S3_KSO_ANNOTATIONS_CSV")
S3_KSO_MOVIE_CSV = os.getenv("S3_KSO_MOVIE_CSV")
S3_KSO_SITE_CSV = os.getenv("S3_KSO_SITE_CSV")
S3_KSO_SPECIES_CSV = os.getenv("S3_KSO_SPECIES_CSV")
S3_KSO_SURVEY_CSV = os.getenv("S3_KSO_SURVEY_CSV")
S3_KSO_TEST_CSV = os.getenv("S3_KSO_TEST_CSV")

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
