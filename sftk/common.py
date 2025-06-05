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

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

S3_SHAREPOINT_PATH = os.path.join("spyfish_metadata", "sharepoint_lists")
# TODO: keep either DEPLOYMENT_CSV vs MOVIES_CSV
S3_SHAREPOINT_DEPLOYMENT_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Deployment.csv")
S3_SHAREPOINT_MOVIE_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Deployment.csv")
S3_SHAREPOINT_DEFINITIONS_CSV = os.path.join(
    S3_SHAREPOINT_PATH, "BUV Metadata Definitions.csv"
)
S3_SHAREPOINT_SITE_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Survey Sites.csv")
S3_SHAREPOINT_SPECIES_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Species.csv")
S3_SHAREPOINT_SURVEY_CSV = os.path.join(S3_SHAREPOINT_PATH, "BUV Survey Metadata.csv")
S3_SHAREPOINT_RESERVES_CSV = os.path.join(S3_SHAREPOINT_PATH, "Marine Reserves.csv")
S3_SHAREPOINT_TEST_CSV = os.path.join(S3_SHAREPOINT_PATH, "Test.csv")

# S3 KSO Files
S3_KSO_PATH = os.path.join("spyfish_metadata", "kso_csvs")
S3_KSO_ANNOTATIONS_CSV = os.path.join(S3_KSO_PATH, "annotations_buv_doc.csv")
S3_KSO_MOVIE_CSV = os.path.join(S3_KSO_PATH, "movies_buv_doc.csv")
S3_KSO_SITE_CSV = os.path.join(S3_KSO_PATH, "sites_buv_doc.csv")
S3_KSO_SPECIES_CSV = os.path.join(S3_KSO_PATH, "species_buv_doc.csv")
S3_KSO_SURVEY_CSV = os.path.join(S3_KSO_PATH, "surveys_buv_doc.csv")
S3_KSO_TEST_CSV = os.path.join(S3_KSO_PATH, "test_buv_doc.csv")


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

MOVIE_EXTENSIONS = [
    "avi",
    "mov",
    "mp4",
    "mpg",
    "wmv",
]
