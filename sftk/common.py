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

DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH")

# Biigle credentials
BIIGLE_API_EMAIL = os.getenv("BIIGLE_API_EMAIL")
BIIGLE_API_TOKEN = os.getenv("BIIGLE_API_TOKEN")  # api token get from ui

# S3 configuration.
# Ask for user input if the env variables are not found.
# TODO check ways to set variables, if there are issues reading the .env file
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


S3_SHAREPOINT_PATH = os.path.join("spyfish_metadata", "sharepoint_lists")
# Both 'Deployment' and 'Movies' exist because 'Deployment' is the term used in SharePoint,
# while 'Movies' is the equivalent term used in KSO (for example, accessed via keyword in
# the Sharepoint to kso copy workflow)
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
S3_KSO_ERRORS_CSV = os.path.join(S3_KSO_PATH, "errors_buv_doc.csv")
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


VALIDATION_RULES = {
    "deployments": {
        "file_name": S3_SHAREPOINT_DEPLOYMENT_CSV,
        # TODO add fps, sampling start and end etc.
        "required": ["DropID", "SurveyID", "SiteID", "FileName", "LinkToVideoFile"],
        "unique": ["DropID"],
        "info_columns": ["SurveyID", "SiteID"],
        "foreign_keys": {"surveys": "SurveyID", "sites": "SiteID"},
        "relationships": [
            # TODO fix replicate in BUV dep many empty vals
            # TODO fix the predefined 0, for now not a problem as no survey has more than 9 entries
            {
                "column": "DropID",
                "rule": "equals",
                "template": "{SurveyID}_{SiteID}_{ReplicateWithinSite:02}",
            },
            {
                "column": "FileName",
                "rule": "equals",
                "template": "{DropID}.mp4",
                # TODO: Remove null allowed
                "allowed_values": ["NO VIDEO BAD DEPLOYMENT"],
                "allow_null": True,
            },
            {
                "column": "LinkToVideoFile",
                "rule": "equals",
                "template": "media/{SurveyID}/{DropID}/{DropID}.mp4",
                # TODO: Remove null allowed
                "allowed_values": ["NO VIDEO BAD DEPLOYMENT"],
                "allow_null": True,
            },
        ],
    },
    "surveys": {
        "file_name": S3_SHAREPOINT_SURVEY_CSV,
        "required": ["SurveyID"],
        "unique": ["SurveyID"],
        "info_columns": ["SurveyName"],
        # TODO it flags the missing surveys in Deployments,
        # maybe ok even tho it technically isn't a foreign key
        "foreign_keys": {
            # "deployments" : "SurveyID",
        },
        "relationships": [],
    },
    "sites": {
        "file_name": S3_SHAREPOINT_SITE_CSV,
        "required": ["SiteID"],
        "unique": ["SiteID"],
        "info_columns": ["SiteName", "LinkToMarineReserve"],
        "foreign_keys": {},
        "relationships": [],
    },
    "species": {
        "file_name": S3_SHAREPOINT_SPECIES_CSV,
        "required": ["AphiaID", "CommonName", "ScientificName"],
        "unique": ["AphiaID", "CommonName", "ScientificName"],
        "info_columns": ["AphiaID", "CommonName", "ScientificName"],
        "foreign_keys": {},
        "relationships": [],
    },
    "reserves": {
        "file_name": S3_SHAREPOINT_RESERVES_CSV,
        "required": [],
        "unique": [],
        "info_columns": [],
        "foreign_keys": {},
        "relationships": [],
    },
}

VALIDATION_PATTERNS = {
    "DropID": r"^[A-Z]{3}_\d{8}_BUV_[A-Z]{3}_\d{3}_\d{2}$",
    "SurveyID": r"^[A-Z]{3}_\d{8}_BUV$",
    "SiteID": r"^[A-Z]{3}_\d+$",
}
