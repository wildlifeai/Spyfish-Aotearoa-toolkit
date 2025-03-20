import boto3
import logging
import typing
from sftk.common import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
from sftk import log_config

class S3Handler(object):
    """
    Singleton class for interacting with an S3 bucket.
    """
    _instance = None

    def __new__(cls, *args, **kwargs) -> "S3Handler":
        """
        Create a new instance of the class if one does not already exist.

        Returns:
            S3Handler: The instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.s3 = boto3.client("s3",
                                            aws_access_key_id=AWS_ACCESS_KEY_ID,
                                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            logging.info("Created a new instance of the S3Handler class.")

        return cls._instance

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns:
            str: The string representation of the class.
        """
        return f"S3Handler({self.s3})"
