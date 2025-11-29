import os
from dotenv import load_dotenv
from minio import Minio
from minio.notificationconfig import NotificationConfig, QueueConfig

# Load environment variables from .env file
load_dotenv()

# Initialize
minio_client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=os.getenv("MINIO_SECURE", "False").lower() == "true",
)

bucket_name = "devis"