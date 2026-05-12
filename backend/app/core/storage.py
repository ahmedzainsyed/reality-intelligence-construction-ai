"""MinIO / S3 storage client."""
from app.core.config import settings
import structlog

logger = structlog.get_logger(__name__)

class StorageClient:
    def __init__(self):
        from minio import Minio
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

    async def upload_bytes(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
        import io
        self.client.put_object(bucket, key, io.BytesIO(data), len(data), content_type=content_type)

    def get_presigned_url(self, bucket: str, key: str, expires_hours: int = 24) -> str:
        from datetime import timedelta
        return self.client.presigned_get_object(bucket, key, expires=timedelta(hours=expires_hours))

_storage_client = None

def get_storage_client() -> StorageClient:
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient()
    return _storage_client

async def check_storage_health() -> bool:
    try:
        client = get_storage_client()
        client.client.list_buckets()
        return True
    except Exception:
        return False
