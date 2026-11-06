"""Backblaze B2 storage through the S3-compatible API.

Two buckets: private (raw EUMETSAT cutouts, arrays, models, state) and public
(exactly what the web app reads). Key layouts are defined here so that every
module agrees on them.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import BucketCredentials, Settings


def time_stamp(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M%S")


def raw_key(sat: str, t: datetime, kind: str) -> str:
    """Private bucket key of a raw Data Tailor cutout. kind is 'l15' or 'clm'."""
    return f"raw/{sat}/{t:%Y/%m/%d}/{time_stamp(t)}_{kind}.nc"


def fetch_report_key(sat: str, t: datetime) -> str:
    return f"reports/fetch/{sat}/{time_stamp(t)}.json"


def state_key(sat: str) -> str:
    return f"state/{sat}.json"


class Bucket:
    """Thin wrapper over a boto3 client bound to one bucket."""

    def __init__(self, creds: BucketCredentials, endpoint: str, client: Any | None = None):
        self.name = creds.name
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=f"https://{endpoint}",
                aws_access_key_id=creds.key_id,
                aws_secret_access_key=creds.application_key,
            )
        self._client = client

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        self._client.put_object(Bucket=self.name, Key=key, Body=data, ContentType=content_type)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        with open(path, "rb") as fh:
            self._client.upload_fileobj(fh, self.name, key, ExtraArgs={"ContentType": content_type})

    def put_json(self, key: str, obj: Any):
        body = json.dumps(obj, indent=1, sort_keys=True, default=str).encode()
        self.put_bytes(key, body, content_type="application/json")

    def get_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.name, Key=key)
        return resp["Body"].read()

    def get_file(self, key: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            self._client.download_fileobj(self.name, key, fh)

    def get_json(self, key: str, default: Any = None) -> Any:
        try:
            return json.loads(self.get_bytes(key))
        except self._client.exceptions.NoSuchKey:
            return default

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.name, Key=key)
            return True
        except Exception as exc:  # botocore ClientError with 404
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list_keys(self, prefix: str, max_keys: int | None = None) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.name, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
                if max_keys and len(keys) >= max_keys:
                    return keys
        return keys

    def open_read(self, key: str) -> io.BytesIO:
        return io.BytesIO(self.get_bytes(key))


class Storage:
    def __init__(self, settings: Settings):
        self.private = Bucket(settings.b2_private, settings.b2_endpoint)
        self.public = Bucket(settings.b2_public, settings.b2_endpoint)
        self.public_url = settings.b2_public_url

    def public_url_for(self, key: str) -> str:
        return f"{self.public_url}/{key}"
