from datetime import UTC, datetime

from thermal import storage
from thermal.config import BucketCredentials


def test_key_layout():
    t = datetime(2024, 3, 16, 22, 0, tzinfo=UTC)
    assert storage.raw_key("msg-0deg", t, "l15") == "raw/msg-0deg/2024/03/16/20240316T220000_l15.nc"
    assert storage.fetch_report_key("msg-0deg", t) == "reports/fetch/msg-0deg/20240316T220000.json"


class FakeClient:
    def __init__(self):
        self.objects = {}

        class _Exc:
            class NoSuchKey(Exception):
                pass

        self.exceptions = _Exc

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        import io

        return {"Body": io.BytesIO(self.objects[Key])}


def test_bucket_json_roundtrip():
    b = storage.Bucket(BucketCredentials("b", "k", "a"), "endpoint", client=FakeClient())
    b.put_json("state/x.json", {"last": "2024"})
    assert b.get_json("state/x.json") == {"last": "2024"}
    assert b.get_json("missing.json", default={}) == {}
