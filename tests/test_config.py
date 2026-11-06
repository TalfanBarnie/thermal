import pytest

from thermal import config


def test_satellites_load():
    sat = config.satellite("msg11-0deg")
    assert sat["collection"] == "EO:EUM:DAT:MSG:HRSEVIRI"
    assert sat["key"] == "msg11-0deg"
    assert "IR_039" in sat["channels"]


def test_unknown_satellite():
    with pytest.raises(KeyError):
        config.satellite("nope")


def test_iceland_box():
    box = config.iceland_box()
    assert box["north"] > box["south"]
    assert box["east"] > box["west"]


def test_settings_from_env(monkeypatch):
    env = {
        "EUMETSAT_CONSUMER_KEY": "k",
        "EUMETSAT_CONSUMER_SECRET": "s",
        "B2_ENDPOINT": "s3.eu-central-003.backblazeb2.com",
        "B2_PRIVATE_BUCKET": "priv",
        "B2_PRIVATE_KEY_ID": "pk",
        "B2_PRIVATE_APPLICATION_KEY": "pa",
        "B2_PUBLIC_BUCKET": "pub",
        "B2_PUBLIC_KEY_ID": "uk",
        "B2_PUBLIC_APPLICATION_KEY": "ua",
        "B2_PUBLIC_URL": "https://pub.s3.eu-central-003.backblazeb2.com/",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = config.load_settings()
    assert s.b2_private.name == "priv"
    assert s.b2_public.key_id == "uk"
    assert s.b2_public_url.endswith("backblazeb2.com")


def test_settings_missing(monkeypatch):
    monkeypatch.delenv("EUMETSAT_CONSUMER_KEY", raising=False)
    with pytest.raises(config.MissingSetting):
        config.load_settings()
