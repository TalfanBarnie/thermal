"""Configuration: YAML files under config/ and credentials from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def config_dir() -> Path:
    return Path(os.environ.get("THERMAL_CONFIG_DIR", _DEFAULT_CONFIG_DIR))


@cache
def load_yaml(name: str) -> dict[str, Any]:
    path = config_dir() / f"{name}.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def satellite(key: str) -> dict[str, Any]:
    sats = load_yaml("satellites")
    if key not in sats:
        raise KeyError(f"unknown satellite key {key!r}; known: {sorted(sats)}")
    cfg = dict(sats[key])
    cfg["key"] = key
    return cfg


def iceland_box() -> dict[str, float]:
    return dict(load_yaml("regions")["iceland_box"])


def monitoring_regions() -> dict[str, Any]:
    return dict(load_yaml("regions").get("monitoring", {}))


def pipeline() -> dict[str, Any]:
    return load_yaml("pipeline")


@dataclass(frozen=True)
class BucketCredentials:
    name: str
    key_id: str
    application_key: str


@dataclass(frozen=True)
class Settings:
    eumetsat_key: str
    eumetsat_secret: str
    b2_endpoint: str
    b2_private: BucketCredentials
    b2_public: BucketCredentials
    b2_public_url: str


class MissingSetting(RuntimeError):
    pass


def _env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "")
    if required and not value:
        raise MissingSetting(f"environment variable {name} is not set")
    return value


def load_settings() -> Settings:
    """Read credentials and bucket settings from the environment.

    Names match the repository secrets and variables described in docs/SPEC.md.
    """
    return Settings(
        eumetsat_key=_env("EUMETSAT_CONSUMER_KEY"),
        eumetsat_secret=_env("EUMETSAT_CONSUMER_SECRET"),
        b2_endpoint=_env("B2_ENDPOINT"),
        b2_private=BucketCredentials(
            name=_env("B2_PRIVATE_BUCKET"),
            key_id=_env("B2_PRIVATE_KEY_ID"),
            application_key=_env("B2_PRIVATE_APPLICATION_KEY"),
        ),
        b2_public=BucketCredentials(
            name=_env("B2_PUBLIC_BUCKET"),
            key_id=_env("B2_PUBLIC_KEY_ID"),
            application_key=_env("B2_PUBLIC_APPLICATION_KEY"),
        ),
        b2_public_url=_env("B2_PUBLIC_URL", required=False).rstrip("/"),
    )
