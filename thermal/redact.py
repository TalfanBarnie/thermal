"""Scrub credentials from log output and reports.

Workflow logs on a public repository are public. GitHub masks repository secrets, but
not values derived from them: the EUMETSAT access token appears in Data Tailor error
text (download URLs carry it as a query parameter) and in the client's retry warnings.
"""

from __future__ import annotations

import logging
import re

_PATTERNS = [
    (re.compile(r"(access_token=)[A-Za-z0-9._\-]+"), r"\1<redacted>"),
    (
        re.compile(r"(\btoken )[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
        r"\1<redacted>",
    ),
    (re.compile(r"(Bearer )[A-Za-z0-9._\-]+"), r"\1<redacted>"),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    """Rewrites every record's message (and exception text) before it is emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:
            pass
        return True


def install() -> None:
    root = logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for f in root.filters):
        root.addFilter(RedactingFilter())
    # handlers attached to the root see the filtered record; library loggers propagate to root
    for h in root.handlers:
        if not any(isinstance(f, RedactingFilter) for f in h.filters):
            h.addFilter(RedactingFilter())
