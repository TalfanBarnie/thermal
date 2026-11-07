import logging

from thermal.redact import RedactingFilter, redact


def test_redact_patterns():
    s = (
        "403 for https://api.eumetsat.int/x?access_token=55885a7e-bc5c-3735-ba7a-d857119886ae "
        "Current token 558482fd-f70c-34e7-aa2e-a974dc5f9b2f expires; Authorization: Bearer abc.def-1"
    )
    r = redact(s)
    assert "55885a7e" not in r and "558482fd" not in r and "abc.def-1" not in r
    assert "access_token=<redacted>" in r and "token <redacted> expires" in r


def test_logging_filter(caplog):
    logger = logging.getLogger("thermal.test")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("retry %s", "https://h/?access_token=abc123")
    assert "abc123" not in caplog.text and "<redacted>" in caplog.text
