"""Default socket timeouts for every HTTP request made by this process.

The EUMETSAT client issues requests without a timeout, so a connection the server drops
silently can block a worker thread indefinitely. Installing a default timeout on
requests.Session.request bounds every call (the client's own retries still apply).
"""

from __future__ import annotations

import requests

_installed = False


def install_default_timeout(connect: float = 30.0, read: float = 300.0) -> None:
    global _installed
    if _installed:
        return
    original = requests.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (connect, read)
        return original(self, method, url, **kwargs)

    requests.Session.request = request_with_timeout  # type: ignore[method-assign]
    _installed = True
