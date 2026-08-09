"""IG REST transport: authentication, quotes, positions, open, close.

This layer knows nothing about factors, signals or position sizing — it only
speaks HTTP to IG. Sizing and the decision to trade at all belong to
`risk_engine.py` (step A.2), which sits above it.

Three safety rules are hardcoded here on purpose:

  1. BASE_URL is pinned to the Demo host, so a config mistake cannot reach the
     live account.
  2. `open_position` refuses an order without a server-side stop distance.
  3. `open_position` refuses a size above MAX_SIZE.

Rules 2 and 3 duplicate checks that `risk_engine.py` will also make. That
duplication is intentional: risk_engine is the gate, these are the last line of
defence if anything ever calls the client directly.

There is deliberately NO history/candles method. IG's price endpoint burns a
weekly quota (~10k data points, 1 candle = 1 point) and AlphaFX already sources
prices from yfinance/SQLite. A method that does not exist cannot burn quota.

Smoke-test against the real Demo account (logs in, reads a quote, lists
positions, places nothing):

    python -m alphafx.execution.ig_client
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..config import load_local_env

# Demo-only host. [DO NOT change to api.ig.com] — it is not in config.py
# precisely so it never looks like a tunable parameter.
BASE_URL = "https://demo-api.ig.com/gateway/deal"

# AUD/USD mini contract: 1 lot = 10,000 AUD, roughly 1 USD per pip.
DEFAULT_EPIC = "CS.D.AUDUSD.MINI.IP"

# Hard per-order size cap. [DO NOT raise] — this is a fat-finger backstop, not a
# tunable. risk_engine.py enforces the same limit one layer up.
MAX_SIZE = 1.0

# Each IG endpoint pins its own Version header; they are NOT interchangeable.
# A wrong version returns 404 with an HTML error page instead of JSON.
_V_SESSION = 3
_V_REFRESH = 1
_V_MARKETS = 3
_V_ACCOUNTS = 1
_V_POSITIONS = 2
_V_OPEN = 2
_V_CONFIRMS = 1
_V_CLOSE = 1

# Plain-language translations of IG's error codes, shown at the moment a call
# fails. The raw code alone (e.g. "error.security.invalid-details") points at the
# wrong cause often enough to be worth a table.
ERROR_HINTS = {
    "validation.pattern.invalid.authenticationRequest.identifier":
        "IG_USERNAME is malformed: it takes the username, not an email address.",
    "error.security.invalid-details":
        "Wrong username or password. Use the demo Web API credentials, "
        "not the ones for the live account.",
    "error.security.account-migrated":
        "This account was migrated to IG's new platform and cannot use v2 CST "
        "auth. login() uses v3 OAuth, so seeing this means something else still "
        "sends a v2 request.",
    "error.security.api-key-invalid":
        "Invalid API key. Check it was generated under the Demo tab in My IG "
        "(a live-account key will not work).",
    "error.security.api-key-missing":
        "The request carried no API key. Check IG_API_KEY in .env.",
    "error.public-api.exceeded-account-allowance":
        "Endpoint quota exceeded. Wait for the weekly allowance to reset.",
    "error.security.account-token-invalid":
        "The session expired. Call login() again.",
    "error.confirms.deal-not-found":
        "No fill confirmation for this deal reference yet. Retry shortly.",
}


class IGError(RuntimeError):
    """Any failure talking to IG: bad credentials, rejected request, no session.

    A real exception rather than SystemExit so callers (bridge, tests, the
    dashboard) can catch it; `scripts/execute_demo.py` turns it into a clean
    message at the CLI boundary.
    """


def check(response: Any) -> Any:
    """Return the response, or raise IGError with IG's error code translated.

    `raise_for_status()` would throw away the `errorCode` in the body and leave
    only "HTTP 400", which is not enough to debug against IG.
    """
    if response.ok:
        return response
    try:
        code = response.json().get("errorCode", "(no errorCode in response body)")
    except ValueError:
        # Not JSON. Almost always a wrong URL or Version header — IG's gateway
        # then returns a Tomcat HTML error page.
        body = (response.text or "").strip()
        if body[:50].lower().startswith(("<!doctype", "<html")):
            code = "(IG returned an HTML error page, not JSON — usually a wrong URL path or Version header)"
        else:
            code = body[:200] or "(empty response body)"
    request = getattr(response, "request", None)
    where = f"{getattr(request, 'method', '?')} {getattr(request, 'url', '?')}"
    hint = ERROR_HINTS.get(code, "")
    raise IGError(
        f"IG rejected the request (HTTP {response.status_code})\n"
        f"  endpoint: {where}\n"
        f"  errorCode: {code}" + (f"\n  likely cause: {hint}" if hint else "")
    )


def _need(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value or value.startswith("your-"):
        raise IGError(
            f"Missing {key}. Fill these three in the .env at the repo root:\n"
            f"  IG_USERNAME / IG_PASSWORD / IG_API_KEY\n"
            f"(.env is gitignored, so it is never committed — and never paste it into a chat.)"
        )
    return value


class IGClient:
    """Thin REST client for one IG Demo account.

    Credentials are read in `__init__`, never at module level, so `import
    ig_client` works with no `.env` present — offline tests and backtests are
    not blocked by missing credentials, they just cannot construct a client.
    """

    def __init__(self, session: Any | None = None) -> None:
        load_local_env()
        api_key = _need("IG_API_KEY")
        self._username = _need("IG_USERNAME")
        self._password = _need("IG_PASSWORD")

        # Injectable so tests can drive the client with a fake transport.
        self.s = session if session is not None else requests.Session()
        self.s.headers.update({
            "X-IG-API-KEY": api_key,
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
        })
        self.account_id: str | None = None
        self._refresh_token: str | None = None
        self._expires_at = 0.0

    # ---- session ----

    def login(self) -> dict:
        """POST /session (Version 3) -> OAuth token.

        v3, not v2: v2's CST + X-SECURITY-TOKEN scheme is rejected outright for
        accounts migrated to IG's new platform (`error.security.account-migrated`),
        regardless of whether the credentials are correct. The cost of v3 is that
        `access_token` lives only ~60s, hence `_ensure_token`.
        """
        response = check(self.s.post(
            f"{BASE_URL}/session",
            json={"identifier": self._username, "password": self._password},
            headers={"Version": str(_V_SESSION)},
        ))
        info = response.json()
        self.account_id = info["accountId"]
        self._apply_token(info["oauthToken"])
        return info

    def _apply_token(self, token: dict) -> None:
        self._refresh_token = token["refresh_token"]
        # Renew 10s early rather than racing the expiry instant.
        self._expires_at = time.time() + int(token["expires_in"]) - 10
        self.s.headers.update({
            "Authorization": f"{token['token_type']} {token['access_token']}",
            "IG-ACCOUNT-ID": self.account_id,
        })

    def _ensure_token(self) -> None:
        """Refresh the 60-second access token before it expires."""
        if not self._refresh_token or time.time() < self._expires_at:
            return
        response = check(self.s.post(
            f"{BASE_URL}/session/refresh-token",
            json={"refresh_token": self._refresh_token},
            headers={"Version": str(_V_REFRESH)},
        ))
        self._apply_token(response.json())

    def _request(self, method: str, path: str, version: int,
                 extra_headers: dict | None = None, **kwargs: Any) -> Any:
        """Single entry point for business calls: renew token, pin Version, check errors."""
        if self._refresh_token is None:
            raise IGError("Not logged in: call login() first")
        self._ensure_token()
        headers = {"Version": str(version)}
        if extra_headers:
            headers.update(extra_headers)
        return check(self.s.request(method, f"{BASE_URL}{path}", headers=headers, **kwargs))

    # ---- read ----

    def get_market(self, epic: str = DEFAULT_EPIC) -> dict:
        """Current bid/offer and market status (non-TRADEABLE at weekends)."""
        return self._request("GET", f"/markets/{epic}", version=_V_MARKETS).json()["snapshot"]

    def get_account(self) -> dict:
        """Balance/equity for the logged-in account — the input to risk_engine's
        1%-per-trade sizing and the drawdown circuit breaker (step A.2).

        Returns IG's account dict; `balance` holds {balance, deposit, profitLoss,
        available}. Verified against the Demo account 2026-08-09.
        """
        accounts = self._request("GET", "/accounts", version=_V_ACCOUNTS).json()["accounts"]
        for account in accounts:
            if account.get("accountId") == self.account_id:
                return account
        raise IGError(f"Account {self.account_id} from login is absent from /accounts")

    def get_positions(self) -> list[dict]:
        return self._request("GET", "/positions", version=_V_POSITIONS).json()["positions"]

    # ---- write ----

    def open_position(self, direction: str, size: float, stop_distance: float,
                      limit_distance: float | None = None,
                      epic: str = DEFAULT_EPIC) -> dict:
        """Market order with a mandatory server-side stop; returns the confirmation.

        direction:     "BUY" or "SELL"
        size:          lots (mini contract, 1 lot = 10,000 AUD)
        stop_distance: stop in points — required, no exceptions
        limit_distance: take-profit in points

        The three checks below duplicate risk_engine's. [DO NOT delete] — they
        are the last line of defence if the gate is ever bypassed.
        """
        if not stop_distance or stop_distance <= 0:
            raise IGError("Order refused: stop_distance is mandatory")
        if size is None or size <= 0:
            raise IGError(f"Order refused: size must be positive (got {size})")
        if size > MAX_SIZE:
            raise IGError(f"Order refused: size {size} exceeds the {MAX_SIZE} lot cap")

        body = {
            "epic": epic,
            "expiry": "-",
            "direction": direction,
            "size": size,
            "orderType": "MARKET",
            "guaranteedStop": False,
            "forceOpen": True,
            "stopDistance": stop_distance,
            "currencyCode": "USD",
        }
        if limit_distance:
            body["limitDistance"] = limit_distance

        response = self._request("POST", "/positions/otc", version=_V_OPEN, json=body)
        return self.confirm(response.json()["dealReference"])

    def confirm(self, deal_reference: str) -> dict:
        """Resolve a dealReference into an actual fill.

        Opening is asynchronous: POST /positions/otc only acknowledges receipt,
        so `dealStatus` here may still be REJECTED. Never treat a dealReference
        as a filled position.
        """
        return self._request("GET", f"/confirms/{deal_reference}", version=_V_CONFIRMS).json()

    def close_position(self, deal_id: str, direction: str, size: float) -> dict:
        """Close a position. `direction` is the OPPOSITE of the held side."""
        body = {"dealId": deal_id, "direction": direction, "size": size, "orderType": "MARKET"}
        # IG models closing as DELETE but requires it sent as POST with a _method header.
        response = self._request("POST", "/positions/otc", version=_V_CLOSE,
                                 extra_headers={"_method": "DELETE"}, json=body)
        return self.confirm(response.json()["dealReference"])


if __name__ == "__main__":
    # Connectivity check: log in, read a quote, list positions. Places nothing.
    client = IGClient()
    client.login()
    print(f"logged in to demo account: {client.account_id}")
    snapshot = client.get_market()
    print(f"{DEFAULT_EPIC}  bid={snapshot['bid']}  offer={snapshot['offer']}  "
          f"status={snapshot['marketStatus']}")
    for position in client.get_positions():
        held, market = position["position"], position["market"]
        print(f"{market['epic']}  {held['direction']}  size={held['size']}  "
              f"open={held['level']}  stop={held.get('stopLevel')}")
