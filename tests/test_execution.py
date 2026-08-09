"""Execution-layer tests: fully offline, no credentials, no IG quota consumed.

The client is driven through an injected fake session, so every request IG would
receive is inspectable. Nothing here touches the network or the developer's .env.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphafx.execution import ig_client
from alphafx.execution.ig_client import BASE_URL, IGClient, IGError, check


# ---- fake transport ----

class FakeResponse:
    def __init__(self, payload=None, status=200, text=None, method="GET", url=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})
        self.request = SimpleNamespace(method=method, url=url)

    def json(self):
        if self._payload is None:
            raise ValueError("response body is not JSON")
        return self._payload


class Call(SimpleNamespace):
    pass


class FakeSession:
    """Routes (METHOD, path) -> payload dict, FakeResponse, or a list consumed in order."""

    def __init__(self, routes: dict):
        self.headers: dict = {}
        self.routes = routes
        self.calls: list[Call] = []

    def post(self, url, json=None, headers=None):
        return self._dispatch("POST", url, headers, json)

    def request(self, method, url, headers=None, **kwargs):
        return self._dispatch(method, url, headers, kwargs.get("json"))

    def _dispatch(self, method, url, headers, body):
        path = url.replace(BASE_URL, "")
        merged = {**self.headers, **(headers or {})}
        self.calls.append(Call(method=method, path=path, headers=merged, body=body))
        key = (method, path)
        if key not in self.routes:
            raise AssertionError(f"unexpected request: {method} {path}")
        value = self.routes[key]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, FakeResponse):
            value.request = SimpleNamespace(method=method, url=url)
            return value
        return FakeResponse(value, method=method, url=url)

    def find(self, method, path):
        return [c for c in self.calls if c.method == method and c.path == path]


class FakeClock:
    def __init__(self, now=1_000.0):
        self.now = now

    def time(self):
        return self.now


def _token(access="a1", refresh="r1", expires=60):
    return {"access_token": access, "refresh_token": refresh,
            "token_type": "Bearer", "expires_in": str(expires)}


LOGIN_OK = {"accountId": "TESTACC1", "oauthToken": _token()}


@pytest.fixture(autouse=True)
def offline_credentials(monkeypatch):
    """Never read the real .env, and hand every test the same fake credentials.

    Without stubbing load_local_env the suite would pick up the developer's real
    IG credentials, which would make the "missing credentials" test pass or fail
    depending on whose machine it runs on.
    """
    monkeypatch.setattr(ig_client, "load_local_env", lambda *a, **k: None)
    for key in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD"):
        monkeypatch.setenv(key, f"fake-{key.lower()}")


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(ig_client, "time", fake)
    return fake


def _logged_in(routes, clock=None):
    session = FakeSession({("POST", "/session"): LOGIN_OK, **routes})
    client = IGClient(session=session)
    client.login()
    return client, session


# ---- hardcoded safety boundaries ----

def test_base_url_is_pinned_to_demo():
    # The single most important line in the package: a config slip must not be
    # able to point the execution layer at the live account.
    assert BASE_URL == "https://demo-api.ig.com/gateway/deal"
    assert "demo-api.ig.com" in BASE_URL and "//api.ig.com" not in BASE_URL


def test_client_exposes_no_history_endpoint():
    # IG's price endpoint burns a weekly quota and AlphaFX gets prices from
    # yfinance/SQLite. Guard against someone re-adding it out of habit.
    assert not any(name in dir(IGClient) for name in ("get_history", "get_candles", "get_prices"))


def test_missing_credentials_raise_a_helpful_error(monkeypatch):
    monkeypatch.delenv("IG_API_KEY", raising=False)
    with pytest.raises(IGError, match="IG_API_KEY"):
        IGClient(session=FakeSession({}))


def test_import_does_not_require_credentials():
    # Credentials are read in __init__, never at module level, so importing works
    # with no .env at all and offline tests/backtests are never blocked by them.
    # Checked in a subprocess: importlib.reload() would rebind IGError and break
    # every later `pytest.raises(IGError)` in this module.
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("IG_")}
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, "-c", "import alphafx.execution.ig_client"],
        capture_output=True, text=True, env=env, cwd=root,
    )
    assert result.returncode == 0, result.stderr


# ---- session: v3 OAuth ----

def test_login_uses_version_3_and_sets_bearer_headers():
    client, session = _logged_in({})
    login = session.find("POST", "/session")[0]

    assert login.headers["Version"] == "3"          # v2 CST auth is rejected outright
    assert login.body == {"identifier": "fake-ig_username", "password": "fake-ig_password"}
    assert client.account_id == "TESTACC1"
    assert session.headers["Authorization"] == "Bearer a1"
    assert session.headers["IG-ACCOUNT-ID"] == "TESTACC1"
    assert session.headers["X-IG-API-KEY"] == "fake-ig_api_key"


def test_request_before_login_is_refused():
    client = IGClient(session=FakeSession({}))
    with pytest.raises(IGError, match="Not logged in"):
        client.get_positions()


def test_access_token_refreshes_once_expired(clock):
    snapshot = {"snapshot": {"bid": 0.65, "offer": 0.6502, "marketStatus": "TRADEABLE"}}
    session = FakeSession({
        ("POST", "/session"): LOGIN_OK,
        ("POST", "/session/refresh-token"): _token(access="a2", refresh="r2"),
        ("GET", "/markets/CS.D.AUDUSD.MINI.IP"): snapshot,
    })
    client = IGClient(session=session)
    client.login()

    # Still inside the 60s window (minus the 10s safety margin): no refresh.
    clock.now += 30
    client.get_market()
    assert session.find("POST", "/session/refresh-token") == []

    # Past the margin: the next call renews first, then proceeds.
    clock.now += 30
    client.get_market()
    refresh = session.find("POST", "/session/refresh-token")
    assert len(refresh) == 1
    assert refresh[0].headers["Version"] == "1"
    assert refresh[0].body == {"refresh_token": "r1"}
    assert session.headers["Authorization"] == "Bearer a2"


# ---- reads ----

def test_get_market_uses_version_3():
    snapshot = {"snapshot": {"bid": 0.65, "offer": 0.6502, "marketStatus": "TRADEABLE"}}
    client, session = _logged_in({("GET", "/markets/CS.D.AUDUSD.MINI.IP"): snapshot})

    assert client.get_market()["marketStatus"] == "TRADEABLE"
    assert session.find("GET", "/markets/CS.D.AUDUSD.MINI.IP")[0].headers["Version"] == "3"


def test_get_positions_uses_version_2():
    client, session = _logged_in({("GET", "/positions"): {"positions": [{"position": {}}]}})

    assert len(client.get_positions()) == 1
    assert session.find("GET", "/positions")[0].headers["Version"] == "2"


def test_get_account_picks_the_logged_in_account():
    accounts = {"accounts": [
        {"accountId": "OTHER1", "balance": {"balance": 1.0}},
        {"accountId": "TESTACC1", "balance": {"balance": 10_000.0}},
    ]}
    client, _ = _logged_in({("GET", "/accounts"): accounts})

    assert client.get_account()["balance"]["balance"] == 10_000.0


def test_get_account_errors_when_account_is_absent():
    client, _ = _logged_in({("GET", "/accounts"): {"accounts": [{"accountId": "OTHER1"}]}})
    with pytest.raises(IGError, match="TESTACC1"):
        client.get_account()


# ---- opening: the last line of defence ----

@pytest.mark.parametrize("stop", [None, 0, -5])
def test_open_position_refuses_without_a_stop(stop):
    client, session = _logged_in({})
    with pytest.raises(IGError, match="stop_distance is mandatory"):
        client.open_position("BUY", size=0.1, stop_distance=stop)
    assert session.find("POST", "/positions/otc") == []  # nothing reached IG


@pytest.mark.parametrize("size", [None, 0, -0.1])
def test_open_position_refuses_a_non_positive_size(size):
    client, session = _logged_in({})
    with pytest.raises(IGError, match="size must be positive"):
        client.open_position("BUY", size=size, stop_distance=30)
    assert session.find("POST", "/positions/otc") == []


def test_open_position_refuses_size_above_the_hard_cap():
    client, session = _logged_in({})
    with pytest.raises(IGError, match="exceeds the"):
        client.open_position("BUY", size=ig_client.MAX_SIZE + 0.1, stop_distance=30)
    assert session.find("POST", "/positions/otc") == []


def test_open_position_sends_a_stop_and_confirms_the_deal():
    client, session = _logged_in({
        ("POST", "/positions/otc"): {"dealReference": "REF123"},
        ("GET", "/confirms/REF123"): {"dealStatus": "ACCEPTED", "level": 0.6501, "reason": "SUCCESS"},
    })

    result = client.open_position("BUY", size=0.1, stop_distance=30, limit_distance=60)

    order = session.find("POST", "/positions/otc")[0]
    assert order.headers["Version"] == "2"
    assert order.body["stopDistance"] == 30          # server-side stop, always present
    assert order.body["limitDistance"] == 60
    assert order.body["orderType"] == "MARKET"
    assert order.body["forceOpen"] is True
    assert order.body["epic"] == "CS.D.AUDUSD.MINI.IP"
    assert session.find("GET", "/confirms/REF123")[0].headers["Version"] == "1"
    assert result["dealStatus"] == "ACCEPTED"


def test_open_position_surfaces_a_rejection():
    # A dealReference is only an acknowledgement. A rejected order must come back
    # as REJECTED, never be mistaken for a filled position.
    client, _ = _logged_in({
        ("POST", "/positions/otc"): {"dealReference": "REF999"},
        ("GET", "/confirms/REF999"): {"dealStatus": "REJECTED", "reason": "MARKET_CLOSED"},
    })

    result = client.open_position("BUY", size=0.1, stop_distance=30)
    assert result["dealStatus"] == "REJECTED" and result["reason"] == "MARKET_CLOSED"


def test_close_position_sends_the_delete_method_header():
    client, session = _logged_in({
        ("POST", "/positions/otc"): {"dealReference": "REFC1"},
        ("GET", "/confirms/REFC1"): {"dealStatus": "ACCEPTED"},
    })

    client.close_position("DEAL1", direction="SELL", size=0.1)

    close = session.find("POST", "/positions/otc")[0]
    assert close.headers["Version"] == "1"
    assert close.headers["_method"] == "DELETE"
    assert close.body == {"dealId": "DEAL1", "direction": "SELL", "size": 0.1, "orderType": "MARKET"}


# ---- error translation ----

def test_check_translates_a_known_error_code():
    response = FakeResponse({"errorCode": "error.security.account-migrated"}, status=403,
                            method="POST", url=f"{BASE_URL}/session")
    with pytest.raises(IGError) as excinfo:
        check(response)
    message = str(excinfo.value)
    assert "error.security.account-migrated" in message
    assert "migrated" in message               # the plain-language hint survives
    assert "POST" in message and "/session" in message


def test_check_flags_an_html_error_page():
    # The signature of a wrong URL or Version header.
    response = FakeResponse(None, status=404, text="<!DOCTYPE html><html>404</html>",
                            method="GET", url=f"{BASE_URL}/prices/X")
    with pytest.raises(IGError, match="Version"):
        check(response)


def test_check_passes_a_successful_response_through():
    response = FakeResponse({"ok": True})
    assert check(response) is response
