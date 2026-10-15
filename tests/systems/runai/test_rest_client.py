# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ssl
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock

import pytest
import requests
from requests import Response

from cloudai.systems.runai import RunAISystem
from cloudai.systems.runai.runai_rest_client import RunAIRestClient


class DummyResponse(Response):
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200) -> None:
        super().__init__()
        self._json: Dict[str, Any] = json_data
        self.status_code: int = status_code
        self._content = b"non-empty" if json_data else b""

    def json(
        self,
        *,
        cls: Optional[Any] = None,
        object_hook: Optional[Any] = None,
        parse_float: Optional[Any] = None,
        parse_int: Optional[Any] = None,
        parse_constant: Optional[Any] = None,
        object_pairs_hook: Optional[Any] = None,
        **kwds: Any,
    ) -> Dict[str, Any]:
        _ = (cls, object_hook, parse_float, parse_int, parse_constant, object_pairs_hook, kwds)
        return self._json


@pytest.fixture
def dummy_token_response() -> DummyResponse:
    return DummyResponse({"accessToken": "dummy_token"})


@pytest.fixture(params=[True, False])
def dummy_client(
    monkeypatch: pytest.MonkeyPatch, dummy_token_response: DummyResponse, request: pytest.FixtureRequest
) -> RunAIRestClient:
    def fake_post(
        self: requests.Session, url: str, json: Dict[str, Any], timeout: float, verify: bool
    ) -> DummyResponse:
        assert timeout == 30
        assert verify is request.param
        return dummy_token_response

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return RunAIRestClient("http://dummy", "app_id", "app_secret", verify_certs=request.param)


def test_init_access_token(dummy_client: RunAIRestClient) -> None:
    assert dummy_client.access_token == "dummy_token"
    assert "Bearer dummy_token" in str(dummy_client.session.headers.get("Authorization", ""))


def test_get_access_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        self: requests.Session, url: str, json: Dict[str, Any], timeout: float, verify: bool
    ) -> DummyResponse:
        assert timeout == 30
        assert verify is True
        return DummyResponse({}, 200)

    monkeypatch.setattr(requests.Session, "post", fake_post)
    with pytest.raises(ValueError):
        RunAIRestClient("http://dummy", "app_id", "app_secret")


def test_request_success(dummy_client: RunAIRestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(
        method: str,
        url: str,
        headers: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: float = 0,
        verify: bool = True,
    ) -> DummyResponse:
        assert timeout == 30
        assert verify is dummy_client.verify_certs
        return DummyResponse({"result": "ok"}, 200)

    monkeypatch.setattr(requests, "request", fake_request)
    result = dummy_client._request("GET", "/test", params={"a": 1})
    assert result == {"result": "ok"}


def test_request_http_error(dummy_client: RunAIRestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(
        method: str,
        url: str,
        headers: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: float = 0,
        verify: bool = True,
    ) -> DummyResponse:
        assert timeout == 30
        assert verify is dummy_client.verify_certs
        return DummyResponse({"error": "fail"}, 400)

    monkeypatch.setattr(requests, "request", fake_request)
    with pytest.raises(requests.exceptions.HTTPError):
        dummy_client._request("GET", "/error")


@pytest.mark.parametrize(
    "method_name, args, kwargs, expected_method, expected_path",
    [
        ("get_clusters", (), {"params": {"key": "value"}}, "GET", "/api/v1/clusters"),
        ("create_cluster", ({"name": "test"},), {}, "POST", "/api/v1/clusters"),
    ],
)
def test_endpoints(
    dummy_client: RunAIRestClient,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    expected_method: str,
    expected_path: str,
) -> None:
    def fake_request(
        method: str, path: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {"method": method, "path": path, "params": params, "data": data}

    monkeypatch.setattr(dummy_client, "_request", fake_request)
    result = getattr(dummy_client, method_name)(*args, **kwargs)
    assert result["method"] == expected_method
    assert result["path"] == expected_path
    if "params" in kwargs:
        assert result["params"] == kwargs["params"]
    else:
        assert result["params"] is None
    if args and expected_method in {"POST", "PUT", "PATCH"}:
        assert result["data"] == args[0]
    else:
        assert result["data"] is None


def test_fetch_training_logs_respects_tls_config(
    dummy_client: RunAIRestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status = MagicMock(return_value=DummyResponse({"status": "OK"}))
    status.return_value._content = b"OK"
    monkeypatch.setattr(requests, "get", status)
    connect = MagicMock()
    connect.return_value.__enter__.return_value = iter([b"first\n", "second\n"])
    monkeypatch.setattr("cloudai.systems.runai.runai_rest_client.ws_connect", connect)
    output = tmp_path / "training.log"

    dummy_client.fetch_training_logs("https://cluster.example.com", "project", "training", output)

    assert status.call_args.kwargs["timeout"] == 30
    assert status.call_args.kwargs["verify"] is dummy_client.verify_certs
    assert connect.call_args.args[0].startswith("wss://cluster.example.com/")
    context = connect.call_args.kwargs["ssl"]
    assert context.verify_mode == (ssl.CERT_REQUIRED if dummy_client.verify_certs else ssl.CERT_NONE)
    assert context.check_hostname is dummy_client.verify_certs
    assert output.read_text() == "first\nsecond\n"


@pytest.mark.parametrize("verify_certs", [True, False])
def test_system_passes_tls_config(
    runai_system: RunAISystem, monkeypatch: pytest.MonkeyPatch, verify_certs: bool
) -> None:
    assert runai_system.verify_certs is True
    runai_system.verify_certs = verify_certs
    client = MagicMock()
    monkeypatch.setattr("cloudai.systems.runai.runai_system.RunAIRestClient", client)

    assert runai_system.api_client is client.return_value
    assert client.call_args.kwargs["verify_certs"] is verify_certs
