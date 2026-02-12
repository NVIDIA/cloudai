# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple

import pytest
import requests
from requests import Response

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


@pytest.fixture
def dummy_client(monkeypatch: pytest.MonkeyPatch, dummy_token_response: DummyResponse) -> RunAIRestClient:
    def fake_post(self: requests.Session, url: str, json: Dict[str, Any]) -> DummyResponse:
        return dummy_token_response

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return RunAIRestClient("http://dummy", "app_id", "app_secret")


def test_init_access_token(dummy_client: RunAIRestClient) -> None:
    assert dummy_client.access_token == "dummy_token"
    assert "Bearer dummy_token" in str(dummy_client.session.headers.get("Authorization", ""))


def test_get_access_token_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: requests.Session, url: str, json: Dict[str, Any]) -> DummyResponse:
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
    ) -> DummyResponse:
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
    ) -> DummyResponse:
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
