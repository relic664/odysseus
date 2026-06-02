"""Unit tests for Crawl4aiFetchProvider."""
import asyncio
from unittest.mock import MagicMock, patch


import pytest

from services.search.fetch_providers import Crawl4aiFetchProvider, FetchResult


# ── _extract_markdown ──────────────────────────────────────────────

def test_extract_markdown_fit_priority():
    data = {"fit_markdown": "# Fit", "raw_markdown": "# Raw", "markdown": "# Plain"}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "# Fit"


def test_extract_markdown_filtered_fallback():
    data = {"filtered_markdown": "# Filtered", "raw_markdown": "# Raw"}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "# Filtered"


def test_extract_markdown_plain_fallback():
    data = {"markdown": "# Plain"}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "# Plain"


def test_extract_markdown_raw_fallback():
    data = {"raw_markdown": "# Raw"}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "# Raw"


def test_extract_markdown_content_fallback():
    data = {"content": "body text"}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "body text"


def test_extract_markdown_nested_dict():
    data = {"markdown": {"fit_markdown": "# Nested fit"}}
    assert Crawl4aiFetchProvider._extract_markdown(data) == "# Nested fit"


def test_extract_markdown_empty_string_returns_empty():
    data = {"fit_markdown": "", "raw_markdown": "  "}
    assert Crawl4aiFetchProvider._extract_markdown(data) == ""


def test_extract_markdown_non_string_returns_empty():
    data = {"fit_markdown": 123, "markdown": None}
    assert Crawl4aiFetchProvider._extract_markdown(data) == ""


# ── _extract_title ─────────────────────────────────────────────────

def test_extract_title_direct():
    data = {"title": "My Page"}
    assert Crawl4aiFetchProvider._extract_title(data) == "My Page"


def test_extract_title_metadata_fallback():
    data = {"metadata": {"title": "From Metadata"}}
    assert Crawl4aiFetchProvider._extract_title(data) == "From Metadata"


def test_extract_title_direct_wins_over_metadata():
    data = {"title": "Direct", "metadata": {"title": "Meta"}}
    assert Crawl4aiFetchProvider._extract_title(data) == "Direct"


def test_extract_title_empty():
    assert Crawl4aiFetchProvider._extract_title({}) == ""
    assert Crawl4aiFetchProvider._extract_title({"title": ""}) == ""


# ── _decode_results ────────────────────────────────────────────────

def test_decode_results_top_level_list():
    data = [{"success": True}, {"success": False}]
    result = Crawl4aiFetchProvider()._decode_results(data)
    assert len(result) == 2


def test_decode_results_results_key():
    data = {"results": [{"success": True}], "other": "x"}
    result = Crawl4aiFetchProvider()._decode_results(data)
    assert len(result) == 1


def test_decode_results_data_key():
    data = {"data": [{"success": True}]}
    result = Crawl4aiFetchProvider()._decode_results(data)
    assert len(result) == 1


def test_decode_results_single_dict():
    data = {"success": True, "fit_markdown": "# Hi"}
    result = Crawl4aiFetchProvider()._decode_results(data)
    assert len(result) == 1
    assert result[0] is data


def test_decode_results_filters_non_dicts():
    data = [{"success": True}, "string", 42]
    result = Crawl4aiFetchProvider()._decode_results(data)
    assert len(result) == 1


def test_decode_results_unknown_type():
    assert Crawl4aiFetchProvider()._decode_results("hello") == []
    assert Crawl4aiFetchProvider()._decode_results(42) == []


# ── create_from_settings ───────────────────────────────────────────

def test_create_from_settings_default_url():
    prov = Crawl4aiFetchProvider.create_from_settings({})
    assert prov._base_url == "http://localhost:11235"


def test_create_from_settings_custom_url():
    prov = Crawl4aiFetchProvider.create_from_settings({"crawl4ai_url": "http://myhost:9999/"})
    assert prov._base_url == "http://myhost:9999"


def test_create_from_settings_strips_trailing_slash():
    prov = Crawl4aiFetchProvider.create_from_settings({"crawl4ai_url": "http://x.y/"})
    assert prov._base_url == "http://x.y"


def test_create_from_settings_whitespace_url():
    prov = Crawl4aiFetchProvider.create_from_settings({"crawl4ai_url": "  "})
    assert prov._base_url == "http://localhost:11235"


# ── name property ──────────────────────────────────────────────────

def test_provider_name():
    assert Crawl4aiFetchProvider().name == "crawl4ai"


class _FakeAsyncClient:
    """Minimal async-context-manager mock for httpx.AsyncClient."""

    def __init__(self, resp_or_error):
        self._resp_or_error = resp_or_error
        self.post_calls = []

    async def post(self, url, **kw):
        self.post_calls.append((url, kw))
        if isinstance(self._resp_or_error, Exception):
            raise self._resp_or_error
        return self._resp_or_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_fake_client(resp_or_error):
    return _FakeAsyncClient(resp_or_error)


# ── fetch: success path ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_success_fit_markdown():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "fit_markdown": "# Hello World",
        "title": "Test Page",
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com", timeout=10)

    assert result.success is True
    assert result.content == "# Hello World"
    assert result.title == "Test Page"
    assert result.url == "https://example.com"
    assert len(fake_client.post_calls) == 1
    assert fake_client.post_calls[0] == (
        "http://test.local/md",
        {"json": {"url": "https://example.com", "f": "fit"}},
    )


# ── fetch: http error ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_http_500():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 500
    fake_resp.text = "Internal Server Error"

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "HTTP 500" in result.error


@pytest.mark.asyncio
async def test_fetch_http_403():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 403
    fake_resp.text = "Forbidden"

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "HTTP 403" in result.error


# ── fetch: success=false in body ───────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_success_false_top_level():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": False,
        "error_message": "Blocked by anti-bot protection: DataDome captcha",
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "DataDome" in result.error


@pytest.mark.asyncio
async def test_fetch_success_false_nested_result():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "results": [
            {"success": False, "error_message": "Page timed out"}
        ]
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "Page timed out" in result.error


# ── fetch: connection error ────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_connect_error():
    import httpx
    prov = Crawl4aiFetchProvider(base_url="http://unreachable.local")

    fake_client = _make_fake_client(httpx.ConnectError("Connection refused"))

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "Cannot connect" in result.error
    assert "unreachable.local" in result.error


# ── fetch: timeout ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_timeout():
    import httpx
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_client = _make_fake_client(httpx.TimeoutException("Timed out"))

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com", timeout=30)

    assert result.success is False
    assert "timed out" in result.error
    assert "30s" in result.error


# ── fetch: invalid response structure ──────────────────────────────

@pytest.mark.asyncio
async def test_fetch_invalid_response_structure():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = "plain string"

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "Invalid response structure" in result.error


# ── fetch: json decode error ───────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_json_decode_error():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.side_effect = ValueError("Invalid JSON")

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert result.error is not None
