"""Unit tests for Crawl4aiFetchProvider (/crawl endpoint)."""
from unittest.mock import MagicMock, patch

import pytest

from services.search.fetch_providers import Crawl4aiFetchProvider, FetchResult


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


# ── _extract_markdown ──────────────────────────────────────────────

def test_extract_markdown_fit_priority():
    result = {
        "markdown": {
            "fit_markdown": "# Fit content",
            "raw_markdown": "# Raw content",
        }
    }
    assert Crawl4aiFetchProvider._extract_markdown(result) == "# Fit content"


def test_extract_markdown_raw_fallback():
    result = {"markdown": {"raw_markdown": "# Raw only"}}
    assert Crawl4aiFetchProvider._extract_markdown(result) == "# Raw only"


def test_extract_markdown_missing_markdown_key():
    assert Crawl4aiFetchProvider._extract_markdown({}) == ""
    assert Crawl4aiFetchProvider._extract_markdown({"markdown": None}) == ""


def test_extract_markdown_empty_values():
    result = {"markdown": {"fit_markdown": "", "raw_markdown": "  "}}
    assert Crawl4aiFetchProvider._extract_markdown(result) == ""


# ── _extract_title ─────────────────────────────────────────────────

def test_extract_title_from_metadata():
    result = {"metadata": {"title": "My Page"}}
    assert Crawl4aiFetchProvider._extract_title(result) == "My Page"


def test_extract_title_missing_metadata():
    assert Crawl4aiFetchProvider._extract_title({}) == ""
    assert Crawl4aiFetchProvider._extract_title({"metadata": None}) == ""
    assert Crawl4aiFetchProvider._extract_title({"metadata": {"title": ""}}) == ""


# ── create_from_settings ───────────────────────────────────────────

def test_create_from_settings_defaults():
    prov = Crawl4aiFetchProvider.create_from_settings({})
    assert prov._base_url == "http://localhost:11235"
    assert prov._anti_bot is True
    assert prov._timeout == 30
    assert prov._only_text is False


def test_create_from_settings_custom_values():
    prov = Crawl4aiFetchProvider.create_from_settings({
        "crawl4ai_url": "http://custom:9999/",
        "crawl4ai_anti_bot": False,
        "crawl4ai_timeout": 45,
        "crawl4ai_only_text": True,
    })
    assert prov._base_url == "http://custom:9999"
    assert prov._anti_bot is False
    assert prov._timeout == 45
    assert prov._only_text is True


def test_create_from_settings_whitespace_url():
    prov = Crawl4aiFetchProvider.create_from_settings({"crawl4ai_url": "  "})
    assert prov._base_url == "http://localhost:11235"


# ── name property ──────────────────────────────────────────────────

def test_provider_name():
    assert Crawl4aiFetchProvider().name == "crawl4ai"


# ── fetch: success path ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_success():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": True,
            "url": "https://example.com",
            "markdown": {
                "fit_markdown": "# Hello World",
                "raw_markdown": "# Hello World\n\nExtra stuff",
            },
            "metadata": {"title": "Test Page"},
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is True
    assert result.content == "# Hello World"
    assert result.title == "Test Page"
    assert result.url == "https://example.com"
    assert len(fake_client.post_calls) == 1
    assert fake_client.post_calls[0][0] == "http://test.local/crawl"
    payload = fake_client.post_calls[0][1]["json"]
    assert payload["urls"] == ["https://example.com"]
    assert payload["browser_config"]["enable_stealth"] is True
    assert payload["crawler_config"]["magic"] is True


@pytest.mark.asyncio
async def test_fetch_anti_bot_disabled():
    prov = Crawl4aiFetchProvider(base_url="http://test.local", anti_bot=False)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": True,
            "markdown": {"raw_markdown": "content"},
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        await prov.fetch("https://example.com")

    payload = fake_client.post_calls[0][1]["json"]
    assert payload["browser_config"]["enable_stealth"] is False
    assert payload["crawler_config"]["magic"] is False
    assert payload["crawler_config"]["simulate_user"] is False


@pytest.mark.asyncio
async def test_fetch_only_text_enabled():
    prov = Crawl4aiFetchProvider(base_url="http://test.local", only_text=True)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": True,
            "markdown": {"raw_markdown": "content"},
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        await prov.fetch("https://example.com")

    payload = fake_client.post_calls[0][1]["json"]
    assert payload["crawler_config"]["only_text"] is True


@pytest.mark.asyncio
async def test_fetch_only_text_default_false():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": True,
            "markdown": {"raw_markdown": "content"},
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        await prov.fetch("https://example.com")

    payload = fake_client.post_calls[0][1]["json"]
    assert payload["crawler_config"]["only_text"] is False


@pytest.mark.asyncio
async def test_fetch_uses_custom_timeout():
    prov = Crawl4aiFetchProvider(base_url="http://test.local", timeout=45)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{"success": True, "markdown": {"raw_markdown": "ok"}}],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client) as mock_cls:
        await prov.fetch("https://example.com")

    # The timeout is passed to AsyncClient constructor
    assert mock_cls.call_args[1]["timeout"] == 45


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


# ── fetch: top-level success=false ─────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_top_level_failure():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": False,
        "error_message": "Server configuration error",
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "Server configuration error" in result.error


# ── fetch: per-result success=false ────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_result_level_failure():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": False,
            "error_message": "Blocked by anti-bot protection: DataDome captcha",
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "DataDome" in result.error


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
    prov = Crawl4aiFetchProvider(base_url="http://test.local", timeout=30)

    fake_client = _make_fake_client(httpx.TimeoutException("Timed out"))

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "timed out" in result.error
    assert "30s" in result.error


# ── fetch: invalid response ────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_missing_results():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"success": True}

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "missing results" in result.error


@pytest.mark.asyncio
async def test_fetch_empty_results():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"success": True, "results": []}

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is False
    assert "missing results" in result.error


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


# ── fetch: markdown fallback ───────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_falls_back_to_raw_markdown():
    prov = Crawl4aiFetchProvider(base_url="http://test.local")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "success": True,
        "results": [{
            "success": True,
            "markdown": {"raw_markdown": "# Only raw available"},
        }],
    }

    fake_client = _make_fake_client(fake_resp)

    with patch("services.search.fetch_providers.httpx.AsyncClient", return_value=fake_client):
        result = await prov.fetch("https://example.com")

    assert result.success is True
    assert result.content == "# Only raw available"
