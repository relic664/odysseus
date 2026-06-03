"""Unit tests for FetchManager: factory registration, fallback chain, availability."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from services.search.fetch_manager import FetchManager
from services.search.fetch_providers import FetchProvider, FetchResult, SimpleFetchProvider, Crawl4aiFetchProvider


class _DummyProvider(FetchProvider):
    """Minimal provider for testing."""

    def __init__(self, name: str, should_fail: bool = False, error_msg: str = "fail"):
        self._name = name
        self._should_fail = should_fail
        self._error_msg = error_msg
        self.fetch_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def fetch(self, url: str, timeout: int = 15) -> FetchResult:
        self.fetch_count += 1
        if self._should_fail:
            return FetchResult(url=url, title="", content="", success=False, error=self._error_msg)
        return FetchResult(url=url, title="ok", content="success content", success=True)


# ── is_available ───────────────────────────────────────────────────

def test_is_available_simple_always_true():
    # Simple is registered at import time
    assert FetchManager.is_available("simple") is True


def test_is_available_crawl4ai_registered():
    # Crawl4ai factory is registered at import time
    assert FetchManager.is_available("crawl4ai") is True


def test_is_available_unknown_false():
    assert FetchManager.is_available("nonexistent_provider") is False


# ── register / get ─────────────────────────────────────────────────

def test_register_and_get():
    fm = FetchManager()
    prov = _DummyProvider("test_reg")
    fm.register(prov)
    assert fm.get("test_reg") is prov


def test_get_unknown_returns_none():
    fm = FetchManager()
    assert fm.get("does_not_exist") is None


# ── register_factory ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_factory():
    fm = FetchManager()
    factory_called = []

    def my_factory(settings):
        factory_called.append(settings)
        return _DummyProvider("factory_prov")

    fm.register_factory("factory_test", my_factory)

    await fm.fetch("https://example.com", "factory_test", {"key": "val"})

    assert len(factory_called) == 1
    assert factory_called[0] == {"key": "val"}


# ── _resolve_provider ──────────────────────────────────────────────

def test_resolve_registered_provider():
    fm = FetchManager()
    prov = _DummyProvider("direct")
    fm.register(prov)
    resolved = fm._resolve_provider("direct", {})
    assert resolved is prov


def test_resolve_factory_creates_and_caches():
    fm = FetchManager()

    def factory(settings):
        return _DummyProvider("cached_prov")

    fm.register_factory("cached_test", factory)

    # First call creates
    p1 = fm._resolve_provider("cached_test", {})
    assert p1 is not None

    # Second call returns cached
    p2 = fm._resolve_provider("cached_test", {})
    assert p2 is p1


def test_resolve_unknown_returns_none():
    fm = FetchManager()
    assert fm._resolve_provider("no_such_provider", {}) is None


# ── fetch: single provider success ─────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_single_provider_success():
    fm = FetchManager()
    prov = _DummyProvider("ok_prov")
    fm.register(prov)

    result = await fm.fetch("https://example.com", "ok_prov", {})

    assert result.success is True
    assert result.content == "success content"
    assert prov.fetch_count == 1


# ── fetch: fallback chain ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_fallback_to_second_provider():
    fm = FetchManager()
    bad = _DummyProvider("bad", should_fail=True, error_msg="bad error")
    good = _DummyProvider("good")
    fm.register(bad)
    fm.register(good)

    result = await fm.fetch("https://example.com", "bad", {}, fallbacks=["good"])

    assert result.success is True
    assert result.content == "success content"
    assert bad.fetch_count == 1
    assert good.fetch_count == 1


@pytest.mark.asyncio
async def test_fetch_fallback_all_fail():
    fm = FetchManager()
    bad1 = _DummyProvider("bad1", should_fail=True, error_msg="err1")
    bad2 = _DummyProvider("bad2", should_fail=True, error_msg="err2")
    fm.register(bad1)
    fm.register(bad2)

    result = await fm.fetch("https://example.com", "bad1", {}, fallbacks=["bad2"])

    assert result.success is False
    assert "All fetch providers failed" in result.error
    assert bad1.fetch_count == 1
    assert bad2.fetch_count == 1


@pytest.mark.asyncio
async def test_fetch_stops_on_first_success():
    fm = FetchManager()
    good = _DummyProvider("good")
    also_good = _DummyProvider("also_good")
    fm.register(good)
    fm.register(also_good)

    result = await fm.fetch("https://example.com", "good", {}, fallbacks=["also_good"])

    assert result.success is True
    assert good.fetch_count == 1
    assert also_good.fetch_count == 0


# ── fetch: skips unavailable providers ─────────────────────────────

@pytest.mark.asyncio
async def test_fetch_skips_unavailable_fallback():
    fm = FetchManager()
    good = _DummyProvider("good")
    fm.register(good)

    result = await fm.fetch("https://example.com", "nonexistent", {}, fallbacks=["good"])

    assert result.success is True
    assert good.fetch_count == 1


@pytest.mark.asyncio
async def test_fetch_all_unavailable():
    fm = FetchManager()

    result = await fm.fetch("https://example.com", "nope1", {}, fallbacks=["nope2"])

    assert result.success is False
    assert "All fetch providers failed" in result.error


# ── fetch: deduplicates fallback chain ─────────────────────────────

@pytest.mark.asyncio
async def test_fetch_deduplicates_primary_in_fallbacks():
    fm = FetchManager()
    prov = _DummyProvider("dedup")
    fm.register(prov)

    result = await fm.fetch("https://example.com", "dedup", {}, fallbacks=["dedup"])

    assert result.success is True
    assert prov.fetch_count == 1  # Only called once, not twice


# ── fetch: timeout param passed through ────────────────────────────

@pytest.mark.asyncio
async def test_fetch_timeout_passed_to_provider():
    fm = FetchManager()

    class _TimeoutCaptureProvider(FetchProvider):
        @property
        def name(self):
            return "tc"

        async def fetch(self, url, timeout=15):
            self.captured_timeout = timeout
            return FetchResult(url=url, title="", content="ok", success=True)

    prov = _TimeoutCaptureProvider()
    fm.register(prov)

    await fm.fetch("https://example.com", "tc", {}, timeout=42)

    assert prov.captured_timeout == 42


# ── fetch: exception in provider caught ────────────────────────────

@pytest.mark.asyncio
async def test_fetch_provider_exception_tried_next():
    fm = FetchManager()

    class _RaisingProvider(FetchProvider):
        @property
        def name(self):
            return "raises"

        async def fetch(self, url, timeout=15):
            raise RuntimeError("boom")

    good = _DummyProvider("good")
    fm.register(_RaisingProvider())
    fm.register(good)

    result = await fm.fetch("https://example.com", "raises", {}, fallbacks=["good"])

    assert result.success is True
    assert good.fetch_count == 1


# ── crawl4ai factory integration ───────────────────────────────────

@pytest.mark.asyncio
async def test_crawl4ai_factory_creates_with_url():
    fm = FetchManager()
    fm.register_factory("c4ai_custom", Crawl4aiFetchProvider.create_from_settings)

    result = fm._resolve_provider("c4ai_custom", {
        "crawl4ai_url": "http://custom:1234/",
        "crawl4ai_anti_bot": False,
        "crawl4ai_timeout": 45,
    })

    assert isinstance(result, Crawl4aiFetchProvider)
    assert result._base_url == "http://custom:1234"
    assert result._anti_bot is False
    assert result._timeout == 45


@pytest.mark.asyncio
async def test_crawl4ai_factory_defaults():
    fm = FetchManager()
    fm.register_factory("c4ai_default", Crawl4aiFetchProvider.create_from_settings)

    result = fm._resolve_provider("c4ai_default", {})

    assert isinstance(result, Crawl4aiFetchProvider)
    assert result._base_url == "http://localhost:11235"
    assert result._anti_bot is True
    assert result._timeout == 30
