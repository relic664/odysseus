"""Fetch provider abstraction for webpage content extraction."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from .content import fetch_webpage_content

logger = logging.getLogger(__name__)

FETCH_PROVIDER_INFO = {
    "simple": ("Simple (HTTP + BS4)", False, False),
    "crawl4ai": ("Crawl4ai", False, True),
}


@dataclass
class FetchResult:
    """Standardized fetch result. Content is provider-agnostic text."""
    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None


class FetchProvider(ABC):
    """Interface for webpage fetch providers.

    All providers are async. Providers that wrap sync libraries
    use asyncio.to_thread() internally.
    """

    @abstractmethod
    async def fetch(self, url: str, timeout: int = 15) -> FetchResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SimpleFetchProvider(FetchProvider):
    """Built-in fetcher: HTTP + BeautifulSoup.

    Wraps fetch_webpage_content() from services/search/content.py.
    Keeps SSRF protection and HTML parsing. Uses asyncio.to_thread()
    to avoid blocking the event loop. Extracts only title and content
    from the result.
    """

    @property
    def name(self) -> str:
        return "simple"

    async def fetch(self, url: str, timeout: int = 15) -> FetchResult:
        try:
            result = await asyncio.to_thread(fetch_webpage_content, url, timeout=timeout)
            return FetchResult(
                url=result.get("url", url),
                title=result.get("title", ""),
                content=result.get("content", ""),
                success=result.get("success", False),
                error=result.get("error") or None,
            )
        except Exception as e:
            logger.error(f"SimpleFetchProvider failed for {url}: {e}")
            return FetchResult(
                url=url,
                title="",
                content="",
                success=False,
                error=str(e),
            )


class Crawl4aiFetchProvider(FetchProvider):
    """Fetch provider using the Crawl4ai REST API (/md endpoint).

    Returns clean markdown content. Handles JS-rendered pages and
    dynamic content. Connects to a self-hosted crawl4ai instance
    (typically a Docker container). No API key required.
    """

    DEFAULT_BASE_URL = "http://localhost:11235"

    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "crawl4ai"

    @classmethod
    def create_from_settings(cls, settings: Dict[str, Any]) -> "Crawl4aiFetchProvider":
        base_url = (settings.get("crawl4ai_url") or "").strip()
        if not base_url:
            base_url = cls.DEFAULT_BASE_URL
        return cls(base_url=base_url)

    @staticmethod
    def _extract_markdown(data: Dict[str, Any]) -> str:
        """Extract markdown content from crawl4ai response.

        Tries fields in order of quality: fit > filtered > raw.
        Also checks for nested markdown object (older API versions).
        """
        for key in ("fit_markdown", "filtered_markdown", "markdown", "raw_markdown", "content", "page_content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        markdown_field = data.get("markdown")
        if isinstance(markdown_field, dict):
            for key in ("fit_markdown", "filtered_markdown", "markdown", "raw_markdown"):
                value = markdown_field.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    @staticmethod
    def _extract_title(data: Dict[str, Any]) -> str:
        title = data.get("title", "")
        if title and isinstance(title, str):
            return title.strip()
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            title = metadata.get("title", "")
            if title and isinstance(title, str):
                return title.strip()
        return ""

    async def fetch(self, url: str, timeout: int = 15) -> FetchResult:
        endpoint = f"{self._base_url}/md"
        payload = {"url": url, "f": "fit"}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload)

            if resp.status_code != 200:
                body_preview = resp.text[:300] if resp.text else "(empty)"
                return FetchResult(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error=f"Crawl4ai returned HTTP {resp.status_code}: {body_preview}",
                )

            data = resp.json()

            results = self._decode_results(data)
            if not results:
                return FetchResult(
                    url=url,
                    title="",
                    content="",
                    success=False,
                    error="Invalid response structure from Crawl4ai",
                )

            result = results[0]
            return FetchResult(
                url=url,
                title=self._extract_title(result),
                content=self._extract_markdown(result),
                success=True,
            )

        except httpx.ConnectError as e:
            return FetchResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"Cannot connect to Crawl4ai at {self._base_url}. Is the container running?",
            )
        except httpx.TimeoutException:
            return FetchResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"Crawl4ai timed out after {timeout}s",
            )
        except Exception as e:
            logger.error(f"Crawl4aiFetchProvider failed for {url}: {e}")
            return FetchResult(
                url=url,
                title="",
                content="",
                success=False,
                error=str(e),
            )

    def _decode_results(self, data: Any) -> list:
        """Decode crawl4ai response into a list of result dicts.

        Handles multiple response shapes: top-level array, {results: [...]},
        {data: [...]}, or single object wrapped in a list.
        """
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("results", "data"):
                val = data.get(key)
                if isinstance(val, list):
                    return [item for item in val if isinstance(item, dict)]
            return [data]
        return []
