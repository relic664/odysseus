"""Fetch provider abstraction for webpage content extraction."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .content import fetch_webpage_content

logger = logging.getLogger(__name__)

FETCH_PROVIDER_INFO = {
    "simple": ("Simple (HTTP + BS4)", False, False),
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
