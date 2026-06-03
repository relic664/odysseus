"""Fetch provider selection and fallback management."""

import logging
from typing import Any, Callable, Dict, List, Optional

from .fetch_providers import FetchProvider, FetchResult, SimpleFetchProvider, Crawl4aiFetchProvider

logger = logging.getLogger(__name__)


class FetchManager:
    """Manages fetch provider selection and fallback."""

    _providers: Dict[str, FetchProvider] = {}
    _factories: Dict[str, Callable[[Dict[str, Any]], FetchProvider]] = {}

    @classmethod
    def register(cls, provider: FetchProvider) -> None:
        cls._providers[provider.name] = provider

    @classmethod
    def register_factory(cls, name: str, factory: Callable[[Dict[str, Any]], FetchProvider]) -> None:
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> Optional[FetchProvider]:
        return cls._providers.get(name)

    @classmethod
    def is_available(cls, name: str) -> bool:
        return name in cls._providers or name in cls._factories

    def _resolve_provider(self, name: str, settings: Dict[str, Any]) -> Optional[FetchProvider]:
        if name in self._providers:
            return self._providers[name]
        factory = self._factories.get(name)
        if factory:
            provider = factory(settings)
            self._providers[name] = provider
            return provider
        return None

    async def fetch(
        self,
        url: str,
        provider_name: str,
        settings: Dict[str, Any],
        fallbacks: Optional[List[str]] = None,
        timeout: int = 15,
    ) -> FetchResult:
        chain = [provider_name]
        if fallbacks:
            for fb in fallbacks:
                if fb != provider_name and fb not in chain:
                    chain.append(fb)

        for name in chain:
            provider = self._resolve_provider(name, settings)
            if not provider:
                logger.warning(f"Fetch provider '{name}' not available, skipping")
                continue

            try:
                result = await provider.fetch(url, timeout=timeout)
                if result.success:
                    return result
                logger.warning(
                    f"Fetch provider '{name}' failed for {url}: {result.error}"
                )
            except Exception as e:
                logger.error(f"Fetch provider '{name}' raised for {url}: {e}")

        return FetchResult(
            url=url,
            title="",
            content="",
            success=False,
            error=f"All fetch providers failed for {url}",
        )


# Register built-in providers at import time
FetchManager.register(SimpleFetchProvider())
FetchManager.register_factory("crawl4ai", Crawl4aiFetchProvider.create_from_settings)
