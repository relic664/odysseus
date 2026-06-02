"""Fetch provider selection and fallback management."""

import logging
from typing import Dict, List, Optional

from .fetch_providers import FetchProvider, FetchResult, SimpleFetchProvider

logger = logging.getLogger(__name__)


class FetchManager:
    """Manages fetch provider selection and fallback."""

    _providers: Dict[str, FetchProvider] = {}

    @classmethod
    def register(cls, provider: FetchProvider) -> None:
        cls._providers[provider.name] = provider

    @classmethod
    def get(cls, name: str) -> Optional[FetchProvider]:
        return cls._providers.get(name)

    async def fetch(
        self,
        url: str,
        provider_name: str,
        fallbacks: Optional[List[str]] = None,
        timeout: int = 15,
    ) -> FetchResult:
        chain = [provider_name]
        if fallbacks:
            for fb in fallbacks:
                if fb != provider_name and fb not in chain:
                    chain.append(fb)

        for name in chain:
            provider = self._providers.get(name)
            if not provider:
                logger.warning(f"Fetch provider '{name}' not registered, skipping")
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
