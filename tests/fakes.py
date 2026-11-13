import asyncio

from moderationkit.exceptions import ProviderError
from moderationkit.types import (
    CategorySignal,
    ClassificationResult,
    ProviderCapabilities,
    ProviderMetadata,
)


class FakeProvider:
    capabilities = ProviderCapabilities()

    def __init__(self, scores=None, *, fail=False, partial=False, coverage=True):
        self.scores = scores or {}
        self.fail = fail
        self.partial = partial
        self.coverage = coverage
        self.closed = False
        self.calls = 0

    def classify_text(self, request, *, categories, options):
        self.calls += 1
        if self.fail or request.text == "fail":
            raise ProviderError("synthetic failure")
        chosen = categories[:1] if self.partial else categories
        return ClassificationResult(
            categories=tuple(
                CategorySignal(name=c.id, probability=self.scores.get(c.id, 0.1)) for c in chosen
            ),
            metadata=ProviderMetadata(provider="independent_fake", model="fixture-v1"),
            coverage_complete=self.coverage,
        )

    async def aclassify_text(self, request, *, categories, options):
        await asyncio.sleep(0)
        return self.classify_text(request, categories=categories, options=options)

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()
