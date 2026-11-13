from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from moderationkit.categories import CategoryDefinition
from moderationkit.types import (
    ClassificationResult,
    InferenceOptions,
    ProviderCapabilities,
    TextInput,
)


@runtime_checkable
class TextProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def classify_text(
        self,
        request: TextInput,
        *,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
    ) -> ClassificationResult: ...

    def close(self) -> None: ...


@runtime_checkable
class AsyncTextProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def aclassify_text(
        self,
        request: TextInput,
        *,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
    ) -> ClassificationResult: ...

    async def aclose(self) -> None: ...
