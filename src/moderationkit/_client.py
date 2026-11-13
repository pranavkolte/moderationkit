from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Sequence
from typing import Any

from moderationkit._batch import summarize, validate_batch
from moderationkit._pipeline import failure, request_options, resolve_categories, text_input
from moderationkit.categories import CategoryDefinition
from moderationkit.exceptions import (
    BatchModerationError,
    ClosedError,
    ConfigurationError,
    InputError,
    ModerationError,
    ModerationTimeout,
    ProviderError,
)
from moderationkit.policies import Policy, RiskScorer
from moderationkit.policies.engine import evaluate
from moderationkit.providers import AsyncTextProvider, TextProvider
from moderationkit.types import (
    BatchItem,
    BatchResult,
    CategorySignal,
    ClassificationResult,
    InferenceOptions,
    ModerationResult,
    ProviderMetadata,
    TextInput,
)


def _timeout(value: float | None) -> float | None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
    ):
        raise ConfigurationError("timeout must be a finite positive number or None")
    return value


class _ClientState:
    def __init__(
        self,
        *,
        provider: Any,
        policy: Policy,
        categories: Sequence[CategoryDefinition] = (),
        timeout: float | None = 60.0,
        raise_on_error: bool = False,
        own_provider: bool = False,
        risk_scorer: RiskScorer | None = None,
        batch_concurrency: int = 4,
    ) -> None:
        self._provider = provider
        self._policy = Policy.model_validate(policy.model_dump())
        self._categories = resolve_categories(self._policy, categories)
        self._timeout = _timeout(timeout)
        if type(batch_concurrency) is not int or batch_concurrency < 1:
            raise ConfigurationError("batch_concurrency must be a positive integer")
        self._batch_concurrency = batch_concurrency
        self._raise = raise_on_error
        self._own = own_provider
        self._risk_scorer = risk_scorer
        self._condition = threading.Condition()
        self._active = 0
        self._closing = False
        self._drained = threading.Event()

    @property
    def policy(self) -> Policy:
        return self._policy.model_copy(deep=True)

    def _enter(self) -> None:
        with self._condition:
            if self._closing:
                raise ClosedError("Moderation client is closed")
            self._active += 1

    def _leave(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def _start_close(self) -> bool:
        with self._condition:
            if self._closing:
                return False
            self._closing = True
            return True

    def _wait_active(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._active == 0)

    def _selection(self) -> tuple[tuple[CategoryDefinition, ...], tuple[CategorySignal, ...]]:
        caps = self._provider.capabilities
        supported = []
        unavailable = []
        for category in self._categories:
            ok = (
                "text" in caps.modalities
                and (caps.custom_categories or not category.id.startswith("custom."))
                and (caps.supported_categories is None or category.id in caps.supported_categories)
            )
            if ok:
                supported.append(category)
            else:
                unavailable.append(
                    CategorySignal(name=category.id, unavailable_reason="unsupported")
                )
        return tuple(supported), tuple(unavailable)

    def _error(
        self, error: ProviderError | InputError, options: InferenceOptions, item_id: str | None
    ) -> ModerationResult:
        return failure(
            self._policy,
            error,
            options=options,
            item_id=item_id,
            provider_name=getattr(self._provider, "name", type(self._provider).__name__),
        )

    def _finish(
        self,
        raw: ClassificationResult,
        unavailable: tuple[CategorySignal, ...],
        options: InferenceOptions,
        request: TextInput,
        started: float,
    ) -> ModerationResult:
        if options.remaining() == 0:
            raise ModerationTimeout("Provider completed after the request deadline")
        requested = {c.id for c in self._categories}
        # Never let an unexpected provider signal stand in for a requested category.
        raw = raw.model_copy(
            update={
                "categories": tuple(
                    c
                    for c in raw.categories
                    if c.name in requested and c.name not in {u.name for u in unavailable}
                )
                + unavailable
            }
        )
        result = evaluate(
            raw,
            self._policy,
            request_id=options.request_id,
            item_id=request.item_id,
            risk_scorer=self._risk_scorer,
        )
        result.metadata.latency_ms = (time.monotonic() - started) * 1000
        result.metadata.unsupported_capabilities = tuple(f"category:{u.name}" for u in unavailable)
        return result

    def _check_raise(self, result: ModerationResult) -> ModerationResult:
        if self._raise and result.status != "completed":
            raise ModerationError(result)
        return result

    def _invalid_item(self, item: object) -> ModerationResult:
        return self._error(
            InputError("batch item must be a TextInput"),
            request_options(None),
            getattr(item, "item_id", None),
        )


class ModGuard(_ClientState):
    def __init__(
        self,
        *,
        provider: TextProvider,
        policy: Policy,
        categories: Sequence[CategoryDefinition] = (),
        timeout: float | None = 60.0,
        raise_on_error: bool = False,
        own_provider: bool = False,
        risk_scorer: RiskScorer | None = None,
        batch_concurrency: int = 4,
    ) -> None:
        super().__init__(
            provider=provider,
            policy=policy,
            categories=categories,
            timeout=timeout,
            raise_on_error=raise_on_error,
            own_provider=own_provider,
            risk_scorer=risk_scorer,
            batch_concurrency=batch_concurrency,
        )

    def moderate_text(
        self,
        text: str,
        *,
        item_id: str | None = None,
        language: str | None = None,
        timeout: float | None = None,
    ) -> ModerationResult:
        return self.moderate(text_input(text, item_id, language), timeout=timeout)

    def _run(self, request: TextInput, timeout: float | None) -> ModerationResult:
        if not isinstance(request, TextInput):
            raise InputError("moderate expects a TextInput")
        options = request_options(self._timeout if timeout is None else _timeout(timeout))
        started = time.monotonic()
        try:
            supported, unavailable = self._selection()
            raw = (
                self._provider.classify_text(request, categories=supported, options=options)
                if supported
                else ClassificationResult(
                    categories=(), metadata=ProviderMetadata(provider=type(self._provider).__name__)
                )
            )
            return self._finish(raw, unavailable, options, request, started)
        except ProviderError as exc:
            result = self._error(exc, options, request.item_id)
            result.metadata.latency_ms = (time.monotonic() - started) * 1000
            return result

    def moderate(self, request: TextInput, *, timeout: float | None = None) -> ModerationResult:
        self._enter()
        try:
            return self._check_raise(self._run(request, timeout))
        finally:
            self._leave()

    def moderate_batch(
        self, items: Sequence[TextInput], *, timeout: float | None = None
    ) -> BatchResult:
        validate_batch(items)
        _timeout(timeout)
        self._enter()
        started = time.monotonic()
        try:
            results = tuple(
                BatchItem(
                    index=i,
                    result=self._run(item, timeout)
                    if isinstance(item, TextInput)
                    else self._invalid_item(item),
                )
                for i, item in enumerate(items)
            )
            report = summarize(results, (time.monotonic() - started) * 1000)
            if self._raise and report.failed:
                raise BatchModerationError(report)
            return report
        finally:
            self._leave()

    def close(self) -> None:
        if not self._start_close():
            self._drained.wait()
            return
        try:
            self._wait_active()
            if self._own:
                self._provider.close()
        finally:
            self._drained.set()

    def __enter__(self) -> ModGuard:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncModGuard(_ClientState):
    def __init__(
        self,
        *,
        provider: AsyncTextProvider,
        policy: Policy,
        categories: Sequence[CategoryDefinition] = (),
        timeout: float | None = 60.0,
        raise_on_error: bool = False,
        own_provider: bool = False,
        risk_scorer: RiskScorer | None = None,
        batch_concurrency: int = 4,
    ) -> None:
        super().__init__(
            provider=provider,
            policy=policy,
            categories=categories,
            timeout=timeout,
            raise_on_error=raise_on_error,
            own_provider=own_provider,
            risk_scorer=risk_scorer,
            batch_concurrency=batch_concurrency,
        )

    async def moderate_text(
        self,
        text: str,
        *,
        item_id: str | None = None,
        language: str | None = None,
        timeout: float | None = None,
    ) -> ModerationResult:
        return await self.moderate(text_input(text, item_id, language), timeout=timeout)

    async def _run(self, request: TextInput, timeout: float | None) -> ModerationResult:
        if not isinstance(request, TextInput):
            raise InputError("moderate expects a TextInput")
        options = request_options(self._timeout if timeout is None else _timeout(timeout))
        started = time.monotonic()
        try:
            supported, unavailable = self._selection()
            if supported:
                raw = await asyncio.wait_for(
                    self._provider.aclassify_text(request, categories=supported, options=options),
                    timeout=options.remaining(),
                )
            else:
                raw = ClassificationResult(
                    categories=(), metadata=ProviderMetadata(provider=type(self._provider).__name__)
                )
            return self._finish(raw, unavailable, options, request, started)
        except TimeoutError:
            result = self._error(
                ModerationTimeout("Provider deadline exceeded"), options, request.item_id
            )
            result.metadata.latency_ms = (time.monotonic() - started) * 1000
            return result
        except ProviderError as exc:
            result = self._error(exc, options, request.item_id)
            result.metadata.latency_ms = (time.monotonic() - started) * 1000
            return result

    async def moderate(
        self, request: TextInput, *, timeout: float | None = None
    ) -> ModerationResult:
        self._enter()
        try:
            return self._check_raise(await self._run(request, timeout))
        finally:
            self._leave()

    async def moderate_batch(
        self, items: Sequence[TextInput], *, timeout: float | None = None
    ) -> BatchResult:
        validate_batch(items)
        _timeout(timeout)
        self._enter()
        started = time.monotonic()
        results: list[BatchItem | None] = [None] * len(items)
        iterator = iter(enumerate(items))

        async def worker() -> None:
            for index, item in iterator:
                result = (
                    await self._run(item, timeout)
                    if isinstance(item, TextInput)
                    else self._invalid_item(item)
                )
                results[index] = BatchItem(index=index, result=result)

        workers = [
            asyncio.create_task(worker()) for _ in range(min(len(items), self._batch_concurrency))
        ]
        try:
            await asyncio.gather(*workers)
            report = summarize(
                tuple(r for r in results if r is not None), (time.monotonic() - started) * 1000
            )
            if self._raise and report.failed:
                raise BatchModerationError(report)
            return report
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._leave()

    async def aclose(self) -> None:
        async def drain() -> None:
            if not self._start_close():
                await asyncio.to_thread(self._drained.wait)
                return
            try:
                await asyncio.to_thread(self._wait_active)
                if self._own:
                    await self._provider.aclose()
            finally:
                self._drained.set()

        task = asyncio.create_task(drain())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def __aenter__(self) -> AsyncModGuard:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
