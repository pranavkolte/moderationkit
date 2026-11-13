from __future__ import annotations

import asyncio
import importlib
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from moderationkit.categories import CategoryDefinition
from moderationkit.exceptions import (
    CapacityError,
    ClosedError,
    MissingDependencyError,
    ModerationTimeout,
    ProviderError,
)
from moderationkit.types import (
    ClassificationResult,
    InferenceOptions,
    ProviderCapabilities,
    ProviderMetadata,
    TextInput,
)

from .config import LayaConfig
from .normalize import normalize
from .questions import compile_questions, questions_hash


class LayaProvider:
    """One reusable local runtime, with bounded admission and serialized inference.

    Timeouts stop waiting; they do not terminate native model execution. close()
    drains accepted work. Async methods can be used from multiple event loops.
    """

    capabilities = ProviderCapabilities()
    name = "laya"

    def __init__(self, config: LayaConfig | None = None) -> None:
        self.config = config or LayaConfig()
        self._runtime: Any = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.config.capacity)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moderation-laya")
        self._closed = False
        self._drained = threading.Event()

    def check_dependencies(self) -> None:
        try:
            version("laya")
        except PackageNotFoundError as exc:
            raise MissingDependencyError(
                "Install the local model with pip install 'moderationkit[laya]'"
            ) from exc

    def _load(self) -> float:
        if self._runtime is not None:
            return 0.0
        start = time.monotonic()
        self.check_dependencies()
        try:
            module = importlib.import_module("laya")
            self._runtime = module.Router(
                vega=False,
                device=self.config.device,
                revision=self.config.revision,
                max_loaded=self.config.max_loaded,
            )
        except ImportError as exc:
            raise MissingDependencyError(
                "Laya dependencies are unavailable; reinstall moderationkit[laya]"
            ) from exc
        return (time.monotonic() - start) * 1000

    def _submit(
        self,
        request: TextInput,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
    ) -> Future[ClassificationResult]:
        if options.remaining() == 0:
            raise ModerationTimeout("Request deadline expired before admission")
        with self._lock:
            if self._closed:
                raise ClosedError("Laya provider is closed")
            if not self._slots.acquire(blocking=False):
                raise CapacityError(
                    "Laya capacity exhausted; retry after outstanding work completes"
                )
            try:
                future = self._executor.submit(
                    self._predict, request, tuple(categories), options, time.monotonic()
                )
            except BaseException:
                self._slots.release()
                raise
            future.add_done_callback(lambda _: self._slots.release())
            return future

    def _predict(
        self,
        request: TextInput,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
        submitted: float,
    ) -> ClassificationResult:
        if options.remaining() == 0:
            raise ModerationTimeout("Request expired in inference queue")
        started = time.monotonic()
        try:
            load_ms = self._load()
            state = {"text": request.text}
            route = self._runtime.route(
                state,
                model=None if self.config.model == "auto" else self.config.model,
                lang=request.language,
            )
            load_start = time.monotonic()
            cold = route.model not in self._runtime.loaded
            self._runtime.load(route.model)
            if cold:
                load_ms += (time.monotonic() - load_start) * 1000
            if options.remaining() == 0:
                raise ModerationTimeout("Request expired during model initialization")
            questions = compile_questions(categories)
            raw = self._runtime.predict(
                state,
                questions,
                model=route.model,
                lang=request.language,
                max_len=self.config.max_len,
            )
        except (OSError, RuntimeError) as exc:
            raise ProviderError(f"Local Laya inference failed ({type(exc).__name__})") from exc
        metadata = ProviderMetadata(
            provider="laya",
            requested_model=self.config.model,
            revision=self._runtime.loaded_revisions.get(route.model),
            runtime_version=version("laya"),
            questions_hash=questions_hash(questions),
            load_ms=load_ms,
            queue_ms=(started - submitted) * 1000,
            inference_ms=max(0.0, (time.monotonic() - started) * 1000 - load_ms),
        )
        return normalize(raw, categories, metadata)

    def classify_text(
        self,
        request: TextInput,
        *,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
    ) -> ClassificationResult:
        self.check_dependencies()
        future = self._submit(request, categories, options)
        try:
            return future.result(timeout=options.remaining())
        except TimeoutError as exc:
            future.cancel()
            raise ModerationTimeout(
                "Laya deadline exceeded; running inference may continue"
            ) from exc

    async def aclassify_text(
        self,
        request: TextInput,
        *,
        categories: Sequence[CategoryDefinition],
        options: InferenceOptions,
    ) -> ClassificationResult:
        self.check_dependencies()
        future = self._submit(request, categories, options)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(wrapped, timeout=options.remaining())
        except TimeoutError as exc:
            future.cancel()
            raise ModerationTimeout(
                "Laya deadline exceeded; running inference may continue"
            ) from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    def warmup(self, *, timeout: float | None = None) -> None:
        from uuid import uuid4

        options = InferenceOptions(
            request_id=str(uuid4()),
            deadline=None if timeout is None else time.monotonic() + timeout,
        )
        self.classify_text(
            TextInput(text="Warmup"),
            categories=(
                CategoryDefinition(id="warmup", description="The text contains a greeting"),
            ),
            options=options,
        )

    async def awarmup(self, *, timeout: float | None = None) -> None:
        await asyncio.to_thread(self.warmup, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            first = not self._closed
            self._closed = True
        if first:
            try:
                self._executor.shutdown(wait=True)
                self._runtime = None
            finally:
                self._drained.set()
        else:
            self._drained.wait()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __enter__(self) -> LayaProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> LayaProvider:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
