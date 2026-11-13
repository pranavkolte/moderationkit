import asyncio
import math
import threading
import time

import pytest

from moderationkit import CategoryDefinition, TextInput
from moderationkit.exceptions import CapacityError, ModerationTimeout, ProviderResponseError
from moderationkit.providers.laya import LayaConfig, LayaProvider
from moderationkit.providers.laya.normalize import normalize
from moderationkit.providers.laya.questions import compile_questions, questions_hash
from moderationkit.types import (
    CategorySignal,
    ClassificationResult,
    InferenceOptions,
    ProviderMetadata,
)

CATEGORIES = (CategoryDefinition(id="harassment", description="Personal abuse"),)


@pytest.mark.parametrize("value", [True, "0.7", None, -1, 2, math.inf, math.nan])
def test_invalid_answer_unavailable(value):
    result = normalize(
        {"answers": {"harassment": {"type": "noul", "noul": value}}, "usage": {"truncated": False}},
        CATEGORIES,
        ProviderMetadata(provider="laya"),
    )
    assert result.categories[0].probability is None


def test_probability_not_answer_confidence_and_missing_coverage():
    raw = {"answers": {"harassment": {"type": "noul", "noul": 0.02, "confidence": 0.98}}}
    result = normalize(raw, CATEGORIES, ProviderMetadata(provider="laya"))
    assert result.categories[0].probability == 0.02
    assert not result.coverage_complete
    with pytest.raises(ProviderResponseError):
        normalize({}, CATEGORIES, ProviderMetadata(provider="laya"))


def test_questions_stable_and_independent():
    questions = compile_questions(
        CATEGORIES + (CategoryDefinition(id="custom.ad", description="Ads"),)
    )
    assert len(questions) == 2
    assert all(q["type"] == "noul" for q in questions.values())
    assert questions_hash(questions) == questions_hash(dict(reversed(list(questions.items()))))


class ControlledLaya(LayaProvider):
    def __init__(self, **kwargs):
        super().__init__(LayaConfig(**kwargs))
        self.started = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()

    def check_dependencies(self):
        pass

    def _predict(self, *args):
        self.started.set()
        assert self.release.wait(3), "test did not release inference"
        self.done.set()
        return ClassificationResult(
            categories=(CategorySignal(name="harassment", probability=0.2),),
            metadata=ProviderMetadata(provider="controlled"),
            coverage_complete=True,
        )


async def test_timeout_retains_capacity_and_loop_responsive():
    provider = ControlledLaya(capacity=1)
    try:
        task = asyncio.create_task(
            provider.aclassify_text(
                TextInput(text="x"),
                categories=CATEGORIES,
                options=InferenceOptions(request_id="x", deadline=time.monotonic() + 0.1),
            )
        )
        assert await asyncio.to_thread(provider.started.wait, 1)
        await asyncio.sleep(0.01)  # this runs while native inference remains blocked
        with pytest.raises(ModerationTimeout):
            await task
        with pytest.raises(CapacityError):
            provider.classify_text(
                TextInput(text="x"), categories=CATEGORIES, options=InferenceOptions(request_id="y")
            )
    finally:
        provider.release.set()
        await provider.aclose()
    assert provider.done.is_set()


async def test_cancel_retains_capacity():
    provider = ControlledLaya(capacity=1)
    try:
        task = asyncio.create_task(
            provider.aclassify_text(
                TextInput(text="x"), categories=CATEGORIES, options=InferenceOptions(request_id="x")
            )
        )
        assert await asyncio.to_thread(provider.started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(CapacityError):
            provider.classify_text(
                TextInput(text="x"), categories=CATEGORIES, options=InferenceOptions(request_id="y")
            )
    finally:
        provider.release.set()
        await provider.aclose()
