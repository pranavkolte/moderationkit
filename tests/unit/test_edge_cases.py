import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from moderationkit import AsyncModGuard, ModGuard, Policy, Rule, TextInput
from moderationkit.exceptions import ClosedError, ConfigurationError
from moderationkit.types import ProviderCapabilities
from tests.fakes import FakeProvider
from tests.unit.test_laya import CATEGORIES, ControlledLaya


def test_entirely_unsupported_observation_is_not_safe():
    policy = Policy(
        id="observations",
        version="1",
        rules=[],
        observation_categories=["politics"],
        detection_thresholds={"politics": 0.5},
    )
    provider = FakeProvider()
    provider.capabilities = ProviderCapabilities(modalities=frozenset())
    result = ModGuard(provider=provider, policy=policy).moderate_text("x")
    assert result.status == "incomplete" and not result.decision.safe
    assert result.metadata.unsupported_capabilities == ("category:politics",)
    assert provider.calls == 0


def test_no_provider_calibration_leak_and_compatible_scorer(policy):
    class ClaimedCalibration(FakeProvider):
        def classify_text(self, *args, **kwargs):
            result = super().classify_text(*args, **kwargs)
            result.metadata.calibration_status = "validated"
            return result

    provider = ClaimedCalibration()
    result = ModGuard(provider=provider, policy=policy).moderate_text("x")
    assert result.metadata.calibration_status == "unavailable"

    class Scorer:
        def compatible(self, classification, policy):
            return classification.metadata.model == "fixture-v1"

        def score(self, classification, policy):
            return 12.0

    result = ModGuard(provider=provider, policy=policy, risk_scorer=Scorer()).moderate_text("x")
    assert result.decision.risk_score == 12 and result.metadata.calibration_status == "validated"
    provider.capabilities = ProviderCapabilities()


def test_explicit_detection_and_multiple_action_thresholds():
    policy = Policy(
        id="graded",
        version="1",
        detection_thresholds={"harassment": 0.4},
        rules=[
            Rule(id="warn", category="harassment", threshold=0.6, action="WARN"),
            Rule(id="block", category="harassment", threshold=0.9, action="BLOCK"),
        ],
    )
    result = ModGuard(provider=FakeProvider({"harassment": 0.5}), policy=policy).moderate_text("x")
    assert result.categories[0].detected
    assert result.decision.action == "ALLOW"
    with pytest.raises(ValidationError):
        Rule(id="bad", category="harassment", threshold=0.5, action="ERROR")


@pytest.mark.parametrize("timeout", [True, "30", float("inf"), -0.1, 0])
def test_invalid_timeouts(policy, timeout):
    with pytest.raises(ConfigurationError):
        ModGuard(provider=FakeProvider(), policy=policy, timeout=timeout)


def test_client_close_waits_for_admitted_calls(policy):
    started = threading.Event()
    release = threading.Event()

    class Blocking(FakeProvider):
        def classify_text(self, *args, **kwargs):
            started.set()
            assert release.wait(2)
            return super().classify_text(*args, **kwargs)

    provider = Blocking()
    guard = ModGuard(provider=provider, policy=policy, own_provider=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        prediction = executor.submit(guard.moderate_text, "x")
        assert started.wait(1)
        closing = executor.submit(guard.close)
        try:
            assert not closing.done()
            assert not provider.closed
        finally:
            release.set()
        assert prediction.result(2).status == "completed"
        closing.result(2)
    assert provider.closed
    with pytest.raises(ClosedError):
        guard.moderate_text("x")


async def test_batch_bounded_and_cancelled_workers_drained(policy):
    started = asyncio.Event()
    release = asyncio.Event()

    class Concurrent(FakeProvider):
        active = 0
        maximum = 0

        async def aclassify_text(self, *args, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            started.set()
            try:
                await release.wait()
                return await super().aclassify_text(*args, **kwargs)
            finally:
                self.active -= 1

    provider = Concurrent()
    async with AsyncModGuard(provider=provider, policy=policy, batch_concurrency=2) as guard:
        task = asyncio.create_task(guard.moderate_batch([TextInput(text="x")] * 100))
        await started.wait()
        await asyncio.sleep(0)
        assert provider.maximum == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.active == 0


async def test_provider_close_drains_queued_work():
    from moderationkit.types import InferenceOptions

    provider = ControlledLaya(capacity=2)
    first = asyncio.create_task(
        provider.aclassify_text(
            TextInput(text="x"), categories=CATEGORIES, options=InferenceOptions(request_id="first")
        )
    )
    assert await asyncio.to_thread(provider.started.wait, 1)
    second = asyncio.create_task(
        provider.aclassify_text(
            TextInput(text="y"),
            categories=CATEGORIES,
            options=InferenceOptions(request_id="second"),
        )
    )
    await asyncio.sleep(0)
    close = asyncio.create_task(provider.aclose())
    await asyncio.sleep(0)
    provider.release.set()
    assert (await first).coverage_complete
    assert (await second).coverage_complete
    await close


def test_sync_deadline_detects_late_provider(policy):
    class Slow(FakeProvider):
        def classify_text(self, *args, **kwargs):
            time.sleep(0.01)
            return super().classify_text(*args, **kwargs)

    result = ModGuard(provider=Slow(), policy=policy, timeout=0.001).moderate_text("x")
    assert result.errors[0].code == "timeout" and not result.decision.safe


def test_provider_must_explicitly_confirm_input_coverage(policy):
    from moderationkit.types import ClassificationResult

    class UnknownCoverage(FakeProvider):
        def classify_text(self, *args, **kwargs):
            raw = super().classify_text(*args, **kwargs)
            return ClassificationResult(categories=raw.categories, metadata=raw.metadata)

    result = ModGuard(provider=UnknownCoverage(), policy=policy).moderate_text("x")
    assert result.status == "incomplete" and not result.decision.safe
