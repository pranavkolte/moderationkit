import asyncio
import math

import pytest
from pydantic import ValidationError

from moderationkit import AsyncModGuard, CategoryDefinition, ModGuard, Policy, Rule, TextInput
from moderationkit.exceptions import (
    BatchModerationError,
    ClosedError,
    ConfigurationError,
    InputError,
    ModerationError,
)
from moderationkit.types import CategorySignal, Decision, ModerationResult, ProviderCapabilities
from tests.fakes import FakeProvider


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -0.1, 1.1, "0.8"])
def test_invalid_probabilities(value):
    with pytest.raises(ValidationError):
        CategorySignal(name="harassment", probability=value)


def test_flags_and_status_validate():
    with pytest.raises(ValidationError):
        Decision(action="ERROR", safe=True, can_publish=True, requires_review=False)
    with pytest.raises(ValidationError):
        TextInput(text="  ")


@pytest.mark.parametrize(
    "scores,action",
    [({}, "ALLOW"), ({"harassment": 0.6}, "REVIEW"), ({"hate_speech": 0.8}, "BLOCK")],
)
def test_threshold_precedence_and_roundtrip(policy, scores, action):
    with ModGuard(provider=FakeProvider(scores), policy=policy) as guard:
        result = guard.moderate_text("Example", item_id="a")
    assert result.decision.action == action
    assert result.decision.safe == (action == "ALLOW")
    assert result.decision.risk_score is None
    assert result.metadata.calibration_status == "unavailable"
    assert result == ModerationResult.model_validate_json(result.model_dump_json())


def test_all_matches_survive_and_policy_snapshot(policy):
    provider = FakeProvider({"hate_speech": 0.9, "harassment": 0.9})
    with ModGuard(provider=provider, policy=policy) as guard:
        copy = guard.policy
        copy.detection_thresholds["hate_speech"] = 0.1
        result = guard.moderate(TextInput(text="Example"))
    assert set(result.policy.matched_rules) == {"hate", "abuse"}
    assert len(result.violations) == 2
    assert result.categories[0].detection_threshold == 0.8


def test_warning_publishes():
    policy = Policy(
        id="warn",
        version="1",
        rules=[Rule(id="warn", category="harassment", threshold=0.5, action="WARN")],
    )
    result = ModGuard(provider=FakeProvider({"harassment": 0.8}), policy=policy).moderate_text("x")
    assert result.decision.safe and result.decision.can_publish
    assert not result.decision.requires_review


@pytest.mark.parametrize(
    "provider", [FakeProvider(partial=True), FakeProvider(coverage=False), FakeProvider(fail=True)]
)
def test_no_silent_safe(policy, provider):
    result = ModGuard(provider=provider, policy=policy).moderate_text("x")
    assert result.decision.action == "ERROR"
    assert not result.decision.safe
    assert result.errors


def test_known_match_preserved_on_partial(policy):
    result = ModGuard(
        provider=FakeProvider({"hate_speech": 0.9}, partial=True), policy=policy
    ).moderate_text("x")
    assert result.status == "incomplete"
    assert result.policy.matched_rules == ("hate",)
    assert result.violations


def test_raise_mode_and_programming_error(policy):
    with pytest.raises(ModerationError) as caught:
        ModGuard(
            provider=FakeProvider(fail=True), policy=policy, raise_on_error=True
        ).moderate_text("x")
    assert caught.value.result.decision.action == "ERROR"

    class Broken(FakeProvider):
        def classify_text(self, *args, **kwargs):
            raise AttributeError("programming error")

    with pytest.raises(AttributeError):
        ModGuard(provider=Broken(), policy=policy).moderate_text("x")


def test_custom_observation_and_unsupported():
    definition = CategoryDefinition(id="custom.promotion", description="Competitor promotions")
    policy = Policy(
        id="custom",
        version="1",
        rules=[Rule(id="promo", category=definition.id, threshold=0.7, action="REVIEW")],
        observation_categories=("politics",),
        detection_thresholds={"politics": 0.5},
    )
    provider = FakeProvider({definition.id: 0.9, "politics": 0.9})
    result = ModGuard(provider=provider, policy=policy, categories=[definition]).moderate_text("x")
    assert result.decision.action == "REVIEW"
    assert [c.detected for c in result.categories] == [True, True]
    assert len(result.violations) == 1
    provider.capabilities = ProviderCapabilities(custom_categories=False)
    result = ModGuard(provider=provider, policy=policy, categories=[definition]).moderate_text("x")
    assert result.status == "incomplete"


def test_optional_unavailable_does_not_block(policy):
    policy = policy.model_copy(
        update={"observation_categories": ("politics",), "detection_thresholds": {"politics": 0.5}}
    )
    provider = FakeProvider()
    provider.capabilities = ProviderCapabilities(
        supported_categories=frozenset({"hate_speech", "harassment"})
    )
    result = ModGuard(provider=provider, policy=policy).moderate_text("x")
    assert result.status == "completed"
    assert result.decision.safe
    assert result.categories[-1].detected is None


def test_invalid_configuration_and_inputs(policy):
    with pytest.raises(ConfigurationError):
        ModGuard(provider=FakeProvider(), policy=policy, timeout=float("nan"))
    with pytest.raises(ValidationError):
        Policy(id="bad", version="1", rules=[])
    with pytest.raises(ValidationError):
        Policy(
            id="bad",
            version="1",
            rules=[
                Rule(id="x", category="harassment", threshold=0.8, action="REVIEW"),
                Rule(id="y", category="harassment", threshold=0.9, action="BLOCK"),
            ],
        )
    with pytest.raises(ConfigurationError):
        ModGuard(
            provider=FakeProvider(),
            policy=policy,
            categories=[CategoryDefinition(id="harassment", description="override")],
        )
    guard = ModGuard(provider=FakeProvider(), policy=policy)
    with pytest.raises(InputError):
        guard.moderate_text("  ")
    with pytest.raises(InputError):
        guard.moderate_batch("wrong")


@pytest.mark.parametrize("owned", [True, False])
def test_lifecycle(policy, owned):
    provider = FakeProvider()
    with ModGuard(provider=provider, policy=policy, own_provider=owned) as guard:
        guard.moderate_text("x")
    guard.close()
    assert provider.closed == owned
    with pytest.raises(ClosedError):
        guard.moderate_text("x")


def test_batch_ids_failures_and_raise(policy):
    items = [TextInput(text="ok", item_id="same"), TextInput(text="fail", item_id="same"), None]
    for raising in (True, False):
        guard = ModGuard(provider=FakeProvider(), policy=policy, raise_on_error=raising)
        if raising:
            with pytest.raises(BatchModerationError) as caught:
                guard.moderate_batch(items)
            report = caught.value.result
        else:
            report = guard.moderate_batch(items)
        assert report.allowed == 1 and report.failed == 2
        assert [item.index for item in report.items] == [0, 1, 2]
        assert len({item.result.request_id for item in report.items}) == 3
        assert report.items[0].result.item_id == report.items[1].result.item_id == "same"
        assert guard.moderate_batch([]).processed == 0


async def test_async_parity_batches_and_ownership(policy):
    provider = FakeProvider({"harassment": 0.9})
    sync = ModGuard(provider=provider, policy=policy).moderate_text("x")
    async with AsyncModGuard(provider=provider, policy=policy, own_provider=True) as guard:
        result = await guard.moderate_text("x")
        assert result.decision == sync.decision
        assert result.categories == sync.categories
        report = await guard.moderate_batch([TextInput(text="x"), TextInput(text="fail"), None])
        assert report.reviewed == 1 and report.failed == 2
    assert provider.closed


async def test_async_timeout_and_cancellation(policy):
    class Slow(FakeProvider):
        async def aclassify_text(self, *args, **kwargs):
            await asyncio.Event().wait()

    async with AsyncModGuard(provider=Slow(), policy=policy, timeout=0.01) as guard:
        result = await guard.moderate_text("x")
        assert result.errors[0].code == "timeout"
        task = asyncio.create_task(guard.moderate_text("x", timeout=10))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
