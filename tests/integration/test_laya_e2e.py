import json
import os
from pathlib import Path

import pytest

from moderationkit import AsyncModGuard, ModGuard, Policy, Rule, TextInput
from moderationkit.evaluation import DatasetRecord, evaluate_dataset
from moderationkit.providers.laya import LayaConfig, LayaProvider

pytestmark = pytest.mark.integration
REVISION = "85cc74ddae6109077ebe2853a9b3077fb6606a78"


@pytest.fixture(scope="module")
def laya():
    pytest.importorskip("laya")
    provider = LayaProvider(LayaConfig(model="english", revision=REVISION))
    provider.warmup(timeout=600)
    yield provider
    provider.close()


@pytest.fixture
def policy():
    return Policy(
        id="integration",
        version="1",
        rules=[
            Rule(id="review", category="harassment", threshold=0.8, action="REVIEW"),
            Rule(id="politics", category="politics", threshold=0.9, action="WARN"),
        ],
    )


def test_real_sync_batch_evaluation(laya, policy):
    with ModGuard(provider=laya, policy=policy, timeout=120) as guard:
        result = guard.moderate_text("Hello, have a pleasant day.")
        assert result.status == "completed", result.model_dump_json()
        assert result.metadata.provider == "laya"
        assert result.metadata.revision == REVISION
        assert len(result.categories) == 2
        assert all(c.confidence is not None for c in result.categories)
        assert not result.metadata.evidence_available
        batch = guard.moderate_batch([TextInput(text="Hello"), TextInput(text="Good morning")])
        assert batch.failed == 0 and batch.processed == 2
        report = evaluate_dataset(
            guard,
            [
                DatasetRecord(
                    id="synthetic",
                    text="Good morning",
                    labels={"harassment": False, "politics": False},
                )
            ],
        )
        assert report["count"] == 1 and report["failures"] == 0
        output = os.environ.get("MODERATIONKIT_INTEGRATION_OUTPUT")
        if output:
            Path(output).write_text(
                json.dumps(
                    {"result": result.model_dump(mode="json"), "evaluation": report}, indent=2
                )
            )


async def test_real_async_parity(laya, policy):
    sync = ModGuard(provider=laya, policy=policy).moderate_text("Have a pleasant day.")
    async with AsyncModGuard(provider=laya, policy=policy) as guard:
        result = await guard.moderate_text("Have a pleasant day.")
        assert result.decision == sync.decision
        assert result.categories == sync.categories
        batch = await guard.moderate_batch([TextInput(text="Hello"), TextInput(text="Morning")])
        assert batch.failed == 0


def test_real_truncation_cannot_publish(laya, policy):
    result = ModGuard(provider=laya, policy=policy, timeout=120).moderate_text("Hello " * 1800)
    assert result.status == "incomplete"
    assert not result.decision.safe


def test_real_auto_route_english(policy):
    with LayaProvider(LayaConfig(revision=REVISION)) as provider:
        with ModGuard(provider=provider, policy=policy, timeout=120) as guard:
            result = guard.moderate_text("Have a pleasant morning", language="en")
            assert result.status == "completed"
            assert result.metadata.revision == REVISION
