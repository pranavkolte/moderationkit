from moderationkit import AsyncModGuard, ModGuard
from moderationkit.providers import AsyncTextProvider, TextProvider
from tests.fakes import FakeProvider


async def test_independent_provider_substitution(policy):
    provider = FakeProvider({"harassment": 0.9})
    assert isinstance(provider, TextProvider)
    assert isinstance(provider, AsyncTextProvider)
    with ModGuard(provider=provider, policy=policy) as client:
        sync = client.moderate_text("x")
    async with AsyncModGuard(provider=provider, policy=policy) as client:
        asynchronous = await client.moderate_text("x")
    assert sync.decision == asynchronous.decision
    assert sync.metadata.provider == "independent_fake"
