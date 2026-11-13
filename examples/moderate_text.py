"""Run: python examples/moderate_text.py (requires moderationkit[laya])."""

from moderationkit import ModGuard
from moderationkit.policies.presets import community
from moderationkit.providers.laya import LayaConfig, LayaProvider

with ModGuard(
    provider=LayaProvider(LayaConfig(model="english")),
    policy=community(),
    own_provider=True,
    timeout=600,  # Allows first-use model download; warmup separately in a service.
) as guard:
    print(guard.moderate_text("Hello, have a pleasant day.").model_dump_json(indent=2))
