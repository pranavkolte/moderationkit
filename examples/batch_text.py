from moderationkit import ModGuard, TextInput
from moderationkit.policies.presets import community
from moderationkit.providers.laya import LayaConfig, LayaProvider

with ModGuard(
    provider=LayaProvider(LayaConfig(model="english")),
    policy=community(),
    own_provider=True,
    timeout=600,
) as guard:
    report = guard.moderate_batch(
        [
            TextInput(text="Hello!", item_id="comment-1"),
            TextInput(text="Have a good day", item_id="comment-2"),
        ]
    )
    print(report.model_dump_json(indent=2))
