import json

from moderationkit import ModGuard
from moderationkit.evaluation import DatasetRecord, evaluate_dataset
from moderationkit.policies.presets import community
from moderationkit.providers.laya import LayaConfig, LayaProvider

# A synthetic smoke dataset verifies the report flow, not production accuracy.
records = [
    DatasetRecord(
        id="greeting",
        text="Good morning",
        labels={
            "harassment": False,
            "hate_speech": False,
            "sexual_content": False,
        },
    )
]
with ModGuard(
    provider=LayaProvider(LayaConfig(model="english")),
    policy=community(),
    own_provider=True,
    timeout=600,
) as guard:
    print(json.dumps(evaluate_dataset(guard, records), indent=2))
