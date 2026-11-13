from moderationkit import CategoryDefinition, ModGuard, Policy, Rule
from moderationkit.providers.laya import LayaConfig, LayaProvider

category = CategoryDefinition(
    id="custom.competitor_promotion",
    description="Unsolicited advertising that asks users to move to a competing service",
)
# Illustrative threshold: evaluate and tune against labelled deployment data.
policy = Policy(
    id="promotions",
    version="1.0",
    rules=[Rule(id="promotion_review", category=category.id, threshold=0.8, action="REVIEW")],
)
with ModGuard(
    provider=LayaProvider(LayaConfig(model="english")),
    policy=policy,
    categories=[category],
    own_provider=True,
    timeout=600,
) as guard:
    print(guard.moderate_text("Try our competitor's service today!").model_dump_json(indent=2))
