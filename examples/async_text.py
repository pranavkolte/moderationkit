import asyncio

from moderationkit import AsyncModGuard
from moderationkit.policies.presets import community
from moderationkit.providers.laya import LayaConfig, LayaProvider


async def main():
    async with AsyncModGuard(
        provider=LayaProvider(LayaConfig(model="english")),
        policy=community(),
        own_provider=True,
        timeout=600,
    ) as guard:
        result = await guard.moderate_text("Have a pleasant morning.")
        print(result.model_dump_json(indent=2))


asyncio.run(main())
