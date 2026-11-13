import pytest

from moderationkit import Policy, Rule


@pytest.fixture
def policy():
    return Policy(
        id="test",
        version="1",
        rules=(
            Rule(id="hate", category="hate_speech", threshold=0.8, action="BLOCK", severity="high"),
            Rule(
                id="abuse", category="harassment", threshold=0.6, action="REVIEW", severity="medium"
            ),
        ),
    )
