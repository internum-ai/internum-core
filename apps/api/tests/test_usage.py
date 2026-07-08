from decimal import Decimal

from api.common.usage import InMemoryUsageTracker, UsageRecord


def test_usage_tracker_records_provider_usage_without_capability_imports() -> None:
    tracker = InMemoryUsageTracker()
    usage = UsageRecord(
        provider="openrouter",
        model="openai/gpt-5.2",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=Decimal("0.00012"),
        capability="document_parsing",
        consumer_id="internal",
        request_id="request-1",
    )

    tracker.record(usage)

    assert tracker.records == [usage]
