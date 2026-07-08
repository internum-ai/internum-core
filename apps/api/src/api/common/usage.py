from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Decimal | None
    capability: str
    consumer_id: str | None = None
    request_id: str | None = None


@runtime_checkable
class UsageTracker(Protocol):
    def record(self, usage: UsageRecord) -> None: ...


@dataclass
class InMemoryUsageTracker:
    records: list[UsageRecord] = field(default_factory=list)

    def record(self, usage: UsageRecord) -> None:
        self.records.append(usage)
