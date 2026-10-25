from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SoilEstimate:
    """A provider result must carry its source and confidence with its values."""

    values: dict[str, float | None]
    source: str
    confidence_label: str
    retrieved_at: str | None = None


class SoilDataProvider(Protocol):
    def estimate(self, latitude: float, longitude: float) -> SoilEstimate | None:
        """Return a sourced estimate, or None when the provider has no defensible result."""


class UnconfiguredRegionalSoilProvider:
    """Explicit no-data implementation; never fabricates soil values from coordinates."""

    status_message = "No trusted regional soil dataset is configured. Soil values are unavailable until a report is entered."

    def estimate(self, latitude: float, longitude: float) -> SoilEstimate | None:
        return None
