from typing import TypedDict, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    # --- Input (from /optimize request) ---
    disaster_type: str
    zones: list
    hospitals: list
    events: list
    coordinates: Optional[list]
    natural_language_query: Optional[str]

    # --- Database Agent outputs ---
    enriched_hospitals: list
    enriched_zones: list
    db_reasoning: str

    # --- Strategy Agent outputs ---
    context: dict
    allocation_results: list
    events_applied: list
    strategy_reasoning: str

    # --- Metadata ---
    fallback_used: bool
    error: Optional[str]
