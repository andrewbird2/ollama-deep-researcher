import operator
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SummaryState:
    """The state of the research assistant."""
    research_topic: str
    search_query: Optional[str] = None
    web_research_results: List[str] = field(default_factory=list)
    sources_gathered: List[str] = field(default_factory=list)
    research_loop_count: int = 0
    running_summary: Optional[str] = None

@dataclass
class SummaryStateInput:
    """The input state for the research assistant."""
    research_topic: str

@dataclass
class SummaryStateOutput:
    """The output state for the research assistant."""
    running_summary: str
