"""Smriti — The neuroscience-inspired memory engine for AI agents.

>>> from smriti import SmritiClient
>>> client = SmritiClient()
>>> client.remember("The auth module uses JWT RS256")
'abc123-...'
>>> client.recall("authentication")
RecallResult(hits=[...], tokens_used=42)
"""

from smriti.client import SmritiClient
from smriti.types import Memory, RecallResult, RecallHit, Stats

__version__ = "0.1.0"
__all__ = ["SmritiClient", "Memory", "RecallResult", "RecallHit", "Stats"]
