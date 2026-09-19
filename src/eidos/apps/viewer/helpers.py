"""
eidos.apps.viewer.helpers -- Small standalone helper functions.

retrieve_nested_key and load_initial_records: no Qt dependency, no shared
state with the rest of the Viewer.
"""

import logging
from typing import Any, Dict, List

import eidos.lib.strategy_selector as rs

logger = logging.getLogger(__name__)


# II. Helper functions 
# ----------------------------------------------------------------------
def retrieve_nested_key(record: Dict[str, Any], keys: List[str]) -> Any:
    """Safely retrieve a nested key value from record; return None if any key is missing."""
    current_value = record
    try:
        for key in keys:
            current_value = current_value[key]
        return current_value
    except (KeyError, TypeError, IndexError):
        return None

def load_initial_records() -> List[Dict[str, Any]]:
    """Load and return the initial record list from the eidos.lib.strategy_selector module."""
    try:
        # rs is an alias for eidos.lib.strategy_selector
        records = rs.select_analysis_records() 
        if not records: return []
        return records
    except Exception as e:
        logger.error("Error: %s", e)
        return []
