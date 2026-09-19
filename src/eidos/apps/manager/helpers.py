"""
eidos.apps.manager.helpers -- Small standalone helper functions.

GUI display-decimal-places computation lives in core.schema.
gui_decimals_from_field, shared with eidos.apps.analyzer's
PhysicsOverridePanel.
"""

import re
from pathlib import Path
from typing import List, Union

from eidos.apps.manager.constants import SCRIPT_DISPLAY_MAP


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """Generates a key for natural sort (e.g., file_10.json after file_2.json)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def format_script_name(filename: str) -> str:
    """Convert a script filename to a human-readable label using SCRIPT_DISPLAY_MAP or title-casing."""
    stem = Path(filename).stem
    # 1. Check display map first
    if stem in SCRIPT_DISPLAY_MAP:
        return SCRIPT_DISPLAY_MAP[stem]
    # 2. Fallback: capitalize words split by underscore
    words = stem.split('_')
    formatted_words = []
    for word in words:
        if word.lower() == 'tt':
            formatted_words.append('TT')
        else:
            formatted_words.append(word.capitalize())
    return ' '.join(formatted_words)
