"""
eidos.apps.manager.file_managers -- ConfigFileManager and FilterFileManager.

Plain file-I/O management for the configs/ and filters/ directories
(listing, loading, saving, duplicating, deleting JSON files) -- QObject
subclasses, but not GUI widgets themselves; used by the editor and
manager panes (see editor_panes.py, manager_panes.py) and by
window.py's TTManagerGUI, which owns the shared instances.
"""

import json
import logging
import os
import re
import shutil
import sys
from typing import Any, Dict, List

from PySide6.QtCore import QObject

from core.io_config import (
    BASE_CDA_YAW_TABLES_DIR,
    BASE_FILTERS_DIR,
    BASE_STRATEGIES_DIR,
    list_json_files_by_creation_time,
)
from core.io_config import (
    BASE_GPX_DIR as BASE_GPX_DATA_DIR,
)
from eidos.apps.manager.helpers import natural_sort_key

logger = logging.getLogger(__name__)


class ConfigFileManager(QObject):
    """Configuration file manager base class, managing files in a specified directory."""
    
    def __init__(self, target_dir: str, parent=None):
        """Initialize with the target config directory path."""
        super().__init__(parent)
        self.CONFIGS_DIR = target_dir
        self.TEMPLATES_DIR = os.path.join(self.CONFIGS_DIR, "templates")
        self.PYTHON_EXECUTABLE = sys.executable
    
    def _get_max_sequence_number(self, base_name: str) -> int:
        """Return the highest sequence number among files matching base_name_NN.json."""
        max_num = 0
        pattern = re.compile(r'^' + re.escape(base_name) + r'_(\d+)\.json$')
        try:
            for filename in os.listdir(self.CONFIGS_DIR):
                match = pattern.match(filename)
                if match:
                    max_num = max(max_num, int(match.group(1)))
        except FileNotFoundError:
            pass 
        return max_num

    def _generate_new_filename(self, base_name: str) -> str:
        """Return the next sequential filename (base_name_NN.json) for base_name."""
        next_num = self._get_max_sequence_number(base_name) + 1
        return f"{base_name}_{next_num:02d}.json"
        
    def get_available_config_files(self) -> List[str]:
        """Return .json config files in CONFIGS_DIR, oldest Added/Duplicated first (see list_json_files_by_creation_time)."""
        return list_json_files_by_creation_time(self.CONFIGS_DIR)

    def load_config_json(self, filename: str) -> Dict[str, Any]:
        """Load and return parsed JSON from filename in CONFIGS_DIR; raise FileNotFoundError if absent."""
        file_path = os.path.join(self.CONFIGS_DIR, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_config_json(self, filename: str, data: Dict[str, Any]):
        """Serialize data to JSON and write it to filename in CONFIGS_DIR."""
        file_path = os.path.join(self.CONFIGS_DIR, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def create_new_config_json(self, template_filename: str) -> str:
        """Copy template_filename from TEMPLATES_DIR to a new sequentially numbered config file; return its name."""
        base_name = os.path.splitext(template_filename)[0]
        template_path = os.path.join(self.TEMPLATES_DIR, template_filename)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        new_filename = self._generate_new_filename(base_name)
        new_path = os.path.join(self.CONFIGS_DIR, new_filename)
        # shutil.copy, NOT copy2: copy2 preserves the template's own mtime,
        # and on APFS a file written with an old mtime gets that SAME old
        # st_birthtime too (verified: copying a 3s-old template produced a
        # new file whose birthtime was the template's, not the actual copy
        # moment) -- silently defeating list_json_files_by_creation_time's
        # whole point (ordering by when THIS file was Added/Duplicated).
        # Plain copy never back-dates mtime, so birthtime comes out right.
        shutil.copy(template_path, new_path)
        return new_filename
    
    def get_template_config_files(self) -> List[str]:
        """Return a naturally-sorted list of .json template files in TEMPLATES_DIR."""
        try:
            files = [f for f in os.listdir(self.TEMPLATES_DIR) if f.endswith('.json') and not f.startswith('.')]
            return sorted(files, key=natural_sort_key)
        except FileNotFoundError:
            return []
    
    def delete_config_json(self, filename: str):
        """Delete filename from CONFIGS_DIR if it exists."""
        file_path = os.path.join(self.CONFIGS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    def duplicate_config_json(self, original_filename: str) -> str:
        """Copy original_filename to a new sequentially numbered file; return the new filename."""
        original_path = os.path.join(self.CONFIGS_DIR, original_filename)
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Source file for duplication not found: {original_path}")
        base_name_match = re.match(r'(.+?)_\d{2,}\.json$', original_filename)
        base_name = base_name_match.group(1) if base_name_match else os.path.splitext(original_filename)[0]
        new_filename = self._generate_new_filename(base_name)
        new_path = os.path.join(self.CONFIGS_DIR, new_filename)
        # shutil.copy, not copy2 -- see create_new_config_json's own
        # comment for why copy2's mtime preservation silently back-dates
        # st_birthtime here too.
        shutil.copy(original_path, new_path)
        return new_filename

    def get_available_gpx_files(self) -> List[str]:
        """Return a sorted list of .gpx files available in BASE_GPX_DATA_DIR."""
        if not os.path.exists(BASE_GPX_DATA_DIR): return []
        return sorted([f for f in os.listdir(BASE_GPX_DATA_DIR) if f.endswith('.gpx')], reverse=True)

    def get_available_cda_tables(self) -> List[str]:
        """Return a sorted list of .csv files available in BASE_CDA_YAW_TABLES_DIR."""
        if not os.path.exists(BASE_CDA_YAW_TABLES_DIR): return []
        return sorted([f for f in os.listdir(BASE_CDA_YAW_TABLES_DIR) if f.endswith('.csv')], reverse=True)

class FilterFileManager(ConfigFileManager):
    """File manager specifically for filter configurations."""
    def __init__(self, parent=None):
        """Initialize with BASE_FILTERS_DIR as the target directory."""
        super().__init__(target_dir=BASE_FILTERS_DIR, parent=parent)
        self.TEMPLATES_DIR = os.path.join(self.CONFIGS_DIR, "templates")
    
    def get_all_run_set_ids(self) -> List[str]:
        """
        Scan _index.parquet files under strategies/ and return all unique run_set_id values,
        sorted descending.
        """
        import glob

        import pandas as pd

        if not os.path.exists(BASE_STRATEGIES_DIR):
            return []

        ids = set()
        search_path = os.path.join(BASE_STRATEGIES_DIR, "*", "_index.parquet")
        index_files = glob.glob(search_path)

        for parquet_path in index_files:
            try:
                df = pd.read_parquet(parquet_path)
                if 'run_set_id' in df.columns:
                    valid_ids = df['run_set_id'].dropna().unique()
                    for rs_id in valid_ids:
                        ids.add(str(rs_id))
            except Exception as e:
                logger.warning("Failed to read index at %s: %s", parquet_path, e)
                continue

        return sorted(list(ids), reverse=True)
