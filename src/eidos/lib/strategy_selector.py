#########################
# strategy_selector.py
#########################
import fnmatch
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.io_config import BASE_FILTERS_DIR, BASE_STRATEGIES_DIR

logger = logging.getLogger(__name__)

# --------------------------------------------------
# II. General utilities
# --------------------------------------------------
def extract_nested_value(record: dict, key_string: str) -> Optional[Any]:
    """
    Retrieve a value from a nested dict using a dot-notation key string.
    Returns None if any key is missing or the value is NaN.
    """
    keys = key_string.split('.')
    current_value = record

    try:
        for key in keys:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                return None
        if current_value is None or (isinstance(current_value, (float, int)) and np.isnan(current_value)):
            return None
        return current_value
    except Exception:
        return None


# --------------------------------------------------
# III. Data loading
# --------------------------------------------------
def load_filter_jsons() -> List[Dict[str, Any]]:
    """
    Load all ``*.json`` files from BASE_FILTERS_DIR and return them as a list of filter condition dicts.
    """
    filter_conditions_list: List[Dict[str, Any]] = []
    filters_dir = BASE_FILTERS_DIR

    if not os.path.exists(filters_dir):
        logger.warning("Filters directory '%s' not found. Returning empty filter list.", filters_dir)
        return filter_conditions_list

    for filename in os.listdir(filters_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(filters_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    filter_data = json.load(f)
                    if isinstance(filter_data, dict):
                        filter_conditions_list.append(filter_data)
                    else:
                        logger.warning(
                            "Filter file %s is not a single dictionary (Type: %s). Skipping.",
                            filename, type(filter_data).__name__,
                        )
            except Exception as e:
                logger.warning("Error occurred while loading filter file %s: %s", file_path, e)

    return filter_conditions_list

# --------------------------------------------------
# IV. DataFrame filter helper
# --------------------------------------------------
def apply_filters_to_dataframe(df: pd.DataFrame, filter_list: List[Dict]) -> pd.DataFrame:
    """
    Apply a list of filter condition dicts to a DataFrame.
    Conditions within each dict are AND-combined; dicts in the list are OR-combined.

    Supported condition formats per key:
      - list of length 2: range [p, q) for float, [N, M] for int
      - list of other length: membership test (isin)
      - dict with operators: {"<": val}, {">": val}, {"==": val}, {"!=": val}
      - str with glob chars: fnmatch pattern
      - other: exact equality
    """
    if not filter_list or df.empty:
        return df

    mask = pd.Series([False] * len(df), index=df.index)

    for condition_dict in filter_list:
        current_mask = pd.Series([True] * len(df), index=df.index)
        for key, cond in condition_dict.items():
            if key not in df.columns:
                current_mask &= False
                continue

            if isinstance(cond, list):
                if len(cond) == 2:
                    if pd.api.types.is_float_dtype(df[key]):
                        current_mask &= (df[key] >= cond[0]) & (df[key] < cond[1])   # half-open [p, q)
                    else:
                        current_mask &= (df[key] >= cond[0]) & (df[key] <= cond[1])  # closed [N, M]
                else:
                    current_mask &= df[key].isin(cond)  # membership test

            elif isinstance(cond, dict):
                for op, val in cond.items():
                    if op == "<":    current_mask &= (df[key] < val)
                    elif op == ">":  current_mask &= (df[key] > val)
                    elif op == "==": current_mask &= (df[key] == val)
                    elif op == "!=": current_mask &= (df[key] != val)

            elif isinstance(cond, str):
                if any(c in cond for c in "*?[]"):
                    regex = fnmatch.translate(cond)
                    current_mask &= df[key].astype(str).str.fullmatch(regex, na=False)
                else:
                    current_mask &= (df[key].astype(str) == cond)

            else:
                current_mask &= (df[key] == cond)

        mask |= current_mask

    return df[mask]

# --------------------------------------------------
# V. Index-based fast loader
# --------------------------------------------------
def get_active_strategy_set_directories() -> List[str]:
    """List all non-hidden subdirectories under the strategies directory."""
    if not os.path.exists(BASE_STRATEGIES_DIR):
        return []
    return [os.path.join(BASE_STRATEGIES_DIR, d) for d in os.listdir(BASE_STRATEGIES_DIR)
            if os.path.isdir(os.path.join(BASE_STRATEGIES_DIR, d)) and not d.startswith(".")]

def load_records_via_index(filter_list: List[Dict]) -> List[Dict]:
    """
    Build lightweight strategy records from _index.parquet files instead of loading
    individual JSON files. Reduces load time from tens of seconds to milliseconds.

    Records are marked with is_index_only=True for on-demand full loading by the Viewer.
    """
    all_matched_light_records = []
    target_dirs = get_active_strategy_set_directories()

    for target_dir in target_dirs:
        index_path = os.path.join(target_dir, "_index.parquet")
        if not os.path.exists(index_path):
            logger.warning(
                "No _index.parquet in %s; its results won't appear "
                "until you run 'hyle-strategy-doctor --fix' to rebuild it.",
                target_dir,
            )
            continue

        try:
            df = pd.read_parquet(index_path)
        except Exception as e:
            logger.warning("Failed to read index at %s: %s", target_dir, e)
            continue

        # A missing required column means this index's schema is stale/
        # corrupt -- fail loud for the whole directory here (via the same
        # warn-and-skip idiom as the two checks above) rather than let each
        # row's .get(..., 0.0) silently fabricate a "0.00s finish" record
        # that would otherwise sit unflagged among 100+ real ones in Viewer.
        required_cols = {
            "run_set_id", "output.metadata.n_seg", "output.metadata.seed",
            "output.results.kpis.total_time_s",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.warning(
                "Index at %s is missing column(s) %s (stale schema?); run "
                "'hyle-strategy-doctor --fix' to rebuild it.",
                target_dir, sorted(missing_cols),
            )
            continue

        strategy_set_dir = os.path.basename(target_dir)
        df['strategy_set_dir'] = strategy_set_dir

        matched_df = apply_filters_to_dataframe(df, filter_list)

        for _, row in matched_df.iterrows():
            run_set_id = str(row["run_set_id"])
            n_seg = int(row["output.metadata.n_seg"])
            seed = int(row["output.metadata.seed"])
            time = float(row["output.results.kpis.total_time_s"])
            display_name = f"N{n_seg}_S{seed}({time:7.2f}s) [strategy_set_dir:{strategy_set_dir}, run_set_id:{run_set_id}]"

            light_rec = {
                "display_name": display_name,
                "strategy_set_dir": strategy_set_dir,
                "run_set_id": run_set_id,
                "N_seg_file": n_seg,
                "Seed_file": seed,
                "time": time,
                "is_index_only": True,  # flag for on-demand loading
                "output": {
                    "results": {
                        "kpis": {
                            "total_time_s": time
                        }
                    }
                }
            }
            all_matched_light_records.append(light_rec)

    return all_matched_light_records

# --------------------------------------------------
# VI. Entry point
# --------------------------------------------------
def select_analysis_records() -> List[Dict[str, Any]]:
    """Load filter conditions and return matching lightweight records via the Parquet index."""
    filter_conditions_list = load_filter_jsons()
    analysis_group = load_records_via_index(filter_conditions_list)

    if not analysis_group:
        return []

    return analysis_group