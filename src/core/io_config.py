###################
# io_config.py
###################
import json
import os
from pathlib import Path

# --------------------------------------------------
# I. Directory name constants
# --------------------------------------------------
RESOURCES_DIR_NAME = "resources"
TEMP_DIR_NAME = "temp"

GPX_DIR_NAME = "gpx"
CDA_YAW_TABLES_DIR_NAME = "cda_yaw_tables"
CONFIGS_DIR_NAME = "configs"
FILTERS_DIR_NAME = "filters"
STRATEGIES_DIR_NAME = "strategies"
EXPORTS_DIR_NAME = "exports"
ACTIVITIES_DIR_NAME = "activities"

ZWO_DIR_NAME = "zwo"
FIT_DIR_NAME = "fit"

ENV_VAR_RESOURCES_DIR = "EIDOS_TT_RESOURCES_DIR"
ENV_VAR_STRATEGIES_DIR = "EIDOS_TT_STRATEGIES_DIR"

# --------------------------------------------------
# II. Base directory resolution
# --------------------------------------------------
# This file lives at <repo_root>/src/core/io_config.py -- "core" is the
# top-level package shared by both eidos/ and hyle/. Data directories
# (gpx/, strategies/, exports/, activities/, configs/, filters/,
# cda_yaw_tables/) intentionally still live at the repo root, not
# inside src/, so PROJECT_ROOT has to walk back up past
# core/ -> src/ to reach it. They're gathered under resources/, alongside
# hyle.apps.cpmodel_estimator's GC_activities/, which isn't part
# of this module's own set -- see that module's REPO_ROOT) rather than
# sitting loose at the repo root: some are git-tracked reference/sample
# data (cda_yaw_tables/, configs/templates/, filters/templates/), others
# are gitignored personal run data (gpx/, configs/, filters/ instances,
# activities/, exports/), and strategies/ is this module's own generated
# output. All of it can be redirected as one unit to synced/external
# storage via ENV_VAR_RESOURCES_DIR -- the point being to use the same
# courses/configs/filters/strategies from more than one machine, not just
# to escape disk space -- with ENV_VAR_STRATEGIES_DIR left as a finer
# override on top for the one subdirectory (strategies/) most likely to
# outgrow wherever the rest of resources/ lives. Since resources/ mixes
# git-tracked templates in with gitignored personal data, redirecting it
# via the env var doesn't carry the tracked files along automatically --
# whoever sets ENV_VAR_RESOURCES_DIR needs to have seeded that directory
# from a checkout's resources/ once first (a one-time manual copy, not
# something this module automates). (GC_activities/ in the hyle module is
# a different case again: it has no working local default at all, since
# it's each user's own pre-existing external GoldenCheetah history -- see
# that module's ENV_VAR_GC_ACTIVITIES_DIR.)
#
# temp/ is a separate, sibling directory at the repo root, not a
# resources/ subdirectory: it holds non-domain bookkeeping the app keeps
# for its own sake (a GUI's remembered window position, a tool's own
# performance cache) rather than EIDOS/HYLE domain material, so it doesn't
# belong under the same umbrella as gpx/results/etc. It's the same kind of
# "tool's own scratch space" as the already-root-level .mypy_cache/ or
# __pycache__/, just one we name and manage ourselves. Everything under it
# is disposable -- losing it means a GUI window resets to a default
# position, or a cache silently recomputes -- so, unlike resources/, it
# needs no fine-grained per-file git-tracking decisions: the whole
# directory is gitignored.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.environ.get(ENV_VAR_RESOURCES_DIR, os.path.join(PROJECT_ROOT, RESOURCES_DIR_NAME))
BASE_TEMP_DIR = os.path.join(PROJECT_ROOT, TEMP_DIR_NAME)

POWER_OFFSET_FILE = os.path.join(BASE_TEMP_DIR, "power_offset_memory.json")
DEFAULT_POWER_OFFSET_S = 6.0  # used only when POWER_OFFSET_FILE doesn't exist yet

BASE_GPX_DIR = os.path.join(RESOURCES_DIR, GPX_DIR_NAME)
BASE_CDA_YAW_TABLES_DIR = os.path.join(RESOURCES_DIR, CDA_YAW_TABLES_DIR_NAME)
BASE_CONFIGS_DIR = os.path.join(RESOURCES_DIR, CONFIGS_DIR_NAME)
BASE_FILTERS_DIR = os.path.join(RESOURCES_DIR, FILTERS_DIR_NAME)
BASE_STRATEGIES_DIR = os.environ.get(ENV_VAR_STRATEGIES_DIR, os.path.join(RESOURCES_DIR, STRATEGIES_DIR_NAME))  # finer override on top of ENV_VAR_RESOURCES_DIR, see module docstring
BASE_EXPORTS_DIR = os.path.join(RESOURCES_DIR, EXPORTS_DIR_NAME)
BASE_ACTIVITIES_DIR = os.path.join(RESOURCES_DIR, ACTIVITIES_DIR_NAME)

# --------------------------------------------------
# III. I/O utilities
# --------------------------------------------------

def create_strategy_export_dir(strategy_set_dir: str, trial_id: str | None = None) -> str:
    """
    Create and return the base export directory for a strategy.

    Structure: {BASE_EXPORTS_DIR}/{strategy_set_dir}/{trial_id}/
    Also creates 'fit' and 'zwo' subdirectories.
    """
    if trial_id:
        base_output_dir = os.path.join(BASE_EXPORTS_DIR, strategy_set_dir, trial_id)
    else:
        base_output_dir = os.path.join(BASE_EXPORTS_DIR, strategy_set_dir)

    os.makedirs(os.path.join(base_output_dir, FIT_DIR_NAME), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, ZWO_DIR_NAME), exist_ok=True)

    return base_output_dir

def find_strategy_json_path(strategy_set_dir: str, run_set_id: str, n_seg: int, seed: int) -> str:
    """
    Locate and return the path of a strategy JSON file matching the given parameters.

    Raises FileNotFoundError if no matching file is found in BASE_STRATEGIES_DIR/strategy_set_dir.
    """
    target_dir = Path(BASE_STRATEGIES_DIR) / strategy_set_dir
    filename_pattern = f"strategy_{run_set_id}_N{n_seg}_S{seed}.json"
    match_files = list(target_dir.glob(filename_pattern))

    if not match_files:
        raise FileNotFoundError(
            f"File not found: {filename_pattern} does not exist in {target_dir}"
        )

    return str(match_files[0])

def load_power_offset_s() -> float:
    """
    Return the last-remembered power-meter/altitude offset [s].

    Shared between eidos.apps.analyzer (the "Altitude lag" spinbox) and
    hyle.apps.fit2gpx_converter (the "Altitude timestamp offset" field) via
    POWER_OFFSET_FILE under temp/ -- same temp/-as-scratch-space convention
    eidos.apps.navigator already uses for its own HUD layout config (see
    that module's CONFIG_FILE). Neither tool is the sole owner of this
    value; whichever one the person last adjusted the offset in should have
    the other pick it up next time it opens.

    Falls back to DEFAULT_POWER_OFFSET_S (no memory file yet, or an
    unreadable/corrupt one) rather than raising -- this is a convenience
    default, not something either caller should have to guard against.
    """
    if os.path.exists(POWER_OFFSET_FILE):
        try:
            with open(POWER_OFFSET_FILE, "r") as f:
                return float(json.load(f)["power_offset_s"])
        except Exception:
            pass
    return DEFAULT_POWER_OFFSET_S

def save_power_offset_s(value: float) -> None:
    """
    Persist value as the shared power-meter/altitude offset memory, so the
    next eidos.apps.analyzer or hyle.apps.fit2gpx_converter session (see
    load_power_offset_s) starts from it instead of DEFAULT_POWER_OFFSET_S.

    Best-effort, like eidos.apps.navigator's own save_layout: temp/ is
    disposable scratch space, so a failed write here shouldn't interrupt
    whatever the caller was doing.
    """
    try:
        os.makedirs(BASE_TEMP_DIR, exist_ok=True)
        with open(POWER_OFFSET_FILE, "w") as f:
            json.dump({"power_offset_s": value}, f)
    except OSError:
        pass

def list_json_files_by_creation_time(directory: str) -> list[str]:
    """
    Return the .json filenames directly in directory, oldest-created first.

    Shared by eidos.apps.manager.file_managers.ConfigFileManager's own
    "Files in use" list (configs/ and filters/, the latter via
    FilterFileManager) and eidos.apps.generator's own config-loading loop,
    so a config/filter Added or Duplicated through the Manager GUI sorts
    -- and, for configs, generates -- in the same oldest-first order in
    both places, rather than each computing its own independent notion of
    "order."

    st_birthtime (file creation time), not st_mtime: editing an existing
    file's contents in place (see ConfigFileManager.save_config_json)
    overwrites the same inode without disturbing its birth time, while
    Add/Duplicate always write to a brand-new filename (see
    create_new_config_json/duplicate_config_json) with a fresh birth time
    -- so this reflects "when was this file Added or Duplicated," not
    "when was it last edited." macOS-only (st_birthtime is a BSD/APFS stat
    field), matching this project's own supported-platform scope.

    Returns bare filenames, not full paths -- matches os.listdir's own
    shape; callers already know their own directory.
    """
    files = [
        f for f in os.listdir(directory)
        if f.endswith(".json") and os.path.isfile(os.path.join(directory, f))
    ]
    return sorted(files, key=lambda f: os.stat(os.path.join(directory, f)).st_birthtime)