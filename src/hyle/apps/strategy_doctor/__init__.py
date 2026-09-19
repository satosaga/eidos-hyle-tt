#!/usr/bin/env python3
"""
HYLE - hyle.apps.strategy_doctor

Scans resources/strategies/ for mismatches between each strategy set's
_index.parquet (the fast-load cache eidos.apps.generator/designer build and
eidos.lib.strategy_selector reads) and the strategy_*.json files actually on
disk, and optionally repairs them.

Background
----------
_index.parquet is a derived cache, never the source of truth -- the
strategy_*.json files are. Two failure modes have been observed in
practice:

  - missing index: eidos.apps.generator only writes _index.parquet after
    ALL strategy JSONs for a strategy set have been saved (see its "4. Save
    Parquet index" step); a process killed mid-run leaves JSONs on disk with
    no index at all. eidos.lib.strategy_selector.load_records_via_index
    silently skips any strategy_set_dir with no _index.parquet, so its
    strategies just don't appear in the Viewer -- it logs a warning
    pointing at this tool, but raises no error.
  - mismatch: eidos.apps.designer.update_index_parquet appends a row per
    new record; eidos.apps.viewer.window.execute_remove_design deletes a
    JSON file and then removes its index row as two separate steps. If
    either of those partially fails (or a JSON file is edited/deleted by
    hand), the index can end up with a "ghost" row pointing at a JSON
    file that no longer exists, or a JSON file that isn't indexed.

Recovery strategy: since the index is fully derivable from the JSON
files (same extract_flat_data()/ExperimentIndexModel machinery
eidos.apps.generator itself uses, now shared in core.pydantic_mapper),
repair for a given strategy_set_dir is always a full rebuild from its JSON
files -- not an incremental patch. This is simpler and self-healing: it
resolves "missing index", "ghost rows", and "unindexed JSON" in one pass
with no special-casing.

Usage
-----
::

    hyle-strategy-doctor          # scan resources/strategies/, print a report only
    hyle-strategy-doctor --fix    # also rebuild _index.parquet for any strategy_set_dir
                                  # with a problem (old index, if any, is kept
                                  # alongside as _index.parquet.bak)

No path arguments, matching the rest of the hyle.apps.* series'
"GUI-only, no path picker" convention (here: "no path picker" full stop)
-- the target is always the whole of BASE_STRATEGIES_DIR, since scanning it
is cheap and there's no good reason to leave part of it unchecked.

Design note
-----------
Like hyle.apps.course_checker (and unlike hyle.apps.cpmodel_estimator/
fit2gpx_converter), this tool has no browser UI: it's a diagnostic run
from a terminal, in the tradition of tools like `brew doctor` -- a
one-shot report to stdout, with a flag to act on it. A live map/chart
view has no role here.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.io_config import BASE_STRATEGIES_DIR
from core.logging_setup import configure_logging
from core.pydantic_mapper import ExperimentIndexModel, extract_flat_data

logger = logging.getLogger(__name__)

INDEX_FILENAME = "_index.parquet"
BACKUP_SUFFIX = ".bak"

# The three columns extract_flat_data()/ExperimentIndexModel always
# produce that together identify a single strategy record, matching the
# strategy_{run_set_id}_N{n_seg}_S{seed}.json filename convention (see
# core.io_config.find_strategy_json_path). Read from JSON content, not
# parsed out of the filename -- run_set_id itself contains an
# underscore ("YYYYmmdd_HHMMSS"), which would make filename parsing
# needlessly fragile for no benefit.
RunKey = tuple[str, int | None, int | None]


@dataclass
class StrategySetDirReport:
    strategy_set_dir: Path
    index_missing: bool = False
    corrupt_json: list[str] = field(default_factory=list)
    missing_from_index: list[RunKey] = field(default_factory=list)
    ghost_in_index: list[RunKey] = field(default_factory=list)
    json_count: int = 0

    @property
    def has_problem(self) -> bool:
        return bool(
            self.index_missing
            or self.corrupt_json
            or self.missing_from_index
            or self.ghost_in_index
        )


def get_active_strategy_set_directories() -> list[Path]:
    """List all non-hidden subdirectories under BASE_STRATEGIES_DIR."""
    base = Path(BASE_STRATEGIES_DIR)
    if not base.exists():
        return []
    return sorted(
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _record_key(data: dict[str, Any]) -> RunKey:
    """Extract the (run_set_id, n_seg, seed) identity key from a parsed
    strategy JSON's content -- the same identity find_strategy_json_path()
    reconstructs from a filename, but read from the trusted content
    instead of re-derived from the (possibly renamed) filename.

    Direct indexing, not .get(): eidos.apps.generator always writes
    run_set_id/output.metadata.n_seg/output.metadata.seed for every
    strategy it saves, so a JSON missing any of them is corrupt, not
    legitimately keyless -- _load_json_records routes that KeyError into
    its own corrupt list the same way a JSON parse failure is, rather
    than silently keying it (run_set_id, None, None), which would
    collide multiple different corrupt records onto the same dict key
    and drop all but the last one.
    """
    run_set_id = str(data["run_set_id"])
    metadata = data["output"]["metadata"]
    n_seg = metadata["n_seg"]
    seed = metadata["seed"]
    return (run_set_id, n_seg, seed)


def _load_json_records(strategy_set_dir: Path) -> tuple[dict[RunKey, dict[str, Any]], list[str]]:
    """Read every strategy_*.json in strategy_set_dir. Returns (key -> parsed data,
    list of filenames that failed to parse or lack their identity fields)."""
    records: dict[RunKey, dict[str, Any]] = {}
    corrupt: list[str] = []
    for json_path in sorted(strategy_set_dir.glob("strategy_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            key = _record_key(data)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            corrupt.append(f"{json_path.name} ({e})")
            continue
        records[key] = data
    return records, corrupt


def _load_index_keys(index_path: Path) -> set[RunKey]:
    """Read the (run_set_id, n_seg, seed) key of every row in an
    existing _index.parquet."""
    df = pd.read_parquet(index_path)
    keys: set[RunKey] = set()
    for _, row in df.iterrows():
        run_set_id = str(row.get("run_set_id", ""))
        n_seg = row.get("output.metadata.n_seg")
        seed = row.get("output.metadata.seed")
        n_seg = int(n_seg) if pd.notna(n_seg) else None
        seed = int(seed) if pd.notna(seed) else None
        keys.add((run_set_id, n_seg, seed))
    return keys


def inspect_strategy_set_dir(strategy_set_dir: Path) -> StrategySetDirReport:
    """Compare a strategy_set_dir's strategy_*.json files against its
    _index.parquet (if any) and report what doesn't match."""
    report = StrategySetDirReport(strategy_set_dir=strategy_set_dir)

    json_records, corrupt = _load_json_records(strategy_set_dir)
    report.corrupt_json = corrupt
    report.json_count = len(json_records)
    json_keys = set(json_records.keys())

    index_path = strategy_set_dir / INDEX_FILENAME
    if not index_path.exists():
        report.index_missing = True
        report.missing_from_index = sorted(json_keys)
        return report

    try:
        index_keys = _load_index_keys(index_path)
    except Exception as e:
        # An unreadable/corrupt parquet file behaves like a missing one:
        # nothing in it can be trusted, so treat every JSON record as
        # unindexed and let --fix rebuild it from scratch.
        logger.warning("failed to read %s: %s", index_path, e)
        report.index_missing = True
        report.missing_from_index = sorted(json_keys)
        return report

    report.missing_from_index = sorted(json_keys - index_keys)
    report.ghost_in_index = sorted(index_keys - json_keys)
    return report


def rebuild_index(strategy_set_dir: Path) -> tuple[int, list[str]]:
    """Rebuild strategy_set_dir's _index.parquet from its strategy_*.json files
    (JSON is the source of truth). Any existing index is preserved as
    _index.parquet.bak (overwriting a previous .bak) before being
    replaced. Returns (records written, filenames skipped because their
    content didn't match the expected schema -- e.g. an older strategy
    JSON missing a field TYPE_MAP now expects; such a file still counts
    toward json_count/detection above, but can't be turned into an index
    row, so it's reported and left out of the rebuilt index rather than
    aborting the whole strategy_set_dir's rebuild).

    If the strategy_set_dir has no strategy JSONs at all, returns (0, [])
    and writes nothing, leaving any existing index/backup as-is for the
    user to look at by hand.
    """
    index_records = []
    skipped: list[str] = []
    for json_path in sorted(strategy_set_dir.glob("strategy_*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # already reported as corrupt during inspection
        try:
            index_records.append(
                ExperimentIndexModel(**extract_flat_data(data)).model_dump()
            )
        except Exception as e:
            skipped.append(f"{json_path.name} ({e.__class__.__name__})")

    if not index_records:
        return 0, skipped

    index_df = pd.DataFrame(index_records)

    index_path = strategy_set_dir / INDEX_FILENAME
    if index_path.exists():
        backup_path = strategy_set_dir / (INDEX_FILENAME + BACKUP_SUFFIX)
        shutil.copy2(index_path, backup_path)

    index_df.to_parquet(index_path, index=False)
    return len(index_records), skipped


def _format_key(key: RunKey) -> str:
    run_set_id, n_seg, seed = key
    return f"run_set_id={run_set_id} n_seg={n_seg} seed={seed}"


def print_report(report: StrategySetDirReport) -> None:
    name = report.strategy_set_dir.name
    if report.index_missing:
        logger.warning("[%s] index MISSING (%d strategy JSON file(s) found)", name, report.json_count)
    elif report.missing_from_index or report.ghost_in_index:
        logger.warning(
            "[%s] MISMATCH: %d missing from index, %d ghost entrie(s) in index",
            name, len(report.missing_from_index), len(report.ghost_in_index),
        )
        for key in report.missing_from_index:
            logger.warning("    missing from index: %s", _format_key(key))
        for key in report.ghost_in_index:
            logger.warning("    ghost in index (no matching JSON): %s", _format_key(key))

    for filename in report.corrupt_json:
        logger.warning("[%s] corrupt JSON, skipped: %s", name, filename)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Scan resources/strategies/ for _index.parquet / strategy JSON "
            "mismatches and optionally repair them."
        )
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Rebuild _index.parquet (from JSON, the source of truth) for "
        "every strategy_set_dir with a problem. Without this flag, only reports.",
    )
    args = parser.parse_args()

    strategy_set_dirs = get_active_strategy_set_directories()
    if not strategy_set_dirs:
        logger.info("No strategy-set directories found under %s", BASE_STRATEGIES_DIR)
        return

    problem_dirs = []
    for strategy_set_dir in strategy_set_dirs:
        report = inspect_strategy_set_dir(strategy_set_dir)
        if report.has_problem:
            problem_dirs.append(report)
            print_report(report)

    if not problem_dirs:
        logger.info("OK: %d strategy-set director(y/ies) scanned, no problems found.", len(strategy_set_dirs))
        return

    logger.info(
        "%d of %d strategy-set director(y/ies) have problems.",
        len(problem_dirs), len(strategy_set_dirs),
    )

    if not args.fix:
        logger.info("Run with --fix to rebuild the affected indexes from JSON.")
        return

    for report in problem_dirs:
        if report.json_count == 0:
            logger.info(
                "[%s] skipped: no strategy JSON files present "
                "(nothing to rebuild from) -- left as-is for manual review.",
                report.strategy_set_dir.name,
            )
            continue
        n, skipped = rebuild_index(report.strategy_set_dir)
        logger.info("[%s] rebuilt %s (%d record(s))", report.strategy_set_dir.name, INDEX_FILENAME, n)
        for filename in skipped:
            logger.info("[%s]   skipped (schema mismatch): %s", report.strategy_set_dir.name, filename)


if __name__ == "__main__":
    main()
