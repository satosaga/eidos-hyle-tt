#!/usr/bin/env python3
"""
HYLE - hyle.apps.fit_combiner

Merge a directory of ``*.fit`` trial files into a single activity FIT
file, written alongside (in the parent of) the dropped directory --
same "output next to the input" convention as
hyle.apps.fit2gpx_converter, just one level up since the input here is
a directory of files rather than a single file.

The actual merge logic lives in core.fit_combiner.combine_fit_files,
shared with eidos.apps.trainer (which calls it in-process on its own
`to_combine/` staging directory at the end of a live session, no GUI
involved). This tool is the standalone counterpart for combining any
directory of FIT files by hand -- e.g. a hand-picked subset of past
trials -- not just a trainer session's own inbox.

Usage
-----
::

    hyle-fit-combiner                    # drag-and-drop picker
    hyle-fit-combiner path/to/trials_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.fit_combiner import combine_fit_files
from core.logging_setup import configure_logging


def _prompt_for_directory() -> Path | None:
    """Get a directory of FIT files from the user when none was given on
    the CLI, via the standard HYLE drag-and-drop prompt (see
    hyle.lib.common.prompt_via_drag_drop). Also accepts a directory drop,
    not just a single file -- the prompt itself has no file/directory
    restriction, it just returns whatever path was dropped."""
    from hyle.lib.common import prompt_via_drag_drop, window_title

    return prompt_via_drag_drop(
        message="Drop a directory of .fit files here",
        win_title=window_title("FIT Combiner"),
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Merge a directory of .fit trial files into a single activity FIT file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing .fit files to combine. If omitted, a drag-and-drop picker opens.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir or _prompt_for_directory()
    if input_dir is None:
        sys.exit("no directory selected")

    if not input_dir.is_dir():
        sys.exit(f"error: {input_dir} is not a directory")

    output_path = combine_fit_files(input_dir, input_dir.parent)
    if output_path is None:
        sys.exit(f"error: no .fit files found in {input_dir}")


if __name__ == "__main__":
    main()
