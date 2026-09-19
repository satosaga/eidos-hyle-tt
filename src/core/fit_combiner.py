########################
# fit_combiner.py
########################
"""
Merge a directory of ``*.fit`` trial files into a single, spec-compliant
activity FIT file.

Shared by two callers with different UIs around the same operation:
eidos.apps.trainer calls combine_fit_files() in-process at the end of a
live session (its own trials, staged under a `to_combine/` inbox it
manages itself), and hyle.apps.fit_combiner is a standalone
drag-and-drop tool for combining any directory of FIT files the user
points it at. Neither caller is GUI/Qt-specific and this module carries
no dependency on either -- it's pure FIT-file logistics, not strategy
computation, which is why it lives in core rather than eidos/lib.

Files are ordered by each file's own first record timestamp, not by
filename. trainer.py's own trial filenames happen to sort correctly by
name too (a fixed-width timestamp prefix), but hyle.apps.fit_combiner
hands this function whatever directory the user drops, where filename
order isn't a safe proxy for recording order.
"""
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from fit_tool.fit_file import FitFile
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import FileType, Sport, SubSport

from core.logging_setup import log_subbanner

logger = logging.getLogger(__name__)


def combine_fit_files(input_dir: Path, output_dir: Path) -> Path | None:
    """
    Merge all ``*.fit`` files directly under input_dir into a single
    activity FIT file written to output_dir.

    Loads every file's record messages, then orders the files by each
    file's own first record timestamp (not filename) before
    concatenating -- filenames aren't a reliable ordering signal for a
    directory the user assembled by hand (see combine_fit_files's
    module docstring). Computes total elapsed time and distance, and
    writes a spec-compliant activity FIT file. Returns the output path,
    or None if input_dir contains no ``*.fit`` files (or none of them
    yielded any usable record messages).
    """
    targets = sorted(input_dir.glob("*.fit"))
    if not targets:
        return None

    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    # 1. FileIdMessage
    start_time_ms = round(datetime.now(timezone.utc).timestamp() * 1000)

    file_id_msg = FileIdMessage()
    file_id_msg.type = FileType.ACTIVITY
    file_id_msg.manufacturer = 255  # Development
    file_id_msg.product_name = "EIDOS^TT Combiner"
    file_id_msg.serial_number = 1
    file_id_msg.time_created = start_time_ms
    builder.add(file_id_msg)

    logger.info("Combining %d files using reference-compliant logic...", len(targets))

    # (start_timestamp, file_path, trial_records, trial_dist) per file with
    # at least one record; files with none contribute nothing (same as
    # before) and can't be given a start_timestamp, so they're dropped here
    # rather than sorted.
    loaded = []
    for file_path in targets:
        try:
            fit = FitFile.from_file(str(file_path))
            trial_records = [r.message for r in fit.records if isinstance(r.message, RecordMessage)]
            if not trial_records:
                logger.warning("Skipped (no records): %s", file_path.name)
                continue
            trial_dist = max((r.distance for r in trial_records if r.distance is not None), default=0.0)
            loaded.append((trial_records[0].timestamp, file_path, trial_records, trial_dist))
            logger.info("Loaded: %s", file_path.name)
        except Exception as e:
            logger.error("Error loading %s: %s", file_path, e)

    if not loaded:
        return None

    # Chronological order by each trial's own recorded start time, not the
    # order files happened to glob/sort in by name.
    loaded.sort(key=lambda item: item[0])

    # A quick back-to-back retry (eidos.apps.trainer) can leave less real
    # wall-clock time between two trials than the next trial's own
    # pre-/post-timed padding (see eidos.apps.trainer.save_to_fit's
    # LEAD_IN_PAD_S/TRAIL_OUT_PAD_S), so its records can start at or
    # before the previous trial's own last kept timestamp. Resolved by
    # shifting the WHOLE overlapping trial's timestamps later by a
    # constant -- its internal spacing (and so every duration/pace
    # computed from it) is unaffected -- just enough that it starts
    # RECORD_SPACING_MS after the previous trial's last kept record. No
    # record is ever dropped: this file already stitches together
    # separate attempts rather than one continuous recording, so a
    # shifted trial's absolute clock time (unused past this function) is
    # the right thing to trade away instead. A no-op whenever trials
    # don't actually overlap (the ordinary case).
    RECORD_SPACING_MS = 1000
    all_records = []
    total_dist = 0.0
    last_kept_ts = None
    for _, file_path, trial_records, trial_dist in loaded:
        if last_kept_ts is not None and trial_records[0].timestamp <= last_kept_ts:
            shift_ms = last_kept_ts + RECORD_SPACING_MS - trial_records[0].timestamp
            logger.info(
                "Shifting %s later by %.1fs to resolve timestamp overlap with the previous trial",
                file_path.name, shift_ms / 1000.0,
            )
            for r in trial_records:
                r.timestamp += shift_ms
        all_records.extend(trial_records)
        total_dist += trial_dist
        last_kept_ts = trial_records[-1].timestamp

    # 2. RecordMessages
    builder.add_all(all_records)

    # 3. Session and Activity messages
    first_rec = all_records[0]
    last_rec = all_records[-1]

    total_elapsed_sec = (last_rec.timestamp - first_rec.timestamp) / 1000.0

    session_msg = SessionMessage()
    session_msg.timestamp = last_rec.timestamp
    session_msg.start_time = first_rec.timestamp
    session_msg.total_elapsed_time = float(total_elapsed_sec)
    session_msg.total_timer_time = float(total_elapsed_sec)  # match elapsed for wall-clock accuracy
    session_msg.total_distance = float(total_dist)
    session_msg.sport = Sport.CYCLING
    session_msg.sub_sport = SubSport.VIRTUAL_ACTIVITY

    if hasattr(first_rec, 'position_lat') and first_rec.position_lat is not None:
        session_msg.start_position_lat = first_rec.position_lat
        session_msg.start_position_long = first_rec.position_long

    builder.add(session_msg)

    activity_msg = ActivityMessage()
    activity_msg.timestamp = last_rec.timestamp
    activity_msg.num_sessions = 1
    activity_msg.total_timer_time = float(total_elapsed_sec)
    builder.add(activity_msg)

    # 4. Save
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    earliest_file_name = loaded[0][1].name
    course_id = re.sub(r'_try\d+\.fit$', '', re.sub(r'^\d{8}_\d{6}_', '', earliest_file_name))

    output_path = output_dir / f"{now_str}_{course_id}_combined.fit"

    fit_file = builder.build()
    fit_file.to_file(str(output_path))

    log_subbanner(logger, f"SUCCESS: COMBINED (REF-MODE): {output_path}")

    return output_path
