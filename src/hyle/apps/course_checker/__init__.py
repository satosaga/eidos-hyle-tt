#!/usr/bin/env python3
"""
HYLE - hyle.apps.course_checker

Load a strategy JSON produced by eidos.apps.generator and open an
interactive 3D + graph viewer of the spline-interpolated course.

The strategy JSON stores the course profile as a zlib-compressed, pickled
Python dict under input.data.compressed_packet (see
`_decode_course_profile` below). This tool decodes it, projects
latitude/longitude to local planar meters, and hands a compact JSON
payload to hyle_course_checker.html, which renders:

  - a 3D polyline of the course (three.js), colorable by segment target
    power / curvature (kappa) / slope, with adjustable vertical
    exaggeration
  - three synchronized 2D charts vs. path distance: altitude, kappa,
    slope
  - a single distance slider that moves a marker on the 3D course and
    a vertical indicator line on all three charts simultaneously,
    while a readout panel shows the exact numeric values at that point

Usage
-----
::

    hyle-course-checker                          # drag-and-drop picker
    hyle-course-checker path/to/strategy_....json

Design note
-----------
This tool necessarily lives in the HYLE^TT logistics layer (it
unpickles EIDOS^TT's internal course-profile representation directly),
so unlike hyle.apps.fit2gpx_converter, there is no FIT-parsing
duplication concern here: HYLE owns this data end-to-end. The HTML
layer only ever receives already-decoded, plain-JSON data -- it never
touches the pickle.

Security note: `pickle.loads` executes arbitrary code embedded in the
pickle stream. This is safe here only because the input is trusted
output from the user's own eidos.apps.generator pipeline, never
third-party or untrusted data.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import pickle
import sys
import tempfile
import webbrowser
import zlib
from pathlib import Path
from typing import Any

from core.logging_setup import configure_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "course_checker.html"
START_MARKER = "/*__COURSE_DATA_JSON__*/"
END_MARKER = "/*__END_COURSE_DATA_JSON__*/"


def _decode_course_profile(data: dict[str, Any]) -> dict[str, list[float]]:
    """Decode the zlib+pickle compressed course profile packet.

    Returns a dict with keys such as latitude_list, longitude_list,
    altitude_list, kappa_list, slope_ratio_list, v_limit_list,
    distance_p_m_list (see eidos.apps.generator's output format).
    """
    packet_b64 = data["input"]["data"]["compressed_packet"]
    raw = base64.b64decode(packet_b64)
    decompressed = zlib.decompress(raw)
    profile = pickle.loads(decompressed)  # noqa: S301 -- trusted, own pipeline output
    return profile


def _latlon_to_local_xy_m(
    lat_list: list[float], lon_list: list[float]
) -> tuple[list[float], list[float]]:
    """Project lat/lon to local planar meters (equirectangular, centered
    on the first point). Good enough for course-length spans (<< 100km);
    not intended for geodesic-accurate use elsewhere.
    """
    R = 6371000.0
    lat0 = lat_list[0]
    lon0 = lon_list[0]
    lat0_rad = math.radians(lat0)
    cos_lat0 = math.cos(lat0_rad)

    x_m = [R * math.radians(lon - lon0) * cos_lat0 for lon in lon_list]
    y_m = [R * math.radians(lat - lat0) for lat in lat_list]
    return x_m, y_m


def _build_segments(strategy: dict[str, Any]) -> list[dict[str, float]]:
    """Convert target_length_list / target_power_list into [start_m, end_m,
    target_p_w] segment records by cumulative sum of segment lengths.
    """
    lengths = strategy["target_length_list"]
    powers = strategy["target_power_list"]
    segments = []
    cursor = 0.0
    for length_m, power_w in zip(lengths, powers):
        segments.append(
            {
                "start_m": cursor,
                "end_m": cursor + length_m,
                "target_p_w": power_w,
            }
        )
        cursor += length_m
    return segments


def build_course_data(strategy_json_path: Path) -> dict[str, Any]:
    with open(strategy_json_path, encoding="utf-8") as f:
        data = json.load(f)

    profile = _decode_course_profile(data)

    lat = profile["latitude_list"]
    lon = profile["longitude_list"]
    x_m, y_m = _latlon_to_local_xy_m(lat, lon)

    metadata = data["output"]["metadata"]
    kpis = data["output"]["results"]["kpis"]
    strategy = data["output"]["results"]["strategy"]

    segments = _build_segments(strategy)

    course_data = {
        "meta": {
            "source_filename": strategy_json_path.name,
            "n_seg": metadata["n_seg"],
            "seed": metadata["seed"],
            "total_time_s": kpis["total_time_s"],
        },
        "distance_m": profile["distance_p_m_list"],
        "lat_deg": lat,
        "lon_deg": lon,
        "x_m": x_m,
        "y_m": y_m,
        "z_m": profile["altitude_list"],
        "kappa": profile["kappa_list"],
        "slope": profile["slope_ratio_list"],
        "v_limit": profile["v_limit_list"],
        "segments": segments,
    }
    return course_data


def render_output_html(course_data: dict[str, Any], output_path: Path) -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if START_MARKER not in template or END_MARKER not in template:
        sys.exit(f"error: template markers not found in {TEMPLATE_PATH.name}")

    data_json = json.dumps(course_data, separators=(",", ":"))

    before, rest = template.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    rendered = before + START_MARKER + data_json + END_MARKER + after

    output_path.write_text(rendered, encoding="utf-8")


def _prompt_for_strategy_json() -> Path | None:
    """Get a strategy JSON path from the user when none was given on the CLI.

    Strategy JSONs end up scattered across strategies/*/, exports/*/*/, and
    ad-hoc copies elsewhere -- there's no single directory a file picker
    could sensibly default to, so this reuses the shared HYLE
    drag-and-drop prompt (see hyle.lib.common.prompt_via_drag_drop) rather than a
    directory-based picker.
    """
    from hyle.lib.common import prompt_via_drag_drop, window_title

    return prompt_via_drag_drop(
        message="Drop a strategy_*.json file here",
        win_title=window_title("Course Checker"),
    )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Open an interactive 3D/graph viewer for an EIDOS^TT course."
    )
    parser.add_argument(
        "strategy_json",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a strategy_*.json file. If omitted, a file picker opens.",
    )
    args = parser.parse_args()

    strategy_json = args.strategy_json or _prompt_for_strategy_json()
    if strategy_json is None:
        sys.exit("no file selected")

    if not strategy_json.exists():
        sys.exit(f"error: {strategy_json} not found")
    if not TEMPLATE_PATH.exists():
        sys.exit(f"error: {TEMPLATE_PATH.name} not found next to this script")

    course_data = build_course_data(strategy_json)

    # This is a throwaway viewer, not a deliverable artifact (unlike
    # eidos.apps.exporter's FIT/ZWO/PDF output) -- it doesn't belong
    # sitting next to the strategy JSON in strategies/*/ or exports/*/*/.
    temp_dir = Path(tempfile.mkdtemp(prefix="hyle_course_checker_"))
    output_path = temp_dir / f"{strategy_json.stem}__course_view.html"
    render_output_html(course_data, output_path)

    logger.info("wrote %s", output_path)
    webbrowser.open(output_path.resolve().as_uri(), new=1)


if __name__ == "__main__":
    main()