##########################
# logging_setup.py
##########################
"""
(EIDOS/HYLE)^TT shared console logging configuration.

Lives in core/ (not eidos/lib/) because it's used by console_scripts
entry points in both eidos.apps.* (manager/generator/viewer/designer/
navigator/trainer/analyzer/exporter) and hyle.apps.* (course_checker/
cpmodel_estimator/fit2gpx_converter/fit_combiner/strategy_doctor) --
core/ is the package shared by both, same as io_config.py/schema.py/etc.
Every one of those entry points calls configure_logging() once, near the
top of its main(), so timestamp/level formatting and stdout routing are
identical everywhere.

For the eidos.apps.* side specifically, this also matters for apps
launched as subprocesses feeding eidos.apps.manager's Execution Log
viewer (a QPlainTextEdit that color-codes stdout/stderr, see
eidos.apps.manager.window.handle_process_output): stream=sys.stdout
(rather than logging's stderr default) matters there so ordinary INFO
output doesn't get mis-colored as an error. hyle.apps.* has no such
viewer -- its apps are standalone CLI tools or open a local browser tab
-- but the same consistent format/timestamps are still worth having
project-wide, and core.fit_combiner (shared by hyle.apps.fit_combiner
and, in-process, eidos.apps.trainer) is a concrete case of log output
that needs to behave correctly regardless of which side is calling it.

This module also defines the shared "banner" convention used for
phase/section boundaries in console output, used consistently across
generator.py, the optimizers, trainer.py/ant_receiver.py, and
core.fit_combiner:

  - log_banner()    -- major phase/section boundary: a bordered 3-line
                        block (row of "=", "=== title ===", row of "=").
  - log_subbanner()  -- minor phase marker within a section: a single
                        "--- title ---" line.

Both are emitted at the custom BANNER logging level (between INFO and
WARNING) rather than plain INFO, purely so eidos.apps.manager's log
viewer can recognize a banner line by its formatted "[BANNER]" levelname
tag and render it in a third, dedicated color -- distinct from both
normal stdout output and real stderr errors -- regardless of which app
or how many process/subprocess hops produced it (see e.g.
eidos.apps.viewer.dialogs._launch_script, which emits a log_banner()
right before launching Navigator/Trainer/Analyzer as a grandchild of
eidos-manager, specifically so that hand-off is visually obvious in the
Manager's log even though the grandchild process itself is otherwise
invisible to the Manager). On the hyle.apps.* side this BANNER level has
no special GUI meaning -- it's just a consistently-styled divider line.
"""
import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Custom level for banners: between INFO (20) and WARNING (30) so
# banners are always visible at the default INFO threshold but remain
# distinguishable from ordinary INFO records via %(levelname)s.
BANNER_LEVEL_NUM = 25
logging.addLevelName(BANNER_LEVEL_NUM, "BANNER")


def _banner(self, message, *args, **kwargs):
    """logging.Logger.banner(msg) -- emit at the custom BANNER level."""
    if self.isEnabledFor(BANNER_LEVEL_NUM):
        self._log(BANNER_LEVEL_NUM, message, args, **kwargs)


# Patched onto the Logger class (once, at import time) so any
# logging.getLogger(__name__) anywhere in the codebase gets .banner()
# for free, the same way it already has .info()/.warning()/etc.
logging.Logger.banner = _banner


# Third-party loggers that are known to be noisy at their own default
# level and would otherwise drown out (EIDOS/HYLE)^TT's own log output
# once force=True below routes everything through one shared handler.
# Each entry is a (logger name, level to clamp it to) pair.
#
#   "fit_tool" -- fit_tool.utils.logging creates logging.getLogger
#   ("fit_tool") and calls .setLevel(logging.INFO) at import time. Its
#   DataMessage.read_from_bytes() (fit_tool/data_message.py) then logs one
#   WARNING per field it doesn't recognize in a message -- routine noise
#   for FIT files with vendor-specific device_info fields fit_tool's
#   profile doesn't know about (the field is simply skipped, harmlessly),
#   not something surfaced by our own code, and it can fire dozens of
#   times per FIT file. Clamped to ERROR so genuine fit_tool errors still
#   come through. Affects every app that touches a FIT file, directly or
#   via core.activity_parser/core.fit_combiner (exporter, trainer,
#   navigator, analyzer, hyle.apps.fit2gpx_converter/fit_combiner, ...).
_THIRD_PARTY_LOGGER_LEVELS = {
    "fit_tool": logging.ERROR,
}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with (EIDOS/HYLE)^TT's standard console format.

    Call once, near the top of an app's main(). force=True so this wins
    even if some already-imported library called basicConfig() first
    (matching the behavior generator.py/exporter.py already relied on;
    fit_tool.utils.logging is one such library -- see
    _THIRD_PARTY_LOGGER_LEVELS above).
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        stream=sys.stdout,
        force=True,
    )
    for logger_name, clamp_level in _THIRD_PARTY_LOGGER_LEVELS.items():
        logging.getLogger(logger_name).setLevel(clamp_level)


def log_banner(logger: logging.Logger, title: str) -> None:
    """Major phase/section boundary: a bordered 3-line block.

    Use for run-level boundaries (start/end of execution, one banner per
    config set, etc.) -- the kind of thing that should be hard to miss
    even scrolling quickly through a long Execution Log.
    """
    border = "=" * 70  # fixed -- no caller has ever varied it
    logger.banner(border)
    logger.banner(f"=== {title} ===")
    logger.banner(border)


def log_subbanner(logger: logging.Logger, title: str) -> None:
    """Minor phase marker within a section: a single "--- title ---" line.

    Use for finer-grained boundaries inside a run (per-segment, per-seed,
    per-attempt markers, trainer session phase changes, etc.).
    """
    logger.banner(f"--- {title} ---")
