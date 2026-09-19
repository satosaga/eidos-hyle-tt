"""
EIDOS^TT Analyzer

Post-race verification tool that overlays a simulated power strategy against
an actual activity FIT file and supports re-simulation with selectable
course profile, power source, and physical parameter overrides.

Three axes of variation (design concept, Three Factors):
    (1) Course profile  : GPX-derived  vs  FIT-derived altitude/geometry
    (2) Physics params  : Strategy settings  vs  user-overridden values
    (3) Power input     : Planned (strategy) vs  Activity (FIT-recorded)

Each combination drives a fresh call to core.simulators, producing a SimTrace
that is plotted alongside the raw FIT measurements on a shared distance axis.

Package layout:

    - models.py -- SimTrace / StrategyRecord / Scenario, strategy loading,
      the re-simulation engine (no Qt import)
    - workers.py -- SimulationWorker / AutoFitWorker (QThread)
    - canvas.py -- AnalysisCanvas (5-panel matplotlib plot)
    - minimap.py -- CourseMinimapWidget
    - widgets.py -- PhysicsOverridePanel and other analyzer-specific Qt widgets
    - window.py -- TTAnalyzerWindow, the QMainWindow tying it all together

This __init__.py keeps the CLI entry point (main()), so the console_scripts
target in pyproject.toml ("eidos.apps.analyzer:main") is unchanged by the split.
"""

import logging
import sys

from PySide6.QtWidgets import QApplication

from core.io_config import create_strategy_export_dir
from core.logging_setup import configure_logging
from eidos.apps.analyzer.models import load_strategy_record
from eidos.apps.analyzer.window import TTAnalyzerWindow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IX. Entry point
# ---------------------------------------------------------------------------

def _parse_cli_args() -> str:
    """Parse CLI arguments and resolve the strategy export directory.

    Same 4-positional-argument format as eidos.apps.trainer, so the two
    tools can be pointed at a strategy with the same invocation:

        eidos-analyzer <StrategySetDir> <RunID> <Nseg> <Seed>

    strategy_set_dir: e.g. "NisekoClassic2026" or "_20260301_170105"
    ts:               timestamp embedded in the strategy filename, e.g. "20260301_170105"
    """
    if len(sys.argv) < 5:
        print("Usage: eidos-analyzer <StrategySetDir> <RunID> <Nseg> <Seed>")
        sys.exit(1)

    strategy_set_dir, ts, n, s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    trial_id = f"N{n}_S{s}"
    return create_strategy_export_dir(strategy_set_dir, f"{ts}_{trial_id}")


def main():
    """Launch the Analyzer as a standalone application."""
    # configure_logging() (core.logging_setup) routes to sys.stdout
    # rather than logging's stderr default: when analyzer is launched as
    # a grandchild of eidos-manager (manager -> viewer -> analyzer, via
    # viewer/dialogs.py's _launch_script, which inherits stdio all the
    # way up with no redirection), the Manager's Execution Log colors
    # stdout/stderr differently. Leaving this on the stderr default would
    # paint every ordinary INFO log line red, indistinguishable from a
    # real error.
    configure_logging()

    record_dir = _parse_cli_args()
    try:
        strategy = load_strategy_record(record_dir)
    except Exception as exc:
        logger.error("Failed to load strategy: %s", exc)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = TTAnalyzerWindow(strategy)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
