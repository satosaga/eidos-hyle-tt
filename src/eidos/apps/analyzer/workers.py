"""
eidos.apps.analyzer.workers -- Background QThread workers.

SimulationWorker runs a single re-simulation off the GUI thread;
AutoFitWorker runs core.calibrator.calibrate() (which itself fans out
across a ProcessPoolExecutor) off the GUI thread and relays progress;
SensitivityWorker runs sample_morris_sensitivity/sample_sobol_sensitivity
the same way, for TTAnalyzerWindow's inline, pre-Auto-Fit Sobol'/Morris
sensitivity bars.
"""

import concurrent.futures
import logging

from PySide6.QtCore import QThread, Signal

import core.calibrator as calibrator
from core.activity_parser import ActivityRecord
from eidos.apps.analyzer.models import Scenario, StrategyRecord, run_scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V. Background simulation worker
# ---------------------------------------------------------------------------

class SimulationWorker(QThread):
    """
    Run a single scenario re-simulation in a background thread.

    Signals:
        finished(SimTrace): Emitted when the simulation completes successfully.
        error(str):         Emitted if an exception is raised.
    """
    finished = Signal(object)   # SimTrace
    error = Signal(str)

    def __init__(
        self,
        strategy: StrategyRecord,
        scenario: Scenario,
        activity_raw: ActivityRecord | None,
        parent=None,
    ):
        super().__init__(parent)
        self._strategy = strategy
        self._scenario = scenario
        self._activity_raw = activity_raw

    def run(self):
        try:
            trace = run_scenario(self._strategy, self._scenario, self._activity_raw)
            logger.info(
                "SimTrace: label=%s finish=%.1fs x_traj len=%d w_traj len=%d w_min=%.0f w_max=%.0f",
                trace.label, trace.finish_time_s, len(trace.x_traj), len(trace.w_traj),
                trace.w_traj.min(), trace.w_traj.max(),
            )
            self.finished.emit(trace)
        except Exception as exc:
            logger.exception("SimulationWorker error")
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# VI. Plot canvas
# ---------------------------------------------------------------------------

class AutoFitWorker(QThread):
    """
    Run calibrator.calibrate() in a background thread.

    Unlike SimulationWorker, calibrate() itself fans out across
    ProcessPoolExecutor workers (see calibrator.calibrate_multistart)
    — this QThread's job is just to keep that off the GUI event loop,
    relay progress, and support cancellation via Qt's normal
    requestInterruption() mechanism (checked between trial completions
    inside calibrate_multistart via the should_cancel callback — see
    that function's docstring for why a cancelled run raises rather
    than silently returning the best-so-far).

    Signals:

    - ``progress(int, int, object)``: (n_done, n_total, best_mse_so_far);
      best_mse_so_far is None until the first trial completes.
    - ``finished(object)``: Emitted with the CalibrationResult on success.
    - ``error(str)``: Emitted if an exception (other than cancellation)
      is raised.
    - ``cancelled()``: Emitted if the run was stopped via
      requestInterruption().
    """
    progress = Signal(int, int, object)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        strategy: StrategyRecord,
        activity_raw: ActivityRecord,
        fixed_overrides: dict,
        free_keys: list[str],
        parent=None,
    ):
        super().__init__(parent)
        self._strategy = strategy
        self._activity_raw = activity_raw
        self._fixed_overrides = fixed_overrides
        self._free_keys = free_keys

    def run(self):
        try:
            result = calibrator.calibrate(
                course_distance_m=self._strategy.course_distance_m,
                simulator_key=self._strategy.simulator_spec.key,
                base_physics=self._strategy.physics_params,
                raw_physical=self._strategy.raw_physical,
                raw_physiological=self._strategy.raw_physiological,
                raw_run=self._strategy.raw_run,
                course_profile=self._strategy.course_profile,
                fixed_overrides=self._fixed_overrides,
                activity_raw=self._activity_raw,
                free_keys=self._free_keys,
                progress_callback=lambda done, total, best: self.progress.emit(done, total, best),
                should_cancel=self.isInterruptionRequested,
            )
            self.finished.emit(result)
        except calibrator.CalibrationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.exception("AutoFitWorker error")
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# VII. Background sensitivity-screening worker (Check Sensitivity)
# ---------------------------------------------------------------------------

class SensitivityWorker(QThread):
    """
    Run calibrator.sample_morris_sensitivity or sample_sobol_sensitivity
    in a background thread -- same reasoning as AutoFitWorker: both fan
    out across ProcessPoolExecutor workers internally (via
    core.calibrator._evaluate_parallel) and would otherwise block
    the GUI event loop.

    Unlike AutoFitWorker, this builds its own CalibrationInputs from
    scratch inside run() rather than delegating that to calibrate() --
    sample_morris_sensitivity/sample_sobol_sensitivity are standalone
    functions with no calibrate()-shaped entry point of their own.

    Supports cancellation via Qt's normal requestInterruption() -- same
    mechanism as AutoFitWorker, but checked at a different granularity:
    sample_morris_sensitivity/sample_sobol_sensitivity poll it roughly
    once per calibrator._SENSITIVITY_TARGET_CHUNK_S inside their shared
    _evaluate_parallel, not once per trial the way calibrate_multistart
    does (a single Sensitivity run has no natural per-trial boundary to
    check between).

    Signals:
    - finished(object): Emitted with a MorrisSensitivityTrials or
      SobolSensitivityTrials on success (whichever `method` asked for).
    - error(str): Emitted if an exception is raised.
    - cancelled(): Emitted if the run was stopped via
      requestInterruption().
    """
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        strategy: StrategyRecord,
        activity_raw: ActivityRecord,
        fixed_overrides: dict,
        free_keys: list[str],
        method: str,
        method_kwargs: dict,
        pool: concurrent.futures.ProcessPoolExecutor | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._strategy = strategy
        self._activity_raw = activity_raw
        self._fixed_overrides = fixed_overrides
        self._free_keys = free_keys
        self._method = method
        self._method_kwargs = method_kwargs
        # TTAnalyzerWindow's own persistent pool, reused across every
        # SensitivityWorker instance for the window's whole lifetime --
        # not created fresh per run (see calibrator._evaluate_parallel's
        # `pool` docstring for why a fresh pool per run is expensive AND,
        # under this app's "cancel the old run when a new one starts"
        # pattern, prone to piling up several full-width pools at once
        # if runs supersede each other faster than one can shut down).
        self._pool = pool

    def run(self):
        try:
            calib = calibrator.build_calibration_inputs(
                course_distance_m=self._strategy.course_distance_m,
                simulator_key=self._strategy.simulator_spec.key,
                base_physics=self._strategy.physics_params,
                raw_physical=self._strategy.raw_physical,
                raw_physiological=self._strategy.raw_physiological,
                raw_run=self._strategy.raw_run,
                course_profile=self._strategy.course_profile,
                fixed_overrides=self._fixed_overrides,
                activity_raw=self._activity_raw,
                free_keys=self._free_keys,
            )
            if self._method == "morris":
                result = calibrator.sample_morris_sensitivity(
                    calib, should_cancel=self.isInterruptionRequested,
                    pool=self._pool, **self._method_kwargs,
                )
            elif self._method == "sobol":
                result = calibrator.sample_sobol_sensitivity(
                    calib, should_cancel=self.isInterruptionRequested,
                    pool=self._pool, **self._method_kwargs,
                )
            else:
                raise ValueError(f"Unknown sensitivity method: {self._method!r}")
            self.finished.emit(result)
        except calibrator.CalibrationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.exception("SensitivityWorker error")
            self.error.emit(str(exc))
