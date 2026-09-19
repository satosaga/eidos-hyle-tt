"""
eidos.apps.analyzer.models -- Data structures and re-simulation engine.

Qt-independent: SimTrace/StrategyRecord/Scenario plus the functions that
load a strategy export and run a single re-simulation. This half of the
Analyzer -- pure data/physics -- has no PySide6 import at all.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

import core.calibrator as calibrator
from core.activity_parser import ActivityRecord, build_zoh_power_blocks
from core.data_manager import (
    build_course_profile,
    unpack_input_data,
)
from core.physics_overrides import build_overridden_params
from core.schema import (
    CourseProfile,
    PhysicsParams,
    PowerBlocks,
    RunSettings,
    SimulationOutput,
)
from core.simulators import SimulatorSpec, resolve_simulator

# ---------------------------------------------------------------------------
# II. Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimTrace:
    """
    Trajectory output from a single simulator run, keyed on the distance axis.

    All arrays share the same length and are indexed by simulation step.
    The distance axis (x_traj) is the s_p road-distance axis used throughout
    EIDOS^TT — identical to SimulationOutput.x_traj.

    Attributes:
        label:          Human-readable scenario name shown in the legend.
        color:          Hex colour string for plot rendering.
        finish_time_s:  Total finish time [s].
        x_traj:         Cumulative road distance [m].
        t_traj:         Elapsed time [s].
        v_traj:         Velocity [m/s].
        p_traj:         Propulsive power [W].
        w_traj:         Remaining W' [J].
    """
    label: str
    color: str
    finish_time_s: float
    x_traj: np.ndarray
    t_traj: np.ndarray
    v_traj: np.ndarray
    p_traj: np.ndarray
    w_traj: np.ndarray

    @classmethod
    def from_simulation_output(
        cls,
        output: SimulationOutput,
        label: str,
        color: str,
    ) -> "SimTrace":
        """Construct a SimTrace from a SimulationOutput returned by the simulator."""
        return cls(
            label=label,
            color=color,
            finish_time_s=output.finish_time,
            x_traj=output.x_traj.copy(),
            t_traj=output.t_traj.copy(),
            v_traj=output.v_traj.copy(),
            p_traj=output.p_traj.copy(),
            w_traj=output.w_traj.copy(),
        )


@dataclass
class StrategyRecord:
    """
    Complete strategy data loaded from a strategy_*.json export directory.

    Mirrors the loading pattern of eidos.apps.viewer's load_full_record_data(),
    but is structured for the Analyzer's re-simulation needs rather than
    the Viewer's display pipeline.

    Attributes:
        record_dir:       Absolute path to the export sub-directory
                          (e.g. exports/_20260414_183136/20260415_015450_N11_S0/).
        json_path:        Path to the strategy_*.json file inside record_dir.
        run_set_id:       Unique identifier string from the JSON root.
        course_distance_m: Total course road distance [m].
        course_latlons:   List of (lat, lon) tuples for course matching.
        course_s_p:       Road-distance array [m] (s_p_fine).
        course_altitude_m: Altitude array [m] on the s_p axis.
        course_slope:     Slope array [rad] on the s_p axis.
        planned_power_blocks: PowerBlocks from the optimised strategy (IF100).
        physics_params:   PhysicsParams built from JSON settings (baseline).
        simulator_spec:   The SimulatorSpec (kernel + builder) resolved from
                          this strategy's own input.settings.engine.simulator --
                          re-simulation must use the same physics kernel that
                          actually produced this strategy, not whatever the
                          registry's current default happens to be.
        raw_physical:     Raw physical settings dict (for param override UI).
        raw_physiological: Raw physiological settings dict (for param override UI).
        raw_run:          Raw run settings dict.
        raw_engine:       This strategy's own input.settings.engine dict
                          verbatim ({"simulator": ..., "optimizer": ...,
                          "optimizer_params": ...} -- the first two are
                          registry keys, the third this strategy's own
                          validated+defaulted optimizer parameters). Carried
                          through unchanged so eidos.apps.analyzer.
                          _on_generate_config can write a config that
                          eidos.apps.generator.load_config_jsons will
                          actually accept (Engine has no default there).
        course_profile:   Full CourseProfile (GPX-derived).
    """
    record_dir: str
    json_path: str
    run_set_id: str
    course_distance_m: float
    course_latlons: list
    course_s_p: np.ndarray
    course_altitude_m: np.ndarray
    course_slope: np.ndarray
    planned_power_blocks: PowerBlocks
    physics_params: PhysicsParams
    simulator_spec: SimulatorSpec
    raw_physical: dict
    raw_physiological: dict
    raw_run: dict
    raw_engine: dict
    course_profile: CourseProfile


@dataclass
class Scenario:
    """
    A single re-simulation scenario with its result.

    Attributes:
        label:           Short name shown in the scenario list.
        color:           Hex colour for plots.
        power_source:    'planned' or 'actual' — which power series to feed.
                         'planned' is reserved for the auto-added "Strategy"
                         scenario (override-free, the Δt panel's reference
                         line); user-created scenarios from the "New
                         Scenario" panel are always 'actual'. Overriding
                         physics params on Planned power would re-play a
                         DE-optimized strategy under conditions it was never
                         optimized for — not a meaningful comparison, since
                         DE doesn't re-solve for the new conditions.
        physics_overrides: Dict of parameter overrides (keys match raw_physical/raw_physiological).
        auto_fit_keys:   Keys whose "Auto Fit" checkbox was checked when this
                         Rebuild was created (empty for a plain manual Rebuild).
                         Purely a UI memory -- physics_overrides already holds
                         the calibrated values regardless -- so that clicking
                         this Rebuild back in the list can restore which
                         checkboxes were on, not just the spinbox values.
        calibration_result: The CalibrationResult that produced
                         physics_overrides, if this Rebuild came from Auto
                         Fit (None for a plain manual Rebuild). Lets "Check
                         Auto Fit" always show THIS Rebuild's own
                         diagnostics rather than whichever Auto Fit run
                         happened to finish most recently.
        trace:           SimTrace result, None until run_scenario() is called.

    Course geometry is always GPX-derived (slope, kappa, heading all sourced
    from the same GPX-fitted profile) -- no FIT-altitude course_source
    option: FIT altitude runs ~8s behind FIT speed and needs a delay
    correction that already lives, as the single authoritative
    implementation, in hyle.apps.fit2gpx_converter, and substituting
    altitude alone while leaving GPX-derived kappa/heading in place would
    be a geometrically inconsistent mix regardless. To compare against a
    specific ride's actual course geometry, regenerate a GPX via
    hyle.apps.fit2gpx_converter from that FIT file and use it as the
    course normally.
    """
    label: str
    color: str
    power_source: Literal["planned", "actual"] = "planned"
    physics_overrides: dict = field(default_factory=dict)
    auto_fit_keys: list = field(default_factory=list)
    calibration_result: "calibrator.CalibrationResult | None" = None
    trace: SimTrace | None = None


# ---------------------------------------------------------------------------
# III. Strategy loading
# ---------------------------------------------------------------------------

def load_strategy_record(record_dir: str) -> StrategyRecord:
    """
    Load a strategy_*.json from an export sub-directory and return a StrategyRecord.

    Follows the same unpack → PhysicsParams → PowerBlocks pattern used by
    eidos.apps.viewer's load_full_record_data().

    Args:
        record_dir: Path to an export sub-directory containing strategy_*.json.

    Returns:
        A fully populated StrategyRecord.

    Raises:
        FileNotFoundError: If no strategy_*.json is found in record_dir.
        KeyError: If required fields are missing after JSON decompression.
    """
    # Locate the single strategy JSON
    candidates = [
        f for f in os.listdir(record_dir)
        if f.startswith("strategy_") and f.endswith(".json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No strategy_*.json found in {record_dir}")
    json_path = os.path.join(record_dir, candidates[0])

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data = unpack_input_data(data)

    p_s = data["input"]["settings"]["physical"]
    q_s = data["input"]["settings"]["physiological"]
    run_s = data["input"]["settings"]["run"]
    cp = data["input"]["data"]["course_profile"]
    strat = data["output"]["results"]["strategy"]

    simulator_spec = resolve_simulator(data["input"]["settings"]["engine"]["simulator"])

    # mu and brake_usability (PhysicalSettings fields, when this simulator
    # even has them) are used only during v_limit pre-computation (course
    # geometry build, already baked into course_profile.v_limit below) and
    # are not carried in PhysicsParams -- reconstructing PhysicalSettings/
    # PhysiologicalSettings/RunSettings here is solely to hand them to
    # simulator_spec's own builder, whatever shape it expects.
    # model_construct(), not model_validate() -- see
    # core.physics_overrides' module docstring for why an already-validated
    # strategy JSON's own settings are reconstructed without re-running
    # validators.
    physical_settings = simulator_spec.physical_param_model.model_construct(**p_s)
    physiological_settings = simulator_spec.physiological_param_model.model_construct(**q_s)
    run_settings = RunSettings(**run_s)

    # build_course_profile reuses this strategy's own stored v_limit and
    # neutral cos_phi/sin_phi placeholders; recompute_course_physics fills
    # in this simulator's own correct v_limit/cos_phi/sin_phi (a no-op for
    # a simulator with no braking-limit or wind model, e.g.
    # core.simulators.sim_stub) -- see both functions' own docstrings.
    course_profile = build_course_profile(cp)
    course_profile = simulator_spec.recompute_course_physics(course_profile, physical_settings)

    physics_params = simulator_spec.build_physics_params(
        physical_settings, physiological_settings, run_settings, course_profile,
    )

    planned_power_blocks = PowerBlocks(
        power=np.array(strat["target_power_list"]),
        length=np.array(strat["target_length_list"]),
    )

    lats = np.array(cp["latitude_list"])
    lons = np.array(cp["longitude_list"])
    course_latlons = list(zip(lats.tolist(), lons.tolist()))

    return StrategyRecord(
        record_dir=record_dir,
        json_path=json_path,
        run_set_id=data["run_set_id"],
        course_distance_m=float(course_profile.s_p_fine[-1]),
        course_latlons=course_latlons,
        course_s_p=course_profile.s_p_fine.copy(),
        course_altitude_m=course_profile.altitude.copy(),
        course_slope=course_profile.slope.copy(),
        planned_power_blocks=planned_power_blocks,
        physics_params=physics_params,
        simulator_spec=simulator_spec,
        raw_physical=p_s,
        raw_physiological=q_s,
        raw_run=run_s,
        raw_engine=data["input"]["settings"]["engine"],
        course_profile=course_profile,
    )


# ---------------------------------------------------------------------------
# IV. Re-simulation engine
# ---------------------------------------------------------------------------

def run_scenario(
    strategy: StrategyRecord,
    scenario: Scenario,
    activity_raw: ActivityRecord | None,
) -> SimTrace:
    """
    Execute one re-simulation and return the resulting SimTrace.

    power_source='planned' drives the physics from the optimized target-power
    strategy (use_sync_hook=False, is_target_power=True), the same clamped
    P_exerting path used by every other EIDOS^TT tool that replays a strategy.
    Reserved for the auto-added, override-free "Strategy" scenario — see
    Scenario's docstring for why user-created scenarios don't use it.

    power_source='actual' replays the FIT-recorded power directly
    (use_sync_hook=False, is_target_power=False), unclamped, via
    build_zoh_power_blocks(activity_raw, ...). PowerBlocks is built
    straight from the FIT file's own raw (variable-spaced, ~1s) samples —
    NOT from any display-resampled record — so block boundaries match the
    true recorded structure rather than an arbitrary display grid.

    Course geometry is always GPX-derived (see Scenario's docstring); there
    is no FIT-altitude course option, so no display-resampled ActivityRecord
    is needed here at all — only the raw one, and only for power replay.

    Both branches call the ordinary njit-compiled entry point directly;
    no sync_hook override or .py_func fallback is needed for either.

    Args:
        strategy: Loaded StrategyRecord (source of baseline params and course).
        scenario: Scenario specifying power source and param overrides.
        activity_raw: Raw (un-resampled) ActivityRecord — e.g.
                  ActivityCandidate.record; required for power_source='actual'.

    Returns:
        SimTrace with full trajectory arrays populated.

    Raises:
        ValueError: If power_source='actual' but no activity_raw is provided.
    """
    params = build_overridden_params(
        simulator_spec=strategy.simulator_spec,
        raw_physical=strategy.raw_physical,
        raw_physiological=strategy.raw_physiological,
        raw_run=strategy.raw_run,
        course_profile=strategy.course_profile,
        overrides=scenario.physics_overrides,
    )

    if scenario.power_source == "actual":
        if activity_raw is None:
            raise ValueError("power_source='actual' requires an ActivityRecord (raw).")

        power_blocks = build_zoh_power_blocks(
            activity_raw, target_distance_m=strategy.course_distance_m
        )

        output = strategy.simulator_spec.kernel(
            0.0, power_blocks, params, True, False, False
        )

    else:
        output = strategy.simulator_spec.kernel(
            0.0, strategy.planned_power_blocks, params, True, False, True
        )

    return SimTrace.from_simulation_output(output, scenario.label, scenario.color)
