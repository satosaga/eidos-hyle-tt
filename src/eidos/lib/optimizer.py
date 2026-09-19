###################
# optimizer.py
###################
"""
Optimizer registry: maps a config's Engine.optimizer key to an
OptimizerSpec bundling one search-strategy implementation's entry
points (run/decode) and its own param_model (validated,
config-tunable parameters -- see EngineValidationModel.optimizer_params
below). Implementations themselves live under eidos.lib.optimizers/
(one module per registry entry) -- see that package's docstring. Mirrors
core.simulators' registry design; see that module's docstring for the
"why bundle these together" rationale (a future optimizer's decode
could differ arbitrarily from opt_tenchi's, the same way a future
simulator's params shape could differ from PhysicsParams).

Also owns EngineValidationModel/EngineSettings (Input.Settings.Engine:
simulator/optimizer/optimizer_params) -- not core.schema, even though
Engine.simulator selects a core.simulators.SIMULATOR_REGISTRY entry.
optimizer/optimizer_params are an EIDOS-only concept (HYLE never selects
an optimizer), and bundling them with a core-owned model would have
made core/schema.py import from here, which core/ may never do.
Simulator-key validation just reaches into
core.simulators.SIMULATOR_REGISTRY instead, which is the allowed
direction (eidos/ importing from core/).

Appendix: `run`'s return shape and OptimizerSpec's field list
---------------------------------------------------------------
- No `count_seeds` entry point: eidos.apps.generator.
  save_experiment_results computes num_seed_trials metadata by counting
  `len(results_for_Nseg)` -- the actual list `run()` returned -- rather
  than a separate, re-derived prediction. A future optimizer that
  decides its own seed count dynamically couldn't honestly implement a
  "count before running" entry point anyway; counting what actually
  happened is strictly more general.
- `run` returns `dict[int, list[SeedResult]]`, not a raw
  `scipy.optimize.OptimizeResult` per seed: generator.py never reads
  anything off a seed's result besides `.x` and `.success`, so a lean
  `SeedResult` (seed/x/success) carries exactly what crosses the
  registry boundary, nothing more.
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple, Type

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from core.schema import PowerBlocks
from core.simulators import SIMULATOR_REGISTRY
from eidos.lib.optimizers import SeedResult
from eidos.lib.optimizers.opt_stub import OPTIMIZER_VERSION as _OPT_STUB_VERSION
from eidos.lib.optimizers.opt_stub import OptStubParams as _OptStubParams
from eidos.lib.optimizers.opt_stub import decode as _opt_stub_decode
from eidos.lib.optimizers.opt_stub import sequential_optimize_Nseg as _opt_stub_run
from eidos.lib.optimizers.opt_tenchi import OPTIMIZER_VERSION as _OPT_TENCHI_VERSION
from eidos.lib.optimizers.opt_tenchi import TenchiParams as _TenchiParams
from eidos.lib.optimizers.opt_tenchi import decode as _opt_tenchi_decode
from eidos.lib.optimizers.opt_tenchi import sequential_optimize_Nseg as _opt_tenchi_run


class OptimizerSpec(NamedTuple):
    # See SimulatorSpec's `key` field docstring (core.simulators) for why
    # this is separate from `version` -- same reasoning here.
    key: str
    version: str
    # This optimizer's own Pydantic model for config-tunable parameters
    # (validated by EngineValidationModel's optimizer_params field below,
    # via resolve_optimizer_params). A future optimizer with
    # no tunables at all (e.g. "opt_stub") is free to declare an empty
    # model -- see eidos.lib.optimizers.opt_stub.OptStubParams.
    param_model: Type[BaseModel]
    # Top-level multi-n_seg multi-seed search entry point -- what
    # eidos.apps.generator calls to plan a strategy from scratch. Returns
    # one SeedResult per seed per n_seg -- see this module's docstring
    # appendix for why this isn't a `tuple[dict, dict]` with a separate
    # "best" aggregate.
    run: Callable[..., dict[int, list[SeedResult]]]
    # Decode a raw search-result vector (scipy OptimizeResult.x) into a
    # PowerBlocks -- what eidos.apps.generator.save_experiment_results
    # calls to turn each seed's raw result into a saveable strategy.
    # A future optimizer's own solution-vector encoding could differ
    # from opt_tenchi's (power/length-weight layout), so this must be
    # resolved from the same optimizer that actually produced the
    # result, not a hardcoded decode function. Takes the optimizer's own
    # validated param_model instance as its last argument (unused by
    # opt_tenchi/opt_stub's own pure-geometry decoding today, but part
    # of the shared contract for a future optimizer whose decoding
    # depends on its own tunables).
    decode: Callable[[np.ndarray, int, float, float, BaseModel], PowerBlocks]


OPTIMIZER_REGISTRY: dict[str, OptimizerSpec] = {
    "opt_tenchi": OptimizerSpec(
        key="opt_tenchi",
        version=_OPT_TENCHI_VERSION,
        param_model=_TenchiParams,
        run=_opt_tenchi_run,
        decode=_opt_tenchi_decode,
    ),
    "opt_stub": OptimizerSpec(
        key="opt_stub",
        version=_OPT_STUB_VERSION,
        param_model=_OptStubParams,
        run=_opt_stub_run,
        decode=_opt_stub_decode,
    ),
}


# Optimizer key for callers with no config JSON / Engine section to read
# (e.g. eidos.apps.manager.editor_panes.ConfigurationEditorPane._get_schema_map
# resolving a brand-new or malformed config's Engine.optimizer_params form).
# A named default, not "whatever's in the registry" -- mirrors core.
# simulators.DEFAULT_SIMULATOR_KEY's own reasoning: once a further
# OPTIMIZER_REGISTRY entry exists, iteration/insertion order must not
# silently decide which optimizer those callers get.
DEFAULT_OPTIMIZER_KEY = "opt_tenchi"


def resolve_optimizer(name: str) -> OptimizerSpec:
    """Look up an optimizer implementation by its OPTIMIZER_REGISTRY key."""
    try:
        return OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {sorted(OPTIMIZER_REGISTRY)}") from None


def resolve_optimizer_params(key: str, raw: dict) -> BaseModel:
    """Validate `raw` (a config JSON's Engine.optimizer_params dict) against
    the given optimizer's own param_model. Plain model_validate() -- no
    separate default-filling step here; whether an omitted key is
    tolerated is entirely up to that param_model's own field definitions.
    An optimizer with real tunable parameters (e.g. eidos.lib.optimizers.
    opt_tenchi.TenchiParams) deliberately makes every one of them required
    (Field(...), no Pydantic default -- see that class's own docstring),
    so raw omitting any of them raises here, the same "must be explicit"
    guarantee core.schema.PhysiologicalSettingsBase's cp/w_prime already
    have; a parameter-less optimizer (eidos.lib.optimizers.opt_stub.
    OptStubParams) validates an empty dict trivially either way. Called by
    EngineValidationModel's validate_and_default_optimizer_params below
    -- see that validator's docstring."""
    return resolve_optimizer(key).param_model.model_validate(raw)


# -----------------------------------------------
class EngineValidationModel(BaseModel):
    """Pydantic validation model for EngineSettings JSON input (simulator/optimizer registry selection)."""
    model_config = ConfigDict(extra="forbid")
    simulator: str = Field(..., title="Simulator", description="core.simulators.SIMULATOR_REGISTRY key selecting the physics kernel")
    optimizer: str = Field(..., title="Optimizer", description="eidos.lib.optimizer.OPTIMIZER_REGISTRY key selecting the search strategy")
    optimizer_params: dict = Field(default_factory=dict, title="Optimizer params", description="Optimizer-specific tuning parameters, validated against the selected optimizer's own OptimizerSpec.param_model")

    @field_validator("simulator")
    @classmethod
    def check_simulator_registered(cls, v: str) -> str:
        if v not in SIMULATOR_REGISTRY:
            raise ValueError(f"Unknown simulator '{v}'. Available: {sorted(SIMULATOR_REGISTRY)}")
        return v

    @field_validator("optimizer")
    @classmethod
    def check_optimizer_registered(cls, v: str) -> str:
        if v not in OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer '{v}'. Available: {sorted(OPTIMIZER_REGISTRY)}")
        return v

    @model_validator(mode="after")
    def validate_and_default_optimizer_params(self) -> "EngineValidationModel":
        """Validate optimizer_params against the selected optimizer's own
        OptimizerSpec.param_model -- runs after check_optimizer_registered
        (model_validators run after all field_validators), so self.optimizer
        is already a confirmed registered key by the time this runs.

        "...and_default" is accurate only for a param_model with no
        required fields (opt_stub.OptStubParams, an empty dict). An
        optimizer with real tunable parameters (opt_tenchi.TenchiParams)
        makes every one required by design, so a JSON input omitting any
        of them raises here rather than silently defaulting."""
        try:
            validated = resolve_optimizer_params(self.optimizer, self.optimizer_params)
        except ValidationError as e:
            raise ValueError(f"Invalid optimizer_params for '{self.optimizer}':\n{e}") from None
        self.optimizer_params = validated.model_dump()
        return self


@dataclass(frozen=True)
class EngineSettings:
    """
    Simulator/optimizer registry selection (Input.Settings.Engine).
    Constructed via from_dict() which runs Pydantic validation before instantiation.
    """
    simulator: str          # core.simulators.SIMULATOR_REGISTRY key
    optimizer: str          # OPTIMIZER_REGISTRY key
    optimizer_params: dict  # validated + defaulted against the selected optimizer's own param_model

    @classmethod
    def from_dict(cls, data: dict) -> "EngineSettings":
        """Validate input dict via Pydantic and return an EngineSettings instance."""
        try:
            valid_obj = EngineValidationModel.model_validate(data)
            return cls(**valid_obj.model_dump())
        except ValidationError as e:
            raise e
