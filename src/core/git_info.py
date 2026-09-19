#####################
# git_info.py
#####################
"""
Read this checkout's git commit/dirty state, and compare it against
what a strategy JSON recorded at generation time.

Background
----------
eidos.apps.generator stamps each strategy JSON's input.versions with
hand-maintained strings (core.data_manager.DATA_MANAGER_VERSION,
core.simulators.sim_kiritsubo.SIMULATOR_VERSION, eidos.lib.optimizer.OPTIMIZER_VERSION)
-- useful as human-readable milestone/variant labels, but they only
change when a developer remembers to bump them, so they can't be
trusted as a precise "did the code actually change" signal.

Several consumers -- eidos.apps.viewer (rebuilding a strategy's plot
data), eidos.apps.exporter (FIT/ZWO export), eidos.apps.trainer (live
"physics-identical" replay / Ghost Rider) -- resolve the *same*
simulator kernel the strategy was generated with (via
core.simulators.resolve_simulator(the strategy's own
input.settings.engine.simulator) -- e.g. core.simulators.sim_kiritsubo or
core.simulators.sim_stub, see core.simulators' own docstring for the
full registry) and re-run it against a strategy JSON's *stored* strategy
(target_power_list/target_length_list). But that's still whatever that
kernel *file* looks like *today*, not necessarily the version that
actually produced the strategy. If that physics logic has changed since
generation, the freshly re-simulated values (e.g. a Viewer graph) can
silently diverge from the strategy's own stored KPIs (total_time_s etc,
which are never recomputed -- only the supplementary re-simulated data
is at risk).

This module gives those three call sites a precise, low-maintenance way
to detect that risk: the running repo's actual git HEAD commit, plus
whether the working tree was dirty, both recorded alongside the
existing hand-maintained version strings (see
eidos.apps.generator.create_json_input_dict). eidos.apps.designer and
core.calibrator intentionally re-simulate with *today's* code (a
new strategy variant; a physics fit against real ride data) and are not
consumers of this check.

Scoping to the files that actually matter
--------------------------------------------------------------
The recorded git_commit is the whole repo's HEAD -- simplest to record,
and correct as raw information -- but comparing *that* directly against
today's HEAD is too blunt: any commit anywhere (a hyle.apps.*
tweak, a README edit) changes it, even though it has no bearing on
whether re-simulating a stored strategy gives the same answer. And,
concretely, eidos.lib.optimizer -- despite being one of the three
hand-maintained "versions" strings -- isn't even part of this risk:
it only runs while originally *searching for* a strategy, never while
re-simulating one that's already been decided (target_power_list /
target_length_list are just replayed through the resolved kernel).
So REPRODUCIBILITY_RELEVANT_PATHS below is deliberately narrower than
"the whole repo" and doesn't include it: every registered simulator's
own kernel file under core/simulators/ (each has its own hand-maintained
SIMULATOR_VERSION and PhysicsParams-equivalent builder -- see
core.simulators' docstring; a new registry entry's file must be added
here too, or edits to it would silently escape this check),
core/data_manager.py (build_course_profile, which decodes the strategy's
embedded course geometry), and core/schema.py (the PowerBlocks/
CourseProfile shapes every simulator's kernel passes data through -- a
field rename or reinterpretation there can change results without any
individual simulator file's own lines changing at all).
core/course_geometry.py's geometry preprocessing (course fitting, curvature,
speed limits, wind geometry) is deliberately excluded too, for the same
reason as the optimizer: build_course_profile decodes the *stored*
course geometry from the strategy JSON rather than recomputing it, so
editing that file has no bearing on re-simulating an already-decided
strategy either (only on eidos.apps.generator's initial generation run,
which stamps its own git_commit/git_dirty fresh each time regardless).
check_reproducibility() below diffs *only* these paths between the
stored and current commit (`git diff --quiet <a> <b> -- <paths>`) before
deciding whether to warn, rather than comparing the two hashes directly.

Escape hatch: none of this makes an old strategy exactly reproducible on
its own -- it only detects when it might not be. Auto-loading the old
simulator kernel file to re-simulate exactly wasn't chosen: doing it
safely would need core/schema.py and core/data_manager.py pulled from
the same old commit too (a partial reload risks a schema mismatch
between old and today's dataclasses), it re-triggers Numba's JIT
compile cost on every use, and even matching source doesn't guarantee
matching results if numpy/scipy/numba's own installed versions have
since changed -- all of which `git worktree` sidesteps for free. So:
to view a strategy with the exact code that produced it,
`git worktree add <dir> <commit>` checks out that commit into a
separate directory without disturbing the current working tree, so the
relevant app can be run from there once and the worktree removed
afterward.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Optional, Sequence

from core.io_config import PROJECT_ROOT

# Files whose content actually affects the outcome of re-simulating an
# already-decided strategy (see module docstring for why each is here,
# and why eidos.lib.optimizer.py deliberately is not).
REPRODUCIBILITY_RELEVANT_PATHS: tuple[str, ...] = (
    "src/core/simulators/sim_kiritsubo.py",
    "src/core/simulators/sim_stub.py",
    "src/core/data_manager.py",
    "src/core/schema.py",
)


def _run_git(*args: str) -> Optional[str]:
    """Run a git command in PROJECT_ROOT, returning stripped stdout, or
    None if git isn't available, this isn't a git checkout (e.g. a
    packaged install with no .git directory), or the command otherwise
    fails. Callers must treat None as "no information available", not
    as an error -- nothing here should ever be allowed to break strategy
    generation or viewing just because git metadata couldn't be read.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def get_git_commit_hash() -> Optional[str]:
    """Full HEAD commit hash of the running checkout, or None if
    unavailable. Cached: this can't change within a single process's
    lifetime."""
    return _run_git("rev-parse", "HEAD") or None


@lru_cache(maxsize=1)
def is_git_dirty() -> Optional[bool]:
    """Whether the working tree has ANY uncommitted changes anywhere in
    the repo (tracked-file modifications, staged changes, or untracked
    files), or None if that can't be determined -- mirrors
    get_git_commit_hash's None case. Cached for the same reason.

    General-purpose whole-repo check. Not used by is_relevant_code_dirty()/
    check_reproducibility() below, which need the narrower, path-scoped
    view -- see REPRODUCIBILITY_RELEVANT_PATHS and the module docstring.
    """
    porcelain = _run_git("status", "--porcelain")
    if porcelain is None:
        return None
    return bool(porcelain)


def _paths_differ(*revs: str, paths: Sequence[str]) -> Optional[bool]:
    """Whether `paths` differ across `revs`, via `git diff --quiet <revs>
    -- <paths>`. Pass one rev to compare the working tree (+ index)
    against it, or two to compare them directly. Returns None if this
    can't be determined (e.g. a rev is unreachable -- rewritten
    history, a commit from a different clone -- or git/the repo itself
    is unavailable); callers should treat None conservatively, as
    "might differ", not as "don't differ".
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", *revs, "--", *paths],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    return None  # e.g. 128: bad revision, not a git repo, etc.


def is_relevant_code_dirty() -> Optional[bool]:
    """Whether REPRODUCIBILITY_RELEVANT_PATHS have uncommitted changes
    right now, relative to HEAD -- the path-scoped counterpart to
    is_git_dirty(), and what eidos.apps.generator actually stamps as
    git_state.is_dirty (see its docstring for why: editing an unrelated
    file, e.g. hyle.apps.cpmodel_estimator, while generating a strategy
    shouldn't taint that strategy's recorded reproducibility state).

    Not cached, unlike is_git_dirty()/get_git_commit_hash(): this is
    called once per strategy at generation time (eidos.apps.generator)
    and again, in an entirely different later process, at comparison
    time (check_reproducibility()) -- there's no within-process reuse
    to cache.
    """
    return _paths_differ("HEAD", paths=REPRODUCIBILITY_RELEVANT_PATHS)


def check_reproducibility(
    stored_git_commit: Optional[str], stored_git_dirty: Optional[bool]
) -> Optional[str]:
    """Compare a strategy JSON's recorded git_state.commit_hash/
    is_dirty (read from input.git_state) against the code currently
    running, and return a one-line warning if a value re-simulated from
    this strategy's stored strategy might not match what was originally
    computed -- or None if there's nothing to warn about. This includes
    both "nothing to compare" (an old strategy predating this feature, or
    git being unavailable right now) and "the commit hash differs, but
    not in any of REPRODUCIBILITY_RELEVANT_PATHS" -- e.g. a commit that
    only touched hyle.apps.cpmodel_estimator or README.md has no bearing
    on this and is deliberately not flagged.
    """
    current_commit = get_git_commit_hash()

    if not stored_git_commit or current_commit is None:
        return None

    if stored_git_dirty:
        return (
            "This strategy was generated while "
            f"{', '.join(REPRODUCIBILITY_RELEVANT_PATHS)} had uncommitted "
            "changes, so its exact code state isn't pinned down by git "
            "alone -- values re-simulated from its stored strategy (e.g. a "
            "graph) are only approximate."
        )

    if stored_git_commit != current_commit:
        if _paths_differ(stored_git_commit, current_commit, paths=REPRODUCIBILITY_RELEVANT_PATHS) is not False:
            # True (confirmed to differ) or None (couldn't tell, e.g. the
            # stored commit is unreachable) -- warn either way. Only a
            # confirmed False (definitely unchanged) suppresses this.
            return (
                f"{', '.join(REPRODUCIBILITY_RELEVANT_PATHS)} may have "
                f"changed since this strategy was generated (commit "
                f"{stored_git_commit[:10]} then vs. {current_commit[:10]} "
                "now) -- values re-simulated from its stored strategy (e.g. "
                "a graph) may not exactly match what was originally "
                "computed. To see it with the exact original code: "
                f"git worktree add <dir> {stored_git_commit[:10]}"
            )

    if is_relevant_code_dirty():
        return (
            f"{', '.join(REPRODUCIBILITY_RELEVANT_PATHS)} has uncommitted "
            "changes right now -- values re-simulated from this strategy's "
            "stored strategy (e.g. a graph) may not exactly match what was "
            "originally computed."
        )

    return None
