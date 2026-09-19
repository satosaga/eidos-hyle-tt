#!/usr/bin/env bash
#
# Blocks a commit that stages changes to one of the hand-maintained
# code_version files (each simulator under core/simulators/, each
# optimizer under eidos/lib/optimizers/, plus core/data_manager.py)
# without the same commit also touching that file's own VERSION
# constant line -- the concrete fix for the pattern found 2026-08: none
# of the original three had ever been bumped in this repo's git
# history, despite the files themselves having been edited. A new
# SIMULATOR_REGISTRY/OPTIMIZER_REGISTRY entry's implementation file
# must be added to VERSIONED_PAIRS below too, or its edits would
# silently escape this check.
#
# Bump policy: bump for a change you'd want to be able to look back and
# identify later -- not for pure refactors/renames (e.g. a file move).
# When that's genuinely the case, skip this hook with:
#   git commit --no-verify
#
# This hook is deliberately blunt: it doesn't try to tell a "real" logic
# change apart from a trivial one (that judgment call, left to a human,
# was part of why bumps kept getting skipped in the first place) -- it
# just checks whether the VERSION line's own diff is part of the commit.
#
# Run via the pre-commit framework (see .pre-commit-config.yaml), not
# git's native core.hooksPath -- one-time setup:
#   pip install -e ".[dev]"
#   pre-commit install

set -euo pipefail

VERSIONED_PAIRS="
src/core/simulators/sim_kiritsubo.py:SIMULATOR_VERSION
src/core/simulators/sim_stub.py:SIMULATOR_VERSION
src/core/data_manager.py:DATA_MANAGER_VERSION
src/eidos/lib/optimizers/opt_tenchi.py:OPTIMIZER_VERSION
src/eidos/lib/optimizers/opt_stub.py:OPTIMIZER_VERSION
"

staged_files=$(git diff --cached --name-only)
blocked=0

for pair in $VERSIONED_PAIRS; do
    file="${pair%%:*}"
    version_const="${pair##*:}"

    if echo "$staged_files" | grep -qxF "$file"; then
        if ! git diff --cached -- "$file" | grep -qE "^[+-][[:space:]]*${version_const}[[:space:]]*="; then
            echo "error: $file is staged, but $version_const wasn't bumped in this commit." >&2
            echo "  Real change -> bump $version_const in $file." >&2
            echo "  Pure refactor/rename (no behavior change) -> git commit --no-verify" >&2
            blocked=1
        fi
    fi
done

exit "$blocked"
