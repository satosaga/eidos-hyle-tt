## What does this change, and why?

<!-- The "why" matters more than the "what" here -- see CONTRIBUTING.md's
"Commit messages and PRs". If this addresses an open issue, link it. -->

## Checklist

- [ ] `pytest`, `mypy src`, and `ruff check src tests` all pass (see `docs/RUNBOOK.md`)
- [ ] If this edits a registered simulator/optimizer/`data_manager.py`: its `*_VERSION` string is bumped in this same PR, or this is a pure refactor with no behavior change
- [ ] If this adds a new simulator/optimizer registry entry: it's registered, added to `scripts/check_code_version_bump.sh`'s `VERSIONED_PAIRS`, and (simulators only) to `core/git_info.py`'s `REPRODUCIBILITY_RELEVANT_PATHS` -- see `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md`
- [ ] Docs updated if this changes documented behavior (`docs/RUNBOOK.md`, `docs/ARCHITECTURE.md`, or the relevant module's own docstring)
- [ ] This PR is scoped to one change, not bundled with an unrelated fix
