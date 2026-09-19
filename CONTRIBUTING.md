# Contributing to (EIDOS/HYLE)^TT

This is a solo-developed research/engineering tool, not a project with
a large existing contributor base. Contributions are welcome, but read
this first so effort doesn't get spent on something unlikely to land.

## What's welcome

- Bug reports, especially with a reproducible config/GPX pair.
- Documentation fixes — including "this doesn't match the code
  anymore," which this project takes seriously (see `docs/RUNBOOK.md`
  and `docs/ARCHITECTURE.md` for what's meant to stay accurate).
- A new simulator or optimizer registry entry (see "Adding a new
  simulator or optimizer" below) — the registry design exists
  specifically to make this kind of contribution low-risk to add
  without touching the existing entries.
- Small, scoped fixes to an existing app.

## What to discuss first

Anything that would change the core design — the physics/physiology
model, the optimizer's course-agnostic philosophy (see the README's
"What EIDOS^TT Is"), the EIDOS/HYLE package split, or the
reproducibility-tracking mechanism — open an issue to discuss before
writing code. `docs/ARCHITECTURE.md` documents the design rules
currently in effect and the reasoning behind them; changes that
contradict it need that reasoning addressed, not just overridden.

## Setting up a dev environment

See `docs/RUNBOOK.md`'s "First-time setup" section — in short:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

This enables five pre-commit hooks: ruff, a check that a registered
simulator/optimizer/`data_manager.py` edit bumps that file's own
`*_VERSION` string, a check that a new registry entry got added to
every list that needs to know about it, a check that `docs/*.md`
file-path references stay valid while `src/` never references a
`docs/*.md` file at all, and a check that `mypy src` reports only
already-triaged errors (see `docs/ARCHITECTURE.md`'s "mypy adoption").
The first three are explained in
`docs/RUNBOOK.md` (`## Linting with ruff`, `## A git commit is blocked
by "wasn't bumped in this commit"`, and the registry-consistency check
under `## First-time setup (fresh clone, or hooks not yet enabled)`) —
read the version-bump one before
your first commit that touches a simulator/optimizer file, since it
will otherwise be a surprise. That section documents a `--no-verify`
escape hatch for pure refactors with no behavior change; note that
`--no-verify` is a git-level flag that skips all five hooks at once,
not just the version-bump check, so only reach for it when nothing
else staged needs any of the other four either.

## Before opening a PR

```bash
pytest
mypy src
pip install -e ".[lint]"   # ruff/interrogate aren't part of [dev]
ruff check src tests
```

See `docs/RUNBOOK.md`'s "Running tests / type-checking" and "Linting
with ruff" sections for what's expected to be clean vs. known,
documented noise (`docs/ARCHITECTURE.md`'s "mypy adoption" section
lists the five specific categories this covers — PySide6/matplotlib
stub gaps, flattened old-style Qt enum access, a registry design
limitation, and a stdlib monkeypatch — anything outside those is
real). `pre-commit run --all-files` runs
all five hooks on demand without actually
committing, if you want to check before you're ready to commit.

## Adding a new simulator or optimizer

See `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md` for the full walkthrough
— it covers the `SimulatorSpec`/`OptimizerSpec` shape, a copy-paste
starting template (`sim_stub.py`/`opt_stub.py`), the naming convention
(codenames, not version numbers), and the checklist of everything that
needs to stay in sync when a new entry is added.

## Commit messages and PRs

Explain *why*, not just *what* — a commit message that only restates
the diff isn't useful later. Keep a PR scoped to one change; a PR that
mixes an unrelated fix in with the main change is harder to review and
harder to revert independently if something's wrong.

## License

By contributing, you agree your contribution is licensed under this
project's license (GNU GPL v3.0 or later — see [LICENSE](LICENSE)).
