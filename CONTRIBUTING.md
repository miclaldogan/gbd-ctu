# Contributing

## Workflow

1. Fork the repository.
2. Create a feature branch from `main` or `develop` using a descriptive name such as `feature/scenario-split-audit`.
3. Install the project in editable mode and run `make lint` plus `make test` before pushing.
4. Open a pull request against the appropriate target branch.
5. Request review, address feedback with follow-up commits, and keep the branch rebased on the target branch.

## Pull Request Expectations

- Keep changes focused on a single research or infrastructure concern.
- Include a short description of the experimental or engineering motivation.
- Document any config, metric, or data-processing changes in the pull request body.
- Add or update tests whenever behavior changes.

## Code Style

- Follow `black` formatting and `flake8` linting.
- Prefer deterministic experiments where feasible by exposing seeds in config.
- Do not commit raw CTU-13 captures, derived private data, or large model checkpoints.
