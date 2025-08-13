Contributing — SpectraMind V50

Thank you for considering a contribution! This repository enforces a mission‑grade workflow:

Development Setup
1.Install tooling:

poetry install --no-root
pipx run pre-commit install

2.Ensure all commands are available:

poetry run ruff --version
poetry run pytest -q

3.Validate the CLI:

poetry run python -m spectramind --help
poetry run python -m spectramind selftest --fast

Branching & Commits
•Use small, logically‑scoped branches: feat/, fix/, chore/, docs/.
•Commit style: Conventional Commits (feat:, fix:, docs:, chore:, etc.).
•Always link issues in PR descriptions.

Testing & Linting
•Pre-commit must pass locally before PRs.
•Unit tests: pytest -q.
•Keep configs in Hydra; do not hardcode values in code paths.

Documentation
•Update README.md and docs/ for user‑facing changes.
•Keep ARCHITECTURE.md synchronized with module I/O and contracts.

Security & Disclosure
•See SECURITY.md for reporting vulnerabilities.
•Do not commit secrets. Use .env, .envrc, or CI secrets.

Thank you!
