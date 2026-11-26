# Contributing

Thanks for taking a look at Football Oracle. This is a research and
paper-trading project; changes should keep it reproducible and should never
introduce real-money execution.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` for local provider runs. Do not commit `.env` or
any API key, webhook URL, database, model artifact, or downloaded dataset.

## Checks

```bash
python -m pytest -q tests
```

The suite is offline: it uses fake clocks and providers plus saved API
payloads. Keep it that way. If you add a provider fixture, remove account
identifiers and live credentials first.

## Guidelines

- Keep the production boundary clear: `papertrader/` is the scheduler, the
  root research modules are experiments, and `reports/` is generated output.
- Add or update a test for behavior changes in the scheduler, ledger,
  providers, sizing, or feature pipeline.
- Do not weaken the decision safeguards: no bets after kickoff, no bets inside
  the safety cut-off, no bets without a complete executable price, and no
  duplicate wallet/match rows.
- Document new environment variables in `.env.example` and `README.md`.
- Third-party data and API terms still apply. Do not add data you do not have
  permission to redistribute.

## Reporting issues

Open an issue with a minimal reproduction. For security-sensitive reports, use
the process in `SECURITY.md` instead of a public issue.
