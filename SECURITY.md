# Security

This is a paper-trading research project. It never handles real money and it
does not need brokerage credentials.

## Secrets

- API keys and Discord webhook URLs belong in `.env`, which is gitignored.
- `.env.example` contains placeholders only.
- Never commit a `.env` file, provider key, webhook URL, database, model
  artifact, or snapshot archive.
- If a key is exposed, revoke it at the provider and rotate the local value.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository, or
contact the repository owner privately. Do not open a public issue with
credentials or exploitable details.

## Scope

Useful reports include accidental credential exposure, unsafe provider
handling, SQLite/data-integrity problems, and logic that could place a bet
outside the documented decision window. Reports about model performance are
welcome as normal issues, not security reports.
