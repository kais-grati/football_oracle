"""Discord notification formatting and delivery.

One message is sent per fixture, listing every wallet that received a stake.
The message always states whether the decision used consensus or the
Bet365 fallback so the alert cannot silently mix the two.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send paper-bet embeds to a Discord webhook."""

    def __init__(
        self,
        webhook_url: str = "",
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ):
        self.webhook_url = (webhook_url or "").strip()
        self.session = session or requests.Session()
        self.timeout = float(timeout)
        self.enabled = bool(self.webhook_url)

    @staticmethod
    def build_bets_payload(
        fixture: dict[str, Any], wallet_bets: list[dict[str, Any]]
    ) -> dict[str, Any]:
        wallet_lines = "\n".join(
            f"**{item['wallet']}**: {item['stake']:.2f} "
            f"({item['stake_pct']:.1f}% of available)"
            for item in wallet_bets
        )
        source = str(fixture.get("odds_source", "consensus"))
        source_label = {
            "consensus": "Consensus",
            "bet365": "Bet365",
            "bet365_fallback": "Bet365 fallback",
            "none": "No price",
        }.get(source, source)
        fallback = source == "bet365_fallback"
        fields = [
            {"name": "Kickoff", "value": str(fixture["kickoff"]),
             "inline": True},
            {"name": "Price source", "value": source_label, "inline": True},
            {"name": "Decision lead", "value": (
                f"{fixture.get('actual_lead_minutes', 0):.0f} min"
            ), "inline": True},
            {"name": "Confidence", "value": (
                f"{fixture['confidence']:.1%}"
            ), "inline": True},
            {"name": "Models", "value": (
                f"logreg {fixture['p_logreg']:.1%} · "
                f"tree {fixture['p_tree']:.1%}"
            ), "inline": True},
            {"name": "Snapshot", "value": str(
                fixture.get("snapshot_at") or "unknown"
            ), "inline": True},
        ]
        if fallback and fixture.get("fallback_reason"):
            fields.append({
                "name": "Fallback reason",
                "value": str(fixture["fallback_reason"]),
                "inline": False,
            })
        fields.append({"name": "Wallets", "value": wallet_lines,
                       "inline": False})
        description = (
            f"{fixture['home_team']} vs {fixture['away_team']} · "
            f"{fixture['league']}"
        )
        if fallback:
            description += (
                "\n*Degraded: consensus unavailable; Bet365 features + "
                "execution.*"
            )
        return {
            "username": "Paper Trader",
            "allowed_mentions": {"parse": []},
            "embeds": [{
                "title": (
                    f"Paper bets — {str(fixture['selection']).title()} "
                    f"@ {fixture['odds_taken']:.2f}"
                ),
                "description": description,
                "color": 0xE0A800 if fallback else 0x2A78D6,
                "fields": fields,
                "footer": {"text": "Paper trading — no real money"},
            }],
        }

    def send_bets(
        self, fixture: dict[str, Any], wallet_bets: list[dict[str, Any]]
    ) -> bool:
        if not self.enabled or not wallet_bets:
            return False
        try:
            response = self.session.post(
                self.webhook_url,
                json=self.build_bets_payload(fixture, wallet_bets),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.warning("Discord notification failed: %s", exc)
            return False
