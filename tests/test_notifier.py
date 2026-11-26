from __future__ import annotations

import requests

from papertrader.notifier import DiscordNotifier

from .fakes import FakeResponse, FakeSession


def fixture(source="consensus", fallback_reason=None):
    return {
        "home_team": "Real Madrid",
        "away_team": "Deportivo Alavés",
        "league": "La Liga",
        "kickoff": "2026-11-28 21:00",
        "selection": "home",
        "odds_taken": 1.60,
        "confidence": 0.72,
        "p_logreg": 0.68,
        "p_tree": 0.70,
        "odds_source": source,
        "fallback_reason": fallback_reason,
        "snapshot_at": "2026-11-28T19:55:00+00:00",
        "actual_lead_minutes": 30.0,
    }


def wallet_bets():
    return [
        {"wallet": "Conservative", "stake": 2.0, "stake_pct": 2.0},
        {"wallet": "Medium", "stake": 5.0, "stake_pct": 5.0},
    ]


def test_payload_lists_wallets_and_consensus_source():
    payload = DiscordNotifier.build_bets_payload(fixture(), wallet_bets())
    embed = payload["embeds"][0]
    assert "Real Madrid vs Deportivo Alavés" in embed["description"]
    assert "Consensus" in str(embed["fields"])
    assert "Conservative" in str(embed["fields"])
    assert "Medium" in str(embed["fields"])
    assert "Degraded" not in embed["description"]


def test_payload_marks_fallback_and_reason():
    payload = DiscordNotifier.build_bets_payload(
        fixture("bet365_fallback", "stale_snapshot:3000s"), wallet_bets()
    )
    embed = payload["embeds"][0]
    assert "Bet365 fallback" in str(embed["fields"])
    assert "Degraded" in embed["description"]
    assert "stale_snapshot:3000s" in str(embed["fields"])
    assert embed["color"] == 0xE0A800


def test_disabled_notifier_does_not_post():
    session = FakeSession([FakeResponse(204, {})])
    notifier = DiscordNotifier("", session=session)
    assert notifier.enabled is False
    assert notifier.send_bets(fixture(), wallet_bets()) is False
    assert session.calls == []


def test_notifier_posts_and_returns_true():
    session = FakeSession([FakeResponse(204, {})])
    notifier = DiscordNotifier("https://example.test/webhook", session=session)
    assert notifier.send_bets(fixture(), wallet_bets()) is True
    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://example.test/webhook"
    assert kwargs["json"]["embeds"][0]["title"].startswith("Paper bets")


def test_notifier_returns_false_on_http_error():
    session = FakeSession([FakeResponse(500, {})])
    notifier = DiscordNotifier("https://example.test/webhook", session=session)
    assert notifier.send_bets(fixture(), wallet_bets()) is False
