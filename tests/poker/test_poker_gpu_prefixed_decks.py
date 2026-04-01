from __future__ import annotations

import importlib

import pytest
import torch

from environments.Poker.PokerGPU import PokerGPU


poker_gpu_module = importlib.import_module("environments.Poker.PokerGPU")


def _install_hand_rank_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    original_exists = poker_gpu_module.Path.exists
    original_stat = poker_gpu_module.Path.stat

    def fake_exists(path: poker_gpu_module.Path) -> bool:
        if path.name == "HandRanks.dat":
            return True
        return original_exists(path)

    def fake_stat(path: poker_gpu_module.Path):
        if path.name == "HandRanks.dat":
            return type("FakeStat", (), {"st_size": 4 * 60})()
        return original_stat(path)

    monkeypatch.setattr(poker_gpu_module.Path, "exists", fake_exists)
    monkeypatch.setattr(poker_gpu_module.Path, "stat", fake_stat)
    monkeypatch.setattr(
        poker_gpu_module.torch,
        "from_file",
        lambda filename, shared, dtype, size: torch.zeros(size, dtype=dtype),
    )


def _build_env(monkeypatch: pytest.MonkeyPatch, n_players: int = 3, n_games: int = 2) -> PokerGPU:
    _install_hand_rank_stubs(monkeypatch)
    return PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=n_players,
        n_games=n_games,
    )


def test_reset_uses_prefixed_decks_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _build_env(monkeypatch)
    first_deck = torch.arange(1, 53, dtype=torch.int32)
    second_deck = torch.roll(torch.arange(1, 53, dtype=torch.int32), shifts=5)
    prefixed_decks = torch.stack([first_deck, second_deck], dim=0)

    env.reset(
        options={
            "active_players": False,
            "q_agent_seat": 0,
            "rotation": 0,
            "prefixed_decks": prefixed_decks,
        }
    )

    assert torch.equal(env.decks.cpu(), prefixed_decks)
    assert env.hands[0, :env.active_players].reshape(-1).tolist() == first_deck[: env.active_players * 2].tolist()
    assert env.hands[1, :env.active_players].reshape(-1).tolist() == second_deck[: env.active_players * 2].tolist()


def test_reset_rejects_prefixed_decks_with_wrong_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _build_env(monkeypatch)

    with pytest.raises(ValueError, match="prefixed_decks must have shape"):
        env.reset(
            options={
                "active_players": False,
                "q_agent_seat": 0,
                "rotation": 0,
                "prefixed_decks": torch.arange(1, 53, dtype=torch.int32),
            }
        )
