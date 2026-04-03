from __future__ import annotations

from pathlib import Path

import torch

from environments.Poker.Player import PokerQNetwork, _resolve_weights_path


def _build_q_network(*, weights_path: str | Path) -> PokerQNetwork:
    return PokerQNetwork(
        weights_path=weights_path,
        device=torch.device("cpu"),
        gamma=0.95,
        update_freq=20,
        state_dim=13,
        action_dim=13,
        learning_rate=1e-3,
        weight_decay=0.0,
    )


def test_resolve_weights_path_treats_empty_string_as_missing() -> None:
    assert _resolve_weights_path("") is None
    assert _resolve_weights_path("   ") is None
    assert _resolve_weights_path(None) is None


def test_q_network_skips_loading_for_empty_weights_path(monkeypatch) -> None:
    load_calls: list[Path | str] = []

    def fake_torch_load(path: Path | str, map_location: torch.device) -> dict[str, torch.Tensor]:
        del map_location
        load_calls.append(path)
        raise AssertionError("torch.load should not run for an empty weights path")

    monkeypatch.setattr(torch, "load", fake_torch_load)

    q_network = _build_q_network(weights_path="")

    assert isinstance(q_network, PokerQNetwork)
    assert load_calls == []


def test_q_network_loads_existing_weights_file(tmp_path: Path) -> None:
    weights_path = tmp_path / "checkpoint.pth"
    source_network = _build_q_network(weights_path="")
    expected_state = source_network.network.state_dict()
    torch.save(expected_state, weights_path)

    loaded_network = _build_q_network(weights_path=weights_path)

    loaded_state = loaded_network.network.state_dict()
    for key, tensor in expected_state.items():
        assert torch.equal(loaded_state[key], tensor)
