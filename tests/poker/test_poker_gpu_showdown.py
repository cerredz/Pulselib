import eval7
import torch

from environments.Poker.PokerGPU import PokerGPU


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _build_env(n_players: int) -> PokerGPU:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=n_players,
        n_games=1,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    env.active_players = n_players
    return env


def test_resolve_terminated_games_pays_strongest_active_hand():
    env = _build_env(n_players=2)
    env.board[0] = torch.tensor([_encode(card) for card in ["2c", "7d", "9h", "Js", "Kd"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("Ah"), _encode("Qh")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("3c"), _encode("4d")], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(40, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [90, 50]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_excludes_folded_players_from_winning():
    env = _build_env(n_players=3)
    env.board[0] = torch.tensor([_encode(card) for card in ["As", "Ks", "Qs", "Js", "2d"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("Ts"), _encode("3c")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("9h"), _encode("9d")], dtype=torch.int32)
    env.hands[0, 2] = torch.tensor([_encode("4c"), _encode("4d")], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([100, 100, 100], dtype=torch.int32)
    env.pots[0] = torch.tensor(90, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [190, 100, 100]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_splits_tied_even_pot():
    env = _build_env(n_players=2)
    env.board[0] = torch.tensor([_encode(card) for card in ["Ah", "Kd", "Qc", "Js", "9d"]], dtype=torch.int32)
    env.hands[0, 0] = torch.tensor([_encode("2c"), _encode("3d")], dtype=torch.int32)
    env.hands[0, 1] = torch.tensor([_encode("2d"), _encode("3c")], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([10, 20], dtype=torch.int32)
    env.pots[0] = torch.tensor(24, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [22, 32]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5
