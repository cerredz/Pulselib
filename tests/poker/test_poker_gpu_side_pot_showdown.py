import eval7
import torch

from environments.Poker.PokerGPU import PokerGPU


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _cards(*card_strs: str) -> torch.Tensor:
    return torch.tensor([_encode(card) for card in card_strs], dtype=torch.int32)


def _build_env(n_players: int, n_games: int = 1) -> PokerGPU:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=n_players,
        n_games=n_games,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    env.active_players = n_players
    return env


def test_resolve_terminated_games_awards_main_and_side_pot_by_commitment() -> None:
    env = _build_env(n_players=3)
    env.board[0] = _cards("As", "Ks", "Qs", "Js", "2d")
    env.hands[0, 0] = _cards("Ts", "3c")
    env.hands[0, 1] = _cards("9h", "9d")
    env.hands[0, 2] = _cards("4c", "4d")
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0, 100], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(110, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [30, 80, 100]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_splits_main_pot_before_awarding_side_pot() -> None:
    env = _build_env(n_players=3)
    env.board[0] = _cards("Ah", "Kd", "Qc", "Js", "3d")
    env.hands[0, 0] = _cards("2c", "4d")
    env.hands[0, 1] = _cards("2d", "4c")
    env.hands[0, 2] = _cards("9h", "9c")
    env.status[0] = torch.tensor([env.ALLIN, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 70, 70], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 30, 30], dtype=torch.int32)
    env.pots[0] = torch.tensor(70, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [15, 125, 70]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_keeps_folded_chips_in_side_pot_without_making_folded_player_eligible() -> None:
    env = _build_env(n_players=3)
    env.board[0] = _cards("As", "Ks", "Qs", "Js", "2d")
    env.hands[0, 0] = _cards("Ts", "3c")
    env.hands[0, 1] = _cards("9h", "9d")
    env.hands[0, 2] = _cards("4c", "4d")
    env.status[0] = torch.tensor([env.ALLIN, env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 100, 100], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(110, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [30, 180, 100]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_handles_multiple_side_pot_layers() -> None:
    env = _build_env(n_players=4)
    env.board[0] = _cards("As", "Ks", "Qs", "Js", "2d")
    env.hands[0, 0] = _cards("Ts", "3c")
    env.hands[0, 1] = _cards("9h", "9d")
    env.hands[0, 2] = _cards("Kc", "Kd")
    env.hands[0, 3] = _cards("4c", "4d")
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN, env.ALLIN, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0, 0, 50], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 30, 50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(140, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [40, 60, 40, 50]
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_supports_batched_games_with_mixed_showdown_shapes() -> None:
    env = _build_env(n_players=3, n_games=2)

    env.board[0] = _cards("As", "Ks", "Qs", "Js", "2d")
    env.hands[0, 0] = _cards("Ts", "3c")
    env.hands[0, 1] = _cards("9h", "9d")
    env.hands[0, 2] = _cards("4c", "4d")
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN, env.ACTIVE], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0, 100], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([10, 50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(110, dtype=torch.int32)
    env.stages[0] = torch.tensor(4, dtype=torch.int32)
    env.is_done[0] = True

    env.board[1] = _cards("2c", "7d", "9h", "Js", "Kd")
    env.hands[1, 0] = _cards("Ah", "Qh")
    env.hands[1, 1] = _cards("3c", "4d")
    env.hands[1, 2] = _cards("5c", "6d")
    env.status[1] = torch.tensor([env.ACTIVE, env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[1] = torch.tensor([50, 50, 50], dtype=torch.int32)
    env.total_invested[1] = torch.tensor([20, 20, 20], dtype=torch.int32)
    env.pots[1] = torch.tensor(60, dtype=torch.int32)
    env.stages[1] = torch.tensor(4, dtype=torch.int32)
    env.is_done[1] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [30, 80, 100]
    assert env.stacks[1].tolist() == [110, 50, 50]
    assert env.pots.tolist() == [0, 0]
    assert env.stages.tolist() == [5, 5]
