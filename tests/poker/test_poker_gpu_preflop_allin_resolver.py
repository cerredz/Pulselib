import eval7
import torch

from environments.Poker.PokerGPU import PokerGPU


def _encode(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _cards(*card_strs: str) -> torch.Tensor:
    return torch.tensor([_encode(card) for card in card_strs], dtype=torch.int32)


def _ordered_deck(*card_strs: str) -> torch.Tensor:
    used_cards = [_encode(card) for card in card_strs]
    remaining_cards = [card for card in range(1, 53) if card not in used_cards]
    return torch.tensor(used_cards + remaining_cards, dtype=torch.int32)


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


def test_resolve_terminated_games_deals_full_board_for_preflop_all_in_showdown() -> None:
    env = _build_env(n_players=2)
    env.hands[0, 0] = _cards("Ah", "Ad")
    env.hands[0, 1] = _cards("Kc", "Kd")
    env.decks[0] = _ordered_deck(
        "Ah",
        "Ad",
        "Kc",
        "Kd",
        "2s",
        "2c",
        "7d",
        "9h",
        "3s",
        "Js",
        "4c",
        "Qd",
    )
    env.deck_positions[0] = torch.tensor(4, dtype=torch.int32)
    env.board[0] = torch.full((5,), -1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(100, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist() == _cards("2c", "7d", "9h", "Js", "Qd").tolist()
    assert env.deck_positions[0].item() == 12
    assert env.pots[0].item() == 0
    assert env.stages[0].item() == 5


def test_resolve_terminated_games_preflop_all_in_uses_runout_to_pick_winner() -> None:
    env = _build_env(n_players=2)
    env.hands[0, 0] = _cards("Ah", "Ad")
    env.hands[0, 1] = _cards("Kc", "Kd")
    env.decks[0] = _ordered_deck(
        "Ah",
        "Ad",
        "Kc",
        "Kd",
        "2s",
        "2c",
        "7d",
        "9h",
        "3s",
        "Js",
        "4c",
        "Qd",
    )
    env.deck_positions[0] = torch.tensor(4, dtype=torch.int32)
    env.board[0] = torch.full((5,), -1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(100, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.stacks[0].tolist() == [100, 0]
    assert env.board[0].tolist() == _cards("2c", "7d", "9h", "Js", "Qd").tolist()


def test_resolve_terminated_games_preflop_all_in_splits_tied_runout() -> None:
    env = _build_env(n_players=2)
    env.hands[0, 0] = _cards("Ac", "Kd")
    env.hands[0, 1] = _cards("Ad", "Kc")
    env.decks[0] = _ordered_deck(
        "Ac",
        "Kd",
        "Ad",
        "Kc",
        "2s",
        "Qh",
        "Jh",
        "Td",
        "3s",
        "2c",
        "4c",
        "7d",
    )
    env.deck_positions[0] = torch.tensor(4, dtype=torch.int32)
    env.board[0] = torch.full((5,), -1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN], dtype=torch.int32)
    env.stacks[0] = torch.tensor([10, 20], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([12, 12], dtype=torch.int32)
    env.pots[0] = torch.tensor(24, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist() == _cards("Qh", "Jh", "Td", "2c", "7d").tolist()
    assert env.stacks[0].tolist() == [22, 32]


def test_resolve_terminated_games_handles_batched_preflop_and_river_showdowns() -> None:
    env = _build_env(n_players=2, n_games=2)

    env.hands[0, 0] = _cards("Ah", "Ad")
    env.hands[0, 1] = _cards("Kc", "Kd")
    env.decks[0] = _ordered_deck(
        "Ah",
        "Ad",
        "Kc",
        "Kd",
        "2s",
        "2c",
        "7d",
        "9h",
        "3s",
        "Js",
        "4c",
        "Qd",
    )
    env.deck_positions[0] = torch.tensor(4, dtype=torch.int32)
    env.board[0] = torch.full((5,), -1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ALLIN, env.ALLIN], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([50, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(100, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = True

    env.board[1] = _cards("2c", "7d", "9h", "Js", "Kd")
    env.hands[1, 0] = _cards("Ah", "Qh")
    env.hands[1, 1] = _cards("3c", "4d")
    env.status[1] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.stacks[1] = torch.tensor([50, 50], dtype=torch.int32)
    env.total_invested[1] = torch.tensor([20, 20], dtype=torch.int32)
    env.pots[1] = torch.tensor(40, dtype=torch.int32)
    env.stages[1] = torch.tensor(4, dtype=torch.int32)
    env.is_done[1] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist() == _cards("2c", "7d", "9h", "Js", "Qd").tolist()
    assert env.board[1].tolist() == _cards("2c", "7d", "9h", "Js", "Kd").tolist()
    assert env.stacks[0].tolist() == [100, 0]
    assert env.stacks[1].tolist() == [90, 50]
    assert env.pots.tolist() == [0, 0]
    assert env.stages.tolist() == [5, 5]


def test_resolve_terminated_games_does_not_run_preflop_board_when_single_survivor_remains() -> None:
    env = _build_env(n_players=2)
    env.decks[0] = _ordered_deck(
        "Ah",
        "Ad",
        "Kc",
        "Kd",
        "2s",
        "2c",
        "7d",
        "9h",
        "3s",
        "Js",
        "4c",
        "Qd",
    )
    env.deck_positions[0] = torch.tensor(4, dtype=torch.int32)
    env.board[0] = torch.full((5,), -1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED], dtype=torch.int32)
    env.stacks[0] = torch.tensor([0, 50], dtype=torch.int32)
    env.pots[0] = torch.tensor(100, dtype=torch.int32)
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = True

    env.resolve_terminated_games()

    assert env.board[0].tolist() == [-1, -1, -1, -1, -1]
    assert env.deck_positions[0].item() == 4
    assert env.pots[0].item() == 100
    assert env.stages[0].item() == 0
