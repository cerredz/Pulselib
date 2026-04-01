import torch

from environments.Poker.PokerGPU import PokerGPU


def _build_env(*, n_players: int = 6, max_players: int | None = None, n_games: int = 1) -> PokerGPU:
    max_players = max_players or n_players
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=n_players,
        max_players=max_players,
        n_games=n_games,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    return env


def test_get_obs_packs_core_fields_for_current_actor() -> None:
    env = _build_env(n_players=4)
    env.active_players = 4
    env.board[0] = torch.tensor([11, 22, 33, -1, -1], dtype=torch.int32)
    env.hands[0, 2] = torch.tensor([7, 19], dtype=torch.int32)
    env.stages[0] = torch.tensor(2, dtype=torch.int32)
    env.button[0] = torch.tensor(1, dtype=torch.int32)
    env.idx[0] = torch.tensor(2, dtype=torch.int32)
    env.pots[0] = torch.tensor(45, dtype=torch.int32)
    env.highest[0] = torch.tensor(12, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    env.stacks[0] = torch.tensor([80, 70, 60, 50], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED, env.ALLIN, env.ACTIVE], dtype=torch.int32)

    obs = env.get_obs()

    assert obs.shape == (1, env.obs_size)
    assert obs[0, 0:5].to(torch.int32).tolist() == [11, 22, 33, -1, -1]
    assert obs[0, 5:7].to(torch.int32).tolist() == [7, 19]
    assert obs[0, 7].item() == 2
    assert obs[0, 8].item() == 1
    assert obs[0, 9].item() == 45
    assert obs[0, 10].item() == 7
    assert obs[0, 11].item() == 60
    assert obs[0, 12].item() == env.ALLIN


def test_get_obs_short_handed_orders_only_live_opponents_and_zero_pads_unused_slots() -> None:
    env = _build_env(n_players=6, max_players=6)
    env.active_players = 4
    env.idx[0] = torch.tensor(3, dtype=torch.int32)
    env.button[0] = torch.tensor(1, dtype=torch.int32)
    env.stacks[0] = torch.tensor([101, 102, 103, 104, 999, 888], dtype=torch.int32)
    env.status[0] = torch.tensor(
        [env.ACTIVE, env.FOLDED, env.ALLIN, env.ACTIVE, env.SITOUT, env.SITOUT],
        dtype=torch.int32,
    )
    env.current_round_bet[0] = torch.tensor([11, 22, 33, 44, 55, 66], dtype=torch.int32)
    env.obs.fill_(777)

    obs = env.get_obs()

    assert obs[0, 13:22].to(torch.int32).tolist() == [
        101, env.ACTIVE, 11,
        102, env.FOLDED, 22,
        103, env.ALLIN, 33,
    ]
    assert obs[0, 22:].tolist() == [0.0] * (env.obs_size - 22)


def test_post_blinds_updates_pot_bets_and_allin_status() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.pots.zero_()
    env.current_round_bet.zero_()
    env.total_invested.zero_()
    env.status.fill_(env.ACTIVE)
    env.stacks.zero_()
    env.bb = torch.tensor([0, 1], dtype=torch.int32)
    env.stacks[0, 0] = 1
    env.stacks[1, 1] = 5

    env.post_blinds()

    assert env.pots.tolist() == [1, 1]
    assert env.current_round_bet.tolist() == [[1, 0], [0, 1]]
    assert env.total_invested.tolist() == [[1, 0], [0, 1]]
    assert env.stacks.tolist() == [[0, 0], [0, 4]]
    assert env.status.tolist() == [[env.ALLIN, env.ACTIVE], [env.ACTIVE, env.ACTIVE]]


def test_deal_players_cards_advances_positions_and_preserves_per_game_order() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.decks[0] = torch.arange(1, 53, dtype=torch.int32)
    env.decks[1] = torch.arange(101, 153, dtype=torch.int32)
    env.deck_positions.zero_()

    first_cards = env.deal_players_cards(4)
    second_cards = env.deal_players_cards(2)

    assert first_cards.tolist() == [[1, 2, 3, 4], [101, 102, 103, 104]]
    assert second_cards.tolist() == [[5, 6], [105, 106]]
    assert env.deck_positions.tolist() == [6, 6]


def test_deal_cards_updates_only_selected_games() -> None:
    env = _build_env(n_players=2, n_games=2)
    env.decks[0] = torch.arange(1, 53, dtype=torch.int32)
    env.decks[1] = torch.arange(101, 153, dtype=torch.int32)
    env.deck_positions[:] = torch.tensor([5, 7], dtype=torch.int32)

    cards = env.deal_cards(torch.tensor([1], dtype=torch.long), 2)

    assert cards.tolist() == [[108, 109]]
    assert env.deck_positions.tolist() == [5, 9]


def test_get_info_reports_active_players_stacks_and_current_seat() -> None:
    env = _build_env(n_players=3)
    env.active_players = 2
    env.idx[0] = torch.tensor(1, dtype=torch.int32)
    env.stacks[0] = torch.tensor([40, 50, 60], dtype=torch.int32)

    info = env.get_info()

    assert info["active_players"] == 2
    assert torch.equal(info["seat_idx"], env.idx)
    assert torch.equal(info["stacks"], env.stacks)


def test_execute_actions_ignores_folded_allin_sitout_and_done_rows() -> None:
    env = _build_env(n_players=4, n_games=4)
    env.idx[:] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    env.status[0] = torch.tensor([env.FOLDED, env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.status[1] = torch.tensor([env.ACTIVE, env.ALLIN, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.status[2] = torch.tensor([env.ACTIVE, env.ACTIVE, env.SITOUT, env.ACTIVE], dtype=torch.int32)
    env.status[3] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.is_done[:] = torch.tensor([False, False, False, True], dtype=torch.bool)
    env.stacks[:] = 100
    env.current_round_bet.zero_()
    env.total_invested.zero_()
    env.pots.zero_()
    env.highest[:] = 10
    env.acted.zero_()

    before_status = env.status.clone()
    before_stacks = env.stacks.clone()
    before_bets = env.current_round_bet.clone()
    before_pots = env.pots.clone()

    env.execute_actions(torch.tensor([0, 12, 2, 1], dtype=torch.long))

    assert torch.equal(env.status, before_status)
    assert torch.equal(env.stacks, before_stacks)
    assert torch.equal(env.current_round_bet, before_bets)
    assert torch.equal(env.pots, before_pots)
    assert env.acted.tolist() == [0, 0, 0, 0]


def test_execute_actions_check_only_marks_actor_as_acted_when_call_cost_is_zero() -> None:
    env = _build_env(n_players=2)
    env.idx[0] = torch.tensor(1, dtype=torch.int32)
    env.highest[0] = torch.tensor(4, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([4, 4], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([4, 4], dtype=torch.int32)
    env.stacks[0] = torch.tensor([50, 60], dtype=torch.int32)
    env.pots[0] = torch.tensor(8, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)

    env.execute_actions(torch.tensor([1], dtype=torch.long))

    assert env.stacks[0].tolist() == [50, 60]
    assert env.current_round_bet[0].tolist() == [4, 4]
    assert env.total_invested[0].tolist() == [4, 4]
    assert env.pots[0].item() == 8
    assert env.acted[0].item() == 1


def test_execute_actions_fractional_raise_rounds_down_to_int_chips() -> None:
    env = _build_env(n_players=2)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet.zero_()
    env.total_invested.zero_()
    env.stacks[0] = torch.tensor([100, 100], dtype=torch.int32)
    env.pots[0] = torch.tensor(10, dtype=torch.int32)
    env.acted.zero_()
    env.last_raise_size[:] = 1

    env.execute_actions(torch.tensor([4], dtype=torch.long))

    assert env.current_round_bet[0, 0].item() == 3
    assert env.total_invested[0, 0].item() == 3
    assert env.pots[0].item() == 13
    assert env.stacks[0, 0].item() == 97
    assert env.highest[0].item() == 3
