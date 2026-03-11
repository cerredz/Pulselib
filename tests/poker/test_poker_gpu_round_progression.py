import torch

from environments.Poker.PokerGPU import PokerGPU


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


def test_step_skips_folded_and_all_in_seats_when_selecting_next_actor():
    env = _build_env(n_players=4)
    env.stacks[0] = torch.tensor([100, 100, 0, 100], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED, env.ALLIN, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(0, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.zeros(4, dtype=torch.int32)
    env.total_invested[0] = torch.zeros(4, dtype=torch.int32)
    env.is_done[0] = False

    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))

    assert not dones[0].item()
    assert info["seat_idx"][0].item() == 3
    assert env.idx[0].item() == 3
    assert env.stages[0].item() == 0


def test_step_marks_round_over_and_transitions_to_next_street():
    env = _build_env(n_players=4)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(2, dtype=torch.int32)
    env.acted[0] = torch.tensor(2, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.zeros(4, dtype=torch.int32)
    env.is_done[0] = False
    env.stages[0] = torch.tensor(0, dtype=torch.int32)

    _, _, dones, _, _ = env.step(torch.tensor([1], dtype=torch.long))

    assert not dones[0].item()
    assert env.stages[0].item() == 1
    assert env.highest[0].item() == 0
    assert env.current_round_bet[0].tolist() == [0, 0, 0, 0]
    assert env.agg[0].item() == ((env.button[0].item() + 1) % env.active_players)
    assert torch.all(env.board[0, 0:3] > 0)


def test_step_ends_hand_early_when_fold_leaves_single_survivor():
    env = _build_env(n_players=2)
    env.stacks[0] = torch.tensor([50, 60], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.zeros(2, dtype=torch.int32)
    env.total_invested[0] = torch.zeros(2, dtype=torch.int32)
    env.pots[0] = torch.tensor(10, dtype=torch.int32)
    env.is_done[0] = False

    _, _, dones, _, _ = env.step(torch.tensor([0], dtype=torch.long))

    assert dones[0].item()
    assert env.status[0].tolist() == [env.FOLDED, env.ACTIVE]
    assert env.pots[0].item() == 0
    assert env.stacks[0].tolist() == [50, 70]


def test_step_closes_multiway_preflop_after_big_blind_checks_option():
    env = _build_env(n_players=4)

    # Fresh reset uses button=0, sb=1, bb=2, first actor=3.
    # Preflop sequence: 3 calls, 0 calls, 1 calls, 2 checks option.
    for _ in range(4):
        _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))
        assert not dones[0].item()

    assert env.stages[0].item() == 1
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1
    assert env.current_round_bet[0].tolist() == [0, 0, 0, 0]
    assert torch.all(env.board[0, 0:3] > 0)


def test_step_closes_multiway_postflop_checkaround_and_advances_to_turn():
    env = _build_env(n_players=4)

    for _ in range(4):
        _, _, dones, _, _ = env.step(torch.tensor([1], dtype=torch.long))
        assert not dones[0].item()

    assert env.stages[0].item() == 1
    assert env.idx[0].item() == 1

    for _ in range(4):
        _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))
        assert not dones[0].item()

    assert env.stages[0].item() == 2
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1
    assert env.current_round_bet[0].tolist() == [0, 0, 0, 0]
    assert env.board[0, 3].item() > 0


def test_step_sets_heads_up_postflop_opener_to_first_active_seat_left_of_button():
    env = _build_env(n_players=2)

    for _ in range(2):
        _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))
        assert not dones[0].item()

    assert env.stages[0].item() == 1
    assert env.button[0].item() == 0
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1


def test_step_reward_uses_acting_seat_equity_before_turn_advances():
    env = _build_env(n_players=3)
    env.w1 = torch.tensor(1.0, device=env.device, dtype=torch.float32)
    env.w2 = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    env.K = torch.tensor(100, device=env.device, dtype=torch.int32)
    env.alpha = torch.tensor(1, device=env.device, dtype=torch.int32)

    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(2, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(1, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0, 1], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([0, 0, 1], dtype=torch.int32)
    env.pots[0] = torch.tensor(1, dtype=torch.int32)
    env.is_done[0] = False
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.equity_dirty[0] = True

    def fake_calculate_equities():
        env.equities.fill_(0.5)
        env.equities[0, 0] = 0.8
        env.equities[0, 1] = 0.2
        env.equities[0, 2] = 0.4
        env.equity_dirty[:] = False

    env.calculate_equities = fake_calculate_equities

    _, rewards, _, _, info = env.step(torch.tensor([1], dtype=torch.long))

    expected_actor_reward = torch.tanh(torch.tensor((0.8 * 2.0) / 100.0)).item()

    assert info["seat_idx"][0].item() == 1
    assert abs(rewards[0].item() - expected_actor_reward) < 1e-6


def test_step_reuses_equities_across_same_street_actions():
    env = _build_env(n_players=2)
    env.stages[0] = torch.tensor(1, dtype=torch.int32)
    env.board[0, 0:3] = torch.tensor([1, 2, 3], dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.zeros(2, dtype=torch.int32)
    env.total_invested[0] = torch.zeros(2, dtype=torch.int32)
    env.is_done[0] = False
    env.equity_dirty[0] = True

    original_calculate_equities = env.calculate_equities
    calculate_calls = 0

    def wrapped_calculate_equities():
        nonlocal calculate_calls
        calculate_calls += 1
        return original_calculate_equities()

    env.calculate_equities = wrapped_calculate_equities

    env.step(torch.tensor([1], dtype=torch.long))
    env.step(torch.tensor([1], dtype=torch.long))

    assert calculate_calls == 1


def test_step_recomputes_equities_after_street_transition():
    env = _build_env(n_players=4)
    env.status[0] = torch.tensor([env.ACTIVE, env.FOLDED, env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(2, dtype=torch.int32)
    env.acted[0] = torch.tensor(2, dtype=torch.int32)
    env.highest[0] = torch.tensor(0, dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    env.total_invested[0] = torch.zeros(4, dtype=torch.int32)
    env.is_done[0] = False
    env.stages[0] = torch.tensor(0, dtype=torch.int32)
    env.equity_dirty[0] = True

    original_calculate_equities = env.calculate_equities
    calculate_calls = 0

    def wrapped_calculate_equities():
        nonlocal calculate_calls
        calculate_calls += 1
        return original_calculate_equities()

    env.calculate_equities = wrapped_calculate_equities

    env.step(torch.tensor([1], dtype=torch.long))

    assert env.stages[0].item() == 1
    assert env.equity_dirty[0].item()

    env.step(torch.tensor([1], dtype=torch.long))

    assert calculate_calls == 2
