import torch

from environments.Poker.PokerGPU import PokerGPU


def _build_env(n_players: int = 3) -> PokerGPU:
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


def _prepare_persistent_stacks(env: PokerGPU, stacks: list[int]) -> None:
    env.stacks[0] = torch.tensor(stacks, dtype=torch.int32)
    env.button = torch.tensor([0], dtype=torch.int32, device=env.device)
    env.button_pos = 0


def _expected_stacks(
    stacks: list[int],
    rotation: int,
    *,
    starting_bbs: int = 100,
    max_bbs: int = 1000,
    bb_amount: int = 1,
) -> list[int]:
    expected = torch.tensor(stacks, dtype=torch.int32)
    expected[expected == 0] = starting_bbs
    expected[expected > max_bbs] = starting_bbs
    expected = torch.roll(expected, rotation, dims=0)
    expected[0] -= bb_amount
    return expected.tolist()


def test_reset_rolls_persistent_stacks_from_options_rotation():
    env = _build_env()
    _prepare_persistent_stacks(env, [10, 20, 30])

    _, info = env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 1})

    assert info["stacks"][0].tolist() == [29, 10, 20]


def test_reset_options_rotation_matches_explicit_rotation_argument():
    options_env = _build_env()
    explicit_env = _build_env()
    original_stacks = [10, 20, 30]
    _prepare_persistent_stacks(options_env, original_stacks)
    _prepare_persistent_stacks(explicit_env, original_stacks)

    _, options_info = options_env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 1})
    _, explicit_info = explicit_env.reset(
        options={"active_players": False, "q_agent_seat": 0, "rotation": 1},
        rotation=1,
    )

    assert options_info["stacks"][0].tolist() == explicit_info["stacks"][0].tolist()
    assert options_info["stacks"][0].tolist() == [29, 10, 20]


def test_reset_zero_rotation_keeps_stack_order_before_blind_post():
    env = _build_env()
    _prepare_persistent_stacks(env, [10, 20, 30])

    _, info = env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    assert info["stacks"][0].tolist() == [9, 20, 30]


def test_reset_wraps_options_rotation_values_larger_than_table_size():
    env = _build_env()
    _prepare_persistent_stacks(env, [10, 20, 30])

    _, info = env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 4})

    assert info["stacks"][0].tolist() == _expected_stacks([10, 20, 30], rotation=4)


def test_reset_restores_invalid_persistent_stacks_before_rotating():
    env = _build_env()
    _prepare_persistent_stacks(env, [0, 1001, 25])

    _, info = env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 1})

    assert info["stacks"][0].tolist() == _expected_stacks([0, 1001, 25], rotation=1)
