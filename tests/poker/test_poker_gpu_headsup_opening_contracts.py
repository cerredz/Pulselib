import torch

from environments.Poker.PokerGPU import PokerGPU


def _build_env() -> PokerGPU:
    env = PokerGPU(
        device=torch.device("cpu"),
        agents=[],
        n_players=2,
        max_players=2,
        n_games=1,
    )
    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})
    return env


def test_reset_heads_up_assigns_button_as_small_blind_and_other_seat_as_big_blind() -> None:
    env = _build_env()

    assert env.button.tolist() == [0]
    assert env.sb.tolist() == [0]
    assert env.bb.tolist() == [1]
    assert env.idx.tolist() == [0]
    assert env.current_round_bet[0].tolist() == [0, 1]
    assert env.total_invested[0].tolist() == [0, 1]
    assert env.pots.tolist() == [1]


def test_reset_heads_up_second_hand_rotates_button_and_preserves_opening_order() -> None:
    env = _build_env()

    env.reset(options={"active_players": False, "q_agent_seat": 0, "rotation": 0})

    assert env.button.tolist() == [1]
    assert env.sb.tolist() == [1]
    assert env.bb.tolist() == [0]
    assert env.idx.tolist() == [1]
    assert env.current_round_bet[0].tolist() == [1, 0]
    assert env.total_invested[0].tolist() == [1, 0]
    assert env.pots.tolist() == [1]


def test_step_heads_up_preflop_call_then_check_advances_to_flop_with_big_blind_acting_first() -> None:
    env = _build_env()

    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))
    assert not dones[0].item()
    assert env.stages[0].item() == 0
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1
    assert env.current_round_bet[0].tolist() == [1, 1]

    _, _, dones, _, info = env.step(torch.tensor([1], dtype=torch.long))

    assert not dones[0].item()
    assert env.stages[0].item() == 1
    assert env.idx[0].item() == 1
    assert info["seat_idx"][0].item() == 1
    assert env.current_round_bet[0].tolist() == [0, 0]
    assert torch.all(env.board[0, 0:3] > 0)


def test_step_heads_up_button_fold_gives_big_blind_the_opening_pot() -> None:
    env = _build_env()
    env.stacks[0] = torch.tensor([100, 99], dtype=torch.int32)
    env.current_round_bet[0] = torch.tensor([0, 1], dtype=torch.int32)
    env.total_invested[0] = torch.tensor([0, 1], dtype=torch.int32)
    env.pots[0] = torch.tensor(1, dtype=torch.int32)
    env.idx[0] = torch.tensor(0, dtype=torch.int32)
    env.agg[0] = torch.tensor(1, dtype=torch.int32)
    env.highest[0] = torch.tensor(1, dtype=torch.int32)
    env.status[0] = torch.tensor([env.ACTIVE, env.ACTIVE], dtype=torch.int32)
    env.acted[0] = torch.tensor(0, dtype=torch.int32)
    env.is_done[0] = False

    _, _, dones, _, _ = env.step(torch.tensor([0], dtype=torch.long))

    assert dones[0].item()
    assert env.status[0].tolist() == [env.FOLDED, env.ACTIVE]
    assert env.stacks[0].tolist() == [100, 100]
    assert env.pots[0].item() == 0
