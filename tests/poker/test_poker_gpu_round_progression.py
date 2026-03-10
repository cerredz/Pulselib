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
