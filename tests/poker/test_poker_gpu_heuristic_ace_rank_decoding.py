import eval7
import torch

from environments.Poker.Player import (
    HeuristicHandsPlayerGPU,
    LoosePassivePlayerGPU,
    SmallBallPlayerGPU,
    TightAggressivePlayerGPU,
)
from environments.Poker.utils import build_actions, load_gpu_agents


STATE_DIM = 40


def _encode_gpu_card(card_str: str) -> int:
    card = eval7.Card(card_str)
    return card.rank + (13 * card.suit) + 1


def _build_states(hands: list[tuple[str, str]], *, pot_size: int = 0) -> torch.Tensor:
    states = torch.zeros((len(hands), STATE_DIM), dtype=torch.int32)
    states[:, 9] = pot_size
    for row_idx, (card1, card2) in enumerate(hands):
        states[row_idx, 5] = _encode_gpu_card(card1)
        states[row_idx, 6] = _encode_gpu_card(card2)
    return states


def test_heuristic_hands_gpu_raises_ace_two_with_one_based_ids() -> None:
    player = HeuristicHandsPlayerGPU(starting_stack=100, player_id=0, device=torch.device("cpu"))
    states = _build_states([("As", "2c")])

    actions = player.action(states)

    assert actions.shape == (1,)
    assert actions[0].item() >= 2


def test_heuristic_hands_gpu_batched_ace_high_and_low_cards_diverge() -> None:
    player = HeuristicHandsPlayerGPU(starting_stack=100, player_id=0, device=torch.device("cpu"))
    states = _build_states([("As", "2c"), ("3c", "4d")])

    actions = player.action(states)

    assert actions[0].item() >= 2
    assert actions[1].item() == 0


def test_tight_aggressive_gpu_calls_ace_seven_instead_of_folding() -> None:
    player = TightAggressivePlayerGPU(starting_stack=100, player_id=0, device=torch.device("cpu"))
    states = _build_states([("As", "7c")])

    actions = player.action(states)

    assert actions[0].item() == 1


def test_loose_passive_gpu_does_not_fold_ace_king() -> None:
    player = LoosePassivePlayerGPU(starting_stack=100, player_id=0, device=torch.device("cpu"))
    states = _build_states([("As", "Kd")])

    actions = player.action(states)

    assert actions[0].item() >= 1


def test_small_ball_gpu_raises_ace_king_when_pot_is_manageable() -> None:
    player = SmallBallPlayerGPU(starting_stack=100, player_id=0, device=torch.device("cpu"))
    states = _build_states([("As", "Kd")], pot_size=20)

    actions = player.action(states)

    assert actions[0].item() >= 2


def test_build_actions_routes_configured_gpu_heuristics_with_ace_high_hands() -> None:
    device = torch.device("cpu")
    agents, agent_types = load_gpu_agents(
        device=device,
        num_players=4,
        agent_types=["tight_aggressive", "heuristic_hands", "loose_passive", "small_ball"],
        starting_stack=100,
        action_space_n=13,
    )
    states = _build_states(
        [("As", "7c"), ("As", "2c"), ("As", "Kd"), ("As", "Kd")],
        pot_size=20,
    )
    curr_players = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    actions = torch.zeros(4, dtype=torch.long, device=device)

    build_actions(states, actions, curr_players, agents, agent_types, device)

    assert actions[0].item() == 1
    assert actions[1].item() >= 2
    assert actions[2].item() >= 1
    assert actions[3].item() >= 2
