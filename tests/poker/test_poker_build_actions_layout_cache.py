from __future__ import annotations

import torch

from environments.Poker.utils import (
    PokerAgentType,
    _build_seat_mask,
    _get_grouped_agent_layout,
    build_actions,
)


class _RecordingAgent:
    def __init__(self, fill_value: int) -> None:
        self.fill_value = fill_value
        self.calls: list[torch.Tensor] = []

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        self.calls.append(states.clone())
        return torch.full((states.shape[0],), self.fill_value, dtype=torch.long, device=states.device)

    def action(self, states: torch.Tensor) -> torch.Tensor:
        self.calls.append(states.clone())
        return torch.full((states.shape[0],), self.fill_value, dtype=torch.long, device=states.device)


def test_grouped_agent_layout_is_cached_by_agent_order() -> None:
    agent_types = [
        PokerAgentType.QLEARNING,
        PokerAgentType.RANDOM,
        PokerAgentType.RANDOM,
        PokerAgentType.HEURISTIC_HANDS,
    ]

    first = _get_grouped_agent_layout(agent_types)
    second = _get_grouped_agent_layout(list(agent_types))

    assert first is second
    assert first == (
        (0, PokerAgentType.QLEARNING, (0,)),
        (1, PokerAgentType.RANDOM, (1, 2)),
        (3, PokerAgentType.HEURISTIC_HANDS, (3,)),
    )


def test_build_seat_mask_matches_all_requested_seats() -> None:
    curr_players = torch.tensor([0, 1, 2, 3, 1, 2], dtype=torch.long)

    mask = _build_seat_mask(curr_players, (1, 2))

    assert mask.tolist() == [False, True, True, False, True, True]


def test_build_actions_reuses_first_agent_for_shared_types() -> None:
    q_agent = _RecordingAgent(fill_value=9)
    random_agent = _RecordingAgent(fill_value=4)
    heuristic_agent = _RecordingAgent(fill_value=7)
    agents = [q_agent, random_agent, _RecordingAgent(fill_value=12), heuristic_agent]
    agent_types = [
        PokerAgentType.QLEARNING,
        PokerAgentType.RANDOM,
        PokerAgentType.RANDOM,
        PokerAgentType.HEURISTIC_HANDS,
    ]
    state = torch.arange(24, dtype=torch.float32).view(6, 4)
    actions = torch.full((6,), -1, dtype=torch.long)
    curr_players = torch.tensor([0, 1, 2, 3, 1, 2], dtype=torch.long)

    build_actions(
        state=state,
        actions=actions,
        curr_players=curr_players,
        agents=agents,
        agent_types=agent_types,
        device=torch.device("cpu"),
    )

    assert q_agent.calls[0].shape[0] == 1
    assert heuristic_agent.calls[0].shape[0] == 1
    assert actions[0].item() == 9
    assert actions[3].item() == 7
    assert actions[1].item() >= 0
    assert actions[2].item() >= 0
    assert actions[4].item() >= 0
    assert actions[5].item() >= 0
