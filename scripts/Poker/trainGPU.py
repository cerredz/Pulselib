from cProfile import Profile
import pstats
import time

import gymnasium as gym
import torch

from environments.Poker.Player import PokerQNetwork
from environments.Poker.utils import PokerAgentType, build_actions, get_rotated_agents, load_gpu_agents
from utils.benchmarking.benchmarking import Benchmarker, YamlBenchmarker
from utils.config import get_config_file, get_result_folder
from utils.plotting import MatplotlibPlotter, Plotter
from utils.torch import load_device


CONFIG_FILENAME = "pokerGPU.yaml"
REWARDS_FILENAME = "rewards_learning_curve"
CHIPS_FILENAME = "total_chips_curve"
POKER_ACTION_SPACE_N = 13
ENV_NAME = "Pulse-Poker-GPU-v1"


def _get_active_q_mask(terminated: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
    return q_mask & ~terminated


def _should_stop_loop(
    step_idx: int,
    terminated: torch.Tensor,
    termination_threshold: torch.Tensor,
    check_interval: int = 5,
) -> bool:
    return step_idx % check_interval == 0 and bool(terminated.float().mean() > termination_threshold)


def train_agent(
    env: gym.Env,
    agents,
    agent_types,
    episodes,
    n_games,
    device,
    results_dir,
    config,
    plotter: Plotter | None = None,
    benchmarker: Benchmarker | None = None,
):
    plotter = plotter or MatplotlibPlotter()
    benchmarker = benchmarker or YamlBenchmarker()
    total_steps = 0
    start_time = time.time()
    scores, reward_scores = [], []
    actions = torch.zeros(n_games, dtype=torch.long, device=device)
    q_agent_idx = agent_types.index(PokerAgentType.QLEARNING)
    q_agent = agents[q_agent_idx]

    for episode in range(episodes):
        rotated_agents, rotated_types, q_seat, rotations = get_rotated_agents(
            agents,
            agent_types,
            episode_idx=episode,
            q_agent_idx=q_agent_idx,
        )

        state, info = env.reset(
            options={
                "rotation": rotations,
                "active_players": True,
                "q_agent_seat": q_seat,
            }
        )

        initial_stacks = info["stacks"][:, q_seat].clone()
        terminated = torch.zeros(n_games, dtype=torch.bool, device=device)
        episode_reward_tensor = torch.tensor(0.0, device=device)
        termination_threshold = torch.tensor(0.8, device=device)
        idx = 0

        while True:
            actions.fill_(0)
            q_mask = info["seat_idx"] == q_seat
            build_actions(state, actions, info["seat_idx"], rotated_agents, rotated_types, device)
            next_state, rewards, dones, truncated, info = env.step(actions)
            del truncated
            active_games = q_mask & ~terminated  # [n_games]
            terminated |= dones
            if active_games.any():
                q_agent.train_step(
                    states=state[active_games],
                    actions=actions[active_games],
                    rewards=rewards[active_games],
                    next_states=next_state[active_games],
                    dones=dones[active_games],
                )

            episode_reward_tensor += rewards[active_games].sum()
            state = next_state

            if _should_stop_loop(idx, terminated, termination_threshold):
                break
            idx += 1

        final_stacks = info["stacks"][:, q_seat]
        episode_profit = (final_stacks - initial_stacks).sum().item()
        episode_reward = episode_reward_tensor.item()
        reward_scores.append(episode_reward)
        scores.append(episode_profit)
        total_steps += n_games * idx

        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            print(
                f"Episode {episode + 1:5d}/{episodes} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Q-Agent Profit: {episode_profit:10.2f} chips | "
                f"Speed: {steps_per_sec:6.1f} steps/sec"
            )

    torch.save(q_agent.network.state_dict(), f"{results_dir}/poker_qnet_final.pth")
    reward_path = results_dir / REWARDS_FILENAME
    chips_path = results_dir / CHIPS_FILENAME
    end_time = time.time()

    plotter.plot_learning_curve(
        scores=reward_scores,
        file_path=str(reward_path),
        window_size=10,
        title="Poker Q-Learning - Total Reward per Episode Batch",
    )
    plotter.plot_learning_curve(
        scores=scores,
        file_path=str(chips_path),
        window_size=10,
        title="Poker Q-Learning - Total Chip Profit per Episode Batch",
    )

    benchmarker.create_benchmark_file(
        env_name=ENV_NAME,
        episodes_return=reward_scores,
        start_time=start_time,
        end_time=end_time,
        total_steps=total_steps,
        config=config,
    )


if __name__ == "__main__":
    config = get_config_file(file_name=CONFIG_FILENAME)
    result_dir = get_result_folder(config["RESULTS_DIR"])
    q_learning_model_weights = result_dir / "poker_qnet_final.pth"
    plotter = MatplotlibPlotter.from_config(config.get("PLOTTING"))
    benchmarker = YamlBenchmarker.from_config(config.get("BENCHMARKING"))

    device = load_device()
    agents, agent_types = load_gpu_agents(
        device,
        config["NUM_PLAYERS"],
        config["AGENTS"],
        config["STARTING_BBS"],
        POKER_ACTION_SPACE_N,
    )
    q_net = PokerQNetwork(
        weights_path=q_learning_model_weights,
        device=device,
        gamma=config["GAMMA"],
        update_freq=config["UPDATE_FREQ"],
        state_dim=config["STATE_SPACE"],
        action_dim=config["ACTION_SPACE"],
        learning_rate=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    ).to(device)

    agents.insert(0, q_net)
    agent_types.insert(0, PokerAgentType.QLEARNING)

    env = gym.make(
        config["ENV_ID"],
        device=device,
        agents=agents,
        n_players=config["NUM_PLAYERS"] + 1,
        n_games=config["N_GAMES"],
        starting_bbs=config["STARTING_BBS"],
        w1=config["W1"],
        w2=config["W2"],
        K=config["K"],
        alpha=config["ALPHA"],
    )

    profiler = Profile()
    profiler.enable()
    train_agent(
        env,
        agents,
        agent_types,
        config["EPISODES"],
        config["N_GAMES"],
        device=device,
        results_dir=result_dir,
        config=config,
        plotter=plotter,
        benchmarker=benchmarker,
    )
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)
