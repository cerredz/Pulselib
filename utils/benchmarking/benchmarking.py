# common benchmark features for different environments
# save the results to a file

from utils.benchmarking.episodes import benchmark_episode
from utils.benchmarking.files import create_files
from utils.config import get_result_folder_env
import json
import yaml

def create_benchmark_file(env_name, episodes_return, start_time, end_time, total_steps, config):
    results_dir=get_result_folder_env(env_name)
    best_path, curr_path = create_files(results_dir=results_dir)
    episode_stats=benchmark_episode(episodes_return=episodes_return)
    training_seconds=end_time-start_time

    benchmark_file_json={
        'env': env_name,
        'config': config,
        'start_time': start_time,
        'end_time': end_time,
        'total_training_seconds': training_seconds,
        'total_steps': total_steps,
        'sps': round(float(total_steps / training_seconds), 4)
    }

    print(curr_path)

    with open(curr_path, 'w') as file:
        yaml.dump(benchmark_file_json, file, default_flow_style=False)
