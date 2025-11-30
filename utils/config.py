import yaml
from pathlib import Path
from typing import Tuple

def get_config_file(file_name: str) -> dict or None:
    config_dir=Path(__file__).parent.parent/"config"
    config_file=config_dir/file_name
    if not Path.exists(config_file):
        return None
    
    with open(config_file, 'r') as file:
        config=yaml.safe_load(file)

    return config

def get_paths(config: dict) -> Tuple[Path, Path, Path]:
    """Generates output paths based on config filenames."""
    # Get the directory of the current script, then go up to root
    root_dir = Path(__file__).parent.parent.parent
    results_dir = root_dir / "results" / "2048"
    results_dir.mkdir(parents=True, exist_ok=True)

    reward_path = results_dir / config["REWARD_RESULT_FILENAME"]
    score_path = results_dir / config["SCORES_RESULT_FILENAME"]
    steps_path = results_dir / config["STEPS_RESULT_FILENAME"]
    
    return reward_path, score_path, steps_path

def get_result_folder(result_dir: str) -> Path:
    res=Path(__file__).parent.parent/"results"/result_dir
    if not Path.exists(res):
        res=res.mkdir(parents=True, exist_ok=True)
    return res