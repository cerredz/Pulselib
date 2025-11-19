import yaml
from pathlib import Path

def get_config_file(file_name: str) -> dict or None:
    config_dir=Path(__file__).parent.parent/"config"
    config_file=config_dir/file_name
    if not Path.exists(config_file):
        return None
    
    with open(config_file, 'r') as file:
        config=yaml.safe_load(file)

    return config