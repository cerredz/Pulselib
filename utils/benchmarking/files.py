# benchmarking utility function for benchmarking file system
from pathlib import Path

def create_files(results_dir):
    assert Path.exists(results_dir), "result folder must exist"
    runs_path=results_dir/"runs"
    if not Path.exists(runs_path): runs_path.mkdir()
    runs_files=[file for file in runs_path.iterdir() if file.is_file()]
    best=results_dir/"best_performance.json"
    curr=runs_path/f"run_{len(runs_files)+1}.yaml"

    return best, curr
