import json
import time
from pathlib import Path
from typing import Dict, Any


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class TrainingLogger:
    """
    A simple and reusable logger for training runs, saving logs to a specific directory.
    """
    def __init__(self, log_dir: str, run_number: int | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if run_number is None:
            # Auto-increment run number
            existing_logs = list(self.log_dir.glob("logs_*.txt"))
            run_numbers = []
            for log_file in existing_logs:
                try:
                    num = int(log_file.stem.split('_')[1])
                    run_numbers.append(num)
                except ValueError:
                    pass
            self.run_number = max(run_numbers) + 1 if run_numbers else 1
        else:
            self.run_number = run_number
            
        self.log_file = self.log_dir / f"logs_{self.run_number}.txt"
        self.log_file.touch()

    def log(self, message: str, metrics: Dict[str, Any] | None = None) -> None:
        """
        Logs a message and an optional dictionary of metrics to the file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        if metrics is not None:
            serializable_metrics = _to_json_serializable(metrics)
            log_entry += f" | Metrics: {json.dumps(serializable_metrics)}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
            
    def get_log_file_path(self) -> str:
        return str(self.log_file)
