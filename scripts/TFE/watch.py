import time
import yaml
import torch
import gymnasium as gym
import environments # Registers Pulse-2048-v1
from pathlib import Path
from models.tfe import TFELightning
from agents.DQN_agent import DQNAgent

# --- Settings ---
ENV_ID = 'Pulse-2048-v1'
CONFIG_PATH = "config/tfe.yaml"
MODEL_WEIGHTS_PATH = Path(__file__).parent.parent / "results" / "2048" / "tfe_light_model_weights.pt"
DELAY = 0.1 # Seconds between moves

if __name__ == "__main__":
    # 1. Load Config
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Setup Single Environment (Not Vectorized)
    env = gym.make(ENV_ID)

    # 3. Load Model Architecture
    model = TFELightning(lr=config['learning_rate']).to(device)
    
    # 4. Load Trained Weights
    if MODEL_WEIGHTS_PATH.exists():
        print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode (disables dropout/batchnorm updates)
    else:
        print("No model weights found! Running with random initialization.")

    # 5. Initialize Agent (Force Epsilon to 0.0 to strictly use the brain)
    agent = DQNAgent(
        action_space=env.action_space,
        model=model,
        gamma=config['gamma'],
        epsilon_start=0.1, # <--- CRITICAL: No exploration, pure exploitation
        epsilon_decay=0.999,
        epsilon_end=0.01,
        batch_size=1,      # Doesn't matter for inference
        weight_decay=0.0,
        target_update=100
    )

    # 6. The Game Loop
    print("\n--- Starting Game ---")
    state, info = env.reset()
    env.render()
    time.sleep(1)

    terminated = False
    truncated = False
    total_score = 0
    step = 0

    while not terminated and not truncated:
        # Get best action from model
        action = agent.action(state)
        
        # Step environment
        state, reward, terminated, truncated, info = env.step(action)
        total_score = info["total_score"]
        step += 1

        # Render and Wait
        # (Optional: clear screen depending on OS to make it look like an animation)
        # import os; os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"Step: {step} | Action: {['Up', 'Down', 'Left', 'Right'][action]}")
        env.render()
        time.sleep(DELAY)

    print(f"Game Over! Final Score: {total_score}")
    env.close()