
from environments.Poker.utils import load_agents
from utils.config import get_config_file, get_result_folder
import gymnasium as gym

CONFIG_FILENAME="poker.yaml"
POKER_ACTION_SPACE_N=13

def train_agent():
    pass

if __name__ == "__main__":
    print("Initializing Poker training script...")
    config=get_config_file(file_name=CONFIG_FILENAME)
    result_dir=get_result_folder(config["RESULTS_DIR"])

    # load the agents that will be playing poker
    q_learning=[i for i in range(len(config["AGENTS"])) if config["AGENTS"][i] == 'qlearning']
    agents=load_agents(config["NUM_PLAYERS"], config["AGENTS"], config["STARTING_STACK"], POKER_ACTION_SPACE_N)
    
    env=gym.make(
        config["ENV_ID"], 
        agents=agents, 
        n=config["NUM_PLAYERS"], 
        bb=config["BIG_BLIND"], 
        starting_stack=config["STARTING_STACK"]
    )

    print("Training Poker Agent...")
    for i in range(config["EPISODES"]):
        state, info = env.reset() 
        terminated, truncated = False, False
        curr_player=state[8] # current player in our state
        print(f"state: {state}",)
        while not terminated and not truncated:
            action=agents[curr_player].action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # update q learning agents
            for q in q_learning:
                if q == curr_player:
                    agents[q].learn(state, action, reward, next_state, done=terminated)
            
            curr_player=next_state[15]
            state=next_state
        
        #if i % 5000 == 0:
        #    print(f"Episode {i}: Current stack: {agents[ql].stack}")


