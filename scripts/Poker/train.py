
import profile
from environments.Poker.utils import debug_state, load_agents
from utils.config import get_config_file, get_result_folder
import gymnasium as gym
from cProfile import Profile
import pstats

CONFIG_FILENAME="poker.yaml"
POKER_ACTION_SPACE_N=13

def train_agent(env, agents, episodes):
    for i in range(episodes):
        state, info = env.reset() 
        terminated, truncated = False, False
        curr_player=state[8] # current player in our state
        while not terminated and not truncated:
            action=agents[curr_player].action(state)
            #debug_state(state, agents[curr_player].id, action)
            next_state, reward, terminated, truncated, info = env.step(action)

            #debug_state(next_state, agents[curr_player].id, action)
            
            # update q learning agents
            if curr_player==0:
                agents[curr_player].learn(state, action, reward, next_state, done=terminated)
            
            curr_player=next_state[8]
            state=next_state
            #break
        #break
    
        if i % 5 == 0:
            print(f"Episode {i}: Current stack: {agents[0].stack}")

if __name__ == "__main__":
    print("Initializing Poker training script...")
    config=get_config_file(file_name=CONFIG_FILENAME)
    result_dir=get_result_folder(config["RESULTS_DIR"])

    # load the agents that will be playing poker
    agents=load_agents(config["NUM_PLAYERS"], config["AGENTS"], config["STARTING_STACK"], POKER_ACTION_SPACE_N)

    env=gym.make(
        config["ENV_ID"], 
        agents=agents, 
        n=config["NUM_PLAYERS"], 
        bb=config["BIG_BLIND"], 
        starting_stack=config["STARTING_STACK"]
    )
    print("Training Poker Agent...")

    profiler = Profile()
    profiler.enable()
    train_agent(env, agents, config["EPISODES"])
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 slowest calls
    


