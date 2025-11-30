
from environments.Poker.utils import load_agents
from utils.config import get_config_file, get_result_folder
import gymnasium as gym

CONFIG_FILENAME="poker.yaml"

def train_agent():
    pass

if __name__ == "__main__":
    config=get_config_file(file_name=CONFIG_FILENAME)
    result_dir=get_result_folder(config["RESULT_DIR"])

    # load the agents that will be playing poker
    q_learning=[i for i in range(len(config["AGENTS"])) if config["AGENTS"][i] == 'qlearning']
    agents=load_agents(config["NUM_PlAYERS"], q_learning)

    env=gym.make(config["ENV_ID"], agents=agents, n=config["PlAYERS"], bb=config["BIG_BLINDS"], starting_stack=config["STARTING_STACK"])

    for i in range(config["NUM_EPISODES"]):
        state, info = env.reset() 
        terminated, truncated = False, False
        episode=[]

        curr_player=state[15] # current player in our state
        while not terminated and not truncated:
            action=agents[curr_player].action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            curr_player=next_state[15]
            state=next_state
        
        for ql in q_learning:
            agents[ql].learn(episode=episode)

        #if i % 5000 == 0:
        #    print(f"Episode {i}: Current stack: {agents[ql].stack}")


