from gymnasium.envs.registration import register

register(
    id='Pulse-2048-v1', # The "public" name
    entry_point='environments.TwentyFourtyEight:Game2048Env',
    max_episode_steps=200000 
)