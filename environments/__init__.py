from gymnasium.envs.registration import register

register(
    id='Pulse-2048-v2', # The "public" name
    entry_point='environments.TFE:TFE',
    max_episode_steps=200000 
)

