from gymnasium.envs.registration import register

register(
    id='Pulse-2048-v2', # The "public" name
    entry_point='environments.TFE:TFE',
    max_episode_steps=200000 
)

register(
    id='Pulse-Poker-v1', # The "public" name
    entry_point='environments.Poker:Poker',
    max_episode_steps=200000 
)

