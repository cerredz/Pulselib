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

register(
    id='Pulse-Poker-GPU-v1',
    entry_point='environments.Poker:PokerGPU',
    max_episode_steps=200000 
)

register(
    id="Pulse-Blackjack-Standard",
    entry_point='environments.blackjack.blackjack:BlackJack',
    max_episode_steps=100000
)

register(
    id="Pulse-Particle-2d",
    entry_point='environments.Particle2D.Particle2D:Particle2D',
    max_episode_steps=100000
)