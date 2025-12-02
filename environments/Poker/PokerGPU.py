from multiprocessing import Value
from environments.Poker.utils import Agent, validate_agents
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import List

class PokerGPU(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS=13
    ACTIVE, FOLDED, ALLIN = 0, 1, 2
    STATE_SPACE=28 # details in depth_notes in rl folder

    def __init__(self, device, agents, n_players=6, n_games=100, starting_bbs=100, max_bbs=1000):
        super().__init__()

        # env 
        self.device=device
        self.agents=validate_agents(agents=agents)
        self.n_players=n_players
        self.n_games=n_games
        self.starting_bbs=starting_bbs
        self.max_bbs=max_bbs

        # action space / observation space
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.raise_fractions=torch.tensor([0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00], device=self.device)
        self.obs_size=12+((self.n_players-1)*3)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(self.obs_size,), dtype=torch.float32)

        # Initialize state tensors as None (will be set in reset)
        self.stacks = None
        self.total_busted=0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # game state tensors
        self.decks=torch.stack([torch.randperm(52, device=self.device) for _ in range(self.n_games)])
        self.deck_positions=torch.zeros(self.n_games, device=self.device, dtype=torch.uint8)
        
        self.board = torch.full((self.n_games, 5), -1, dtype=torch.uint8, device=self.device)
        self.pots = torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)
        self.stages = torch.zeros(self.n_games, dtype=torch.uint32, device=self.device)

        # player state tensors
        # refill/reset stacks
        if self.stacks is None:
            self.stacks = torch.full((self.n_games, self.n_players), self.starting_bbs, dtype=torch.float32, device=self.device)
        else:
            busted = (self.stacks == 0)
            above_max = (self.stacks > self.max_bbs)
            self.stacks[busted] = self.starting_bbs
            self.stacks[above_max] = self.starting_bbs
            self.total_busted+=busted.sum().items()
        
        self.hands = self.deal_cards(self.n_players * 2).view(self.n_games, self.n_players, 2)

        self.current_round_bet = torch.zeros((self.n_games, self.n_players), dtype=torch.uint32, device=self.device)
        self.total_invested = torch.zeros((self.n_games, self.n_players), dtype=torch.uint32, device=self.device)
        self.status = torch.full((self.n_games, self.n_players), self.ACTIVE, dtype=torch.uint8, device=self.device)

        self.button = (self.button + 1) % self.n_players if hasattr(self, 'button_pos') else torch.zeros(self.n_games, dtype=torch.long, device=self.device)
        self.sb = (self.button_pos + 1) % self.n_players
        self.bb = (self.button_pos + 2) % self.n_players

        self.post_blinds()

        self.idx = (self.bb_pos + 1) % self.n_players
        self.highest = torch.ones(self.n_games, dtype=torch.uint32, device=self.device)
        self.agg = self.bb.clone()
        self.acted = torch.zeros(self.n_games, dtype=torch.uint8, device=self.device)

        self.is_done = torch.zeros(self.n_games, dtype=torch.bool, device=self.device)
        return self.get_obs(), self.get_info()

    def get_obs(self):
        obs=torch.tensor()

    def get_info(self):
        pass

    def post_blinds(self):
        # handle the logic of when we "bet" the blinds for the small and big blinds
        # handle only big blind, small blind gets rounded down to 0
        game_idx=torch.arange(self.n_games, device=self.device)
        bb_amount=torch.ones(self.n_games, dtype=torch.uint8, device=self.device)
        self.stacks[game_idx, self.bb] -= bb_amount
        self.current_round_bet[game_idx, self.bb] += bb_amount
        self.total_invested[game_idx, self.bb] += bb_amount
        self.pots += bb_amount
        self.status[game_idx, self.bb] = torch.where(
            self.stacks[game_idx, self.bb] == 0,
            self.ALLIN,
            self.ACTIVE
        )

    def deal_cards(self, n_cards):
        """Deal n_cards from each game's deck"""
        game_idx = torch.arange(self.n_games, device=self.device).unsqueeze(1)
        card_idx = self.deck_positions.unsqueeze(1) + torch.arange(n_cards, device=self.device).unsqueeze(0)
        cards = self.decks[game_idx, card_idx]
        self.deck_positions += n_cards
        return cards

    
