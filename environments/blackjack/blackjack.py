import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

# openai gynasium file that implements a vectorized series of blackjack games using pytorch
# utilizing massive parrallism for games, not just simulating 1 black game, but many

class BlackJack(gym.Env):
    metadata = {'render.modes': ['human']}
    NUM_ACTIONS = 2 # hit, stand
    WIN_REWARD, LOSS_REWARD = 1, -1

    def __init__(self, device, batch_size):
        super().__init__()
        self.device=device
        self.batch_size=batch_size
        self.action_space=spaces.Discrete(self.NUM_ACTIONS)
        self.obs_size=3
        self.observation_space = spaces.Box(low=0, high=10000, shape=(self.obs_size,), dtype=np.float32)
        self.g=torch.arange(self.batch_size, device=self.device, dtype=torch.int32)

    def reset(self, seed=None, options=None):
        base_deck = torch.arange(52, device=self.device)
        base_deck_expanded = base_deck.unsqueeze(0).expand(self.batch_size, -1)
        
        # Generate random indices for each deck in parallel
        random_indices = torch.argsort(torch.rand(self.batch_size, 52, device=self.device), dim=1)
        self.decks = torch.gather(base_deck_expanded, 1, random_indices)

        self.deck_positions=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.terminated=torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.players_cards=torch.zeros((self.batch_size, 20), device=self.device, dtype=torch.int32)
        self.players_card_idx=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.player_card_sums=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.has_ace=torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.obs=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.dealer_cards=torch.zeros((self.batch_size, 20), device=self.device, dtype=torch.int32)
        self.dealer_card_idx=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.dealer_upcard=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.dealer_card_sums=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.dealer_has_ace=torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.rewards=torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.obs=torch.zeros((self.batch_size, 3), device=self.device, dtype=torch.int32)

        # deal the first cards
        self.deal_starting_cards()
        return self.get_obs(), self.get_info()
        
    # deal the starting cards of a blackjack game
    def deal_starting_cards(self):
        # deal the first card to the player
        ranks = torch.clamp(self.decks[self.g, self.deck_positions] % 13 + 1, max=10).to(torch.int32)
        aces = (ranks == 1)
        ranks[aces] = 11
        self.players_cards[self.g, self.players_card_idx] = ranks
        self.players_card_idx += 1
        self.deck_positions += 1
        self.has_ace[aces] = True

        # deal the dealers first card (upcard)
        dealer_rank = torch.clamp(self.decks[self.g, self.deck_positions] % 13 + 1, max=10).to(torch.int32)
        dealer_ace = (dealer_rank == 1)
        dealer_rank[dealer_ace] = 11
        self.dealer_cards[self.g, self.dealer_card_idx] = dealer_rank
        self.dealer_card_idx += 1
        self.dealer_upcard = dealer_rank.clone()
        self.deck_positions += 1
        self.dealer_has_ace[dealer_ace]=True

        # deal the player's second card
        ranks2 = torch.clamp(self.decks[self.g, self.deck_positions] % 13 + 1, max=10).to(torch.int32)
        aces2 = (ranks2 == 1)
        ranks2[aces2] = 11
        self.players_cards[self.g, self.players_card_idx] = ranks2
        self.players_card_idx += 1
        self.deck_positions += 1
        self.has_ace |= aces2

        # deal the dealers second card (hole card)
        dealer_rank2 = torch.clamp(self.decks[self.g, self.deck_positions] % 13 + 1, max=10).to(torch.int32)
        dealer_first_ace = ~dealer_ace & (dealer_rank2 == 1)
        dealer_rank2[dealer_rank2 == 1] = 11
        self.dealer_cards[self.g, self.dealer_card_idx] = dealer_rank2
        self.dealer_has_ace[dealer_first_ace]=True
        self.dealer_card_idx += 1
        self.deck_positions += 1

        self.player_card_sums[self.g] = self.players_cards[self.g].sum(dim=1).to(torch.int32)
        self.dealer_card_sums[self.g] = self.dealer_cards[self.g].sum(dim=1).to(torch.int32)

        # handle edge case (player gets dealt 2 aces, they have sum of 22 currently)
        over_21_ace = (self.player_card_sums > 21) & self.has_ace
        self.player_card_sums[over_21_ace] -= 10
        self.has_ace[over_21_ace]=False

        # handle same edge case for dealer with 2 aces
        # handle edge case (player gets dealt 2 aces, they have sum of 22 currently)
        over_21_ace = (self.dealer_card_sums > 21) & self.dealer_has_ace
        self.dealer_card_sums[over_21_ace] -= 10
        self.dealer_has_ace[over_21_ace]=False

    def get_obs(self):
        # state space: your sum, ace flag, dealers upcard
        self.obs[self.g, 0] = self.player_card_sums[self.g]
        self.obs[self.g, 1] = self.has_ace[self.g].to(torch.int32)
        self.obs[self.g, 2] = self.dealer_upcard[self.g]
        return self.obs

    def get_info(self):
        return {}

    def execute_actions(self, actions):
        assert actions.shape == (self.batch_size,)

       # action 0: hit
        hit_mask=(actions==0) & ~self.terminated
        cards=self.decks[hit_mask, self.deck_positions[hit_mask]]
        ranks=torch.clamp(cards % 13 + 1, max=10).to(torch.int32)
        already_has_ace=self.has_ace[hit_mask]
        ace_mask=(ranks==1)
        ranks[ace_mask & ~already_has_ace] = 11

        batch_idx=self.g[hit_mask]
        card_pos=self.players_card_idx[hit_mask]
        self.players_cards[batch_idx, card_pos] = ranks
        self.has_ace[hit_mask] |= (ace_mask & ~already_has_ace)
        self.player_card_sums[hit_mask] += ranks
        self.deck_positions[hit_mask] += 1
        self.players_card_idx[hit_mask] += 1

        # handle over 21
        over_21_ace=hit_mask & (self.player_card_sums > 21) & self.has_ace
        self.player_card_sums[over_21_ace] -= 10
        self.has_ace[over_21_ace]=False

        # action 1: stand
        stand_mask=(actions==1) & ~self.terminated
        active_dealers=(self.dealer_card_sums < 17) & stand_mask
        
        while active_dealers.any():
            cards=self.decks[active_dealers, self.deck_positions[active_dealers]]
            ranks=torch.clamp(cards % 13 + 1, max=10).to(torch.int32)
            ace_mask=(ranks==1)
            dealer_already_has_ace=self.dealer_has_ace[active_dealers]
            ranks[ace_mask & ~dealer_already_has_ace] = 11

            batch_idx=self.g[active_dealers]
            card_pos = self.dealer_card_idx[active_dealers]
            self.dealer_cards[batch_idx, card_pos] = ranks
            self.dealer_card_idx[active_dealers] += 1  
            self.dealer_has_ace[active_dealers] |= (ace_mask & ~dealer_already_has_ace)
            self.dealer_card_sums[active_dealers] += ranks

            over_21_with_ace = active_dealers & (self.dealer_card_sums > 21) & self.dealer_has_ace
            self.dealer_card_sums[over_21_with_ace] -= 10
            self.dealer_has_ace[over_21_with_ace] = False

            self.deck_positions[active_dealers] += 1
            active_dealers = stand_mask & (self.dealer_card_sums < 17) & (self.dealer_card_sums <= 21)

        return hit_mask, stand_mask

    def calculate_rewards(self, hit_mask, stand_mask):
        # calculate rewards of players that hit 
        over_21_player_card_sums=(self.player_card_sums > 21) & hit_mask
        self.rewards[over_21_player_card_sums]=self.LOSS_REWARD
        self.terminated |= over_21_player_card_sums

        # calculate rewards of players that stood
        stand_batch_idx = self.g[stand_mask]  # Get indices of games that stood
        player_sums=self.player_card_sums[stand_mask]
        dealer_sums=self.dealer_card_sums[stand_mask]
        stand_wins=(dealer_sums > 21) | (player_sums >= dealer_sums)  # treat push as a win
        self.rewards[stand_batch_idx[stand_wins]]=self.WIN_REWARD
        self.rewards[stand_batch_idx[~stand_wins]]=self.LOSS_REWARD
        self.terminated[stand_mask]=True

    def step(self, actions):
        # execute actions
        hit_mask, stand_mask = self.execute_actions(actions)
        # calculate rewards
        self.rewards.zero_()
        self.calculate_rewards(hit_mask, stand_mask)

        return self.get_obs(), self.rewards, self.terminated, None, self.get_info()

