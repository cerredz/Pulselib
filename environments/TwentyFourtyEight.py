import gymnasium as gym
import numpy as np
import random
from typing import List, Tuple

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
            """Initializes the environment, setting up the action and observation spaces."""
            super().__init__() 
            self.action_space = gym.spaces.Discrete(4) # 1) up, 2) down, 3) left, 4) right
            self.observation_space = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(4, 4),
                dtype=np.int32              
            )
            self.board = np.zeros((4, 4), dtype=np.int32) 
            self.total_score = 0
            self.render_mode = 'human'

    def _get_obs(self):
        """Returns the current board state as the observation."""
        return self.board

    def _get_info(self):
        """Returns a dictionary with extra (non-observation) info, like the score."""
        return {"total_score": self.total_score}

    def _add_new_tile(self):
        """Finds an empty cell and adds a new '2' or '4' tile to the board."""
        rows, cols = np.where(self.board == 0)
        if len(rows) == 0: return
        rand_idx = random.choice(range(len(rows)))
        rand_row, rand_col=rows[rand_idx], cols[rand_idx]
        self.board[rand_row][rand_col] = 4 if random.random() < .1 else 2

    def _squash_row(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        non_zero = [tile for tile in row if tile != 0]
        new_row = []
        step_score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i+1]:
                merged_tile = non_zero[i] * 2
                new_row.append(merged_tile)
                step_score += merged_tile
                i += 2 
            else:
                new_row.append(non_zero[i])
                i += 1
            
        final_row = new_row + [0] * (4 - len(new_row))
        return np.array(final_row, dtype=np.int32), step_score

    def _is_game_over(self):
        """Checks if the board is full and no more merges are possible, returning True or False."""
        board_full=np.all(self.board!=0)
        horizontal_merges=np.any((self.board[:, :-1] == self.board[:, 1:]) & (self.board[:, :-1] != 0))
        vertical_merges=np.any((self.board[:-1, :] == self.board[1:, :]) & (self.board[:-1, :] != 0))
        merges_possible = horizontal_merges or vertical_merges
        return not merges_possible and board_full

    def reset(self, seed=None, options=None):
        """Resets the game to a new starting board and returns the first observation and info."""
        super().reset(seed=seed)
        self.board=np.zeros((4,4), dtype=np.int32)
        self.total_score=0
        self._add_new_tile()
        self._add_new_tile()
        return self._get_obs(), self._get_info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Takes an action, updates the board, calculates the reward, checks if the game is over, and returns (obs, reward, terminated, truncated, info)."""
        # when calling squash function, we must rotate the board according to the direction of the action
        assert action in range(4), f"Invalid action: {action}"       
        rotation_degree={0: (-1, 1), 1: (1, -1),2: (0,0), 3: (2, 2)} 

        # rotate board
        original_board = self.board.copy()
        rotated_board=np.rot90(self.board, k=rotation_degree[action][0])

        # game not over, apply action and calculate reward
        new_rotated_board = []
        step_reward=0
        for i in range(4):
            new_row, row_score = self._squash_row(rotated_board[i])
            new_rotated_board.append(new_row)
            step_reward += int(row_score)

        # rotate board back
        rotated_board = np.array(new_rotated_board, dtype=np.int32)
        self.board = np.rot90(rotated_board, k=rotation_degree[action][1])
        self.total_score += step_reward
        
        valid_move = not np.array_equal(original_board, self.board)
        terminated= False

        if valid_move:
            self._add_new_tile()
            terminated=self._is_game_over()
        if not valid_move:
            step_reward=-1
        if terminated: 
            step_reward=-100
        return self._get_obs(), step_reward, terminated, False, self._get_info()

    def render(self):
        """Prints a human-readable representation of the board to the console."""
        print("-" * 21) 
        for row in self.board:
            print(f"|{row[0]:^4}|{row[1]:^4}|{row[2]:^4}|{row[3]:^4}|")
            print("-" * 21)
        print(f"Total Score: {self.total_score}\n")

    def close(self):
        """Performs any necessary cleanup (often not needed for simple environments)."""
        pass # Your logic here