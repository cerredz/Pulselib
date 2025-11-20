import gymnasium as gym
import numpy as np
import random
from typing import List, Tuple

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__() 
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(3, 3), dtype=np.int32              
        )
        self.board = np.zeros((3, 3), dtype=np.int32) 
        self.total_score = 0
        self.render_mode = 'human'
        self.rotations = {0: (1, -1), 1: (-1, 1), 2: (0, 0), 3: (2, 2)}

    def _get_obs(self):
        # FIX 3: Return a COPY to prevent data corruption
        return self.board.copy()

    def _get_info(self):
        return {"total_score": self.total_score}

    def _add_new_tile(self):
        rows, cols = np.where(self.board == 0)
        if len(rows) == 0: return
        rand_idx = random.choice(range(len(rows)))
        rand_row, rand_col = rows[rand_idx], cols[rand_idx]
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
            
        final_row = new_row + [0] * (self.board.shape[1] - len(new_row))
        return np.array(final_row, dtype=np.int32), step_score

    def _is_game_over(self):
        board_full = np.all(self.board != 0)
        if not board_full: return False # Optimization: Quick return
        
        horizontal_merges = np.any((self.board[:, :-1] == self.board[:, 1:]) & (self.board[:, :-1] != 0))
        vertical_merges = np.any((self.board[:-1, :] == self.board[1:, :]) & (self.board[:-1, :] != 0))
        return not (horizontal_merges or vertical_merges)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3,3), dtype=np.int32)
        self.total_score = 0
        self._add_new_tile()
        self._add_new_tile()
        return self._get_obs(), self._get_info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert action in range(4), f"Invalid action: {action}"

        original_board = self.board.copy()
        rotated_board = np.rot90(self.board, k=self.rotations[action][0])

        new_rotated_board = []
        step_score = 0 # This is the RAW score (2, 4, 8, 16...)
        
        for i in range(self.board.shape[0]):
            new_row, row_score = self._squash_row(rotated_board[i])
            new_rotated_board.append(new_row)
            step_score += int(row_score)

        rotated_board = np.array(new_rotated_board, dtype=np.int32)
        self.board = np.rot90(rotated_board, k=self.rotations[action][1])
        
        self.total_score += step_score
        
        valid_move = not np.array_equal(original_board, self.board)
        terminated = False
        reward = 0.0

        if valid_move:
            if step_score > 0:
                reward = np.log2(step_score)
            self._add_new_tile()
            terminated = self._is_game_over()
        else:
            reward = -1
        
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        print("-" * 21) 
        for row in self.board:
            print(f"|{row[0]:^4}|{row[1]:^4}|{row[2]:^4}|{row[3]:^4}|")
            print("-" * 21)
        print(f"Total Score: {self.total_score}\n")

    def close(self):
        pass