from unittest import result
import gymnasium as gym
import numpy as np
from numba import njit, vectorize, guvectorize, int32
import random
from typing import Tuple

#-----------------------------------------------------
# Environment of the game 2048
# supports arbitrary board shapes
# implemented with numba to support ultra-fast cpu operations
# enforces correct game logic with rl rewards
#------------------------------------------------------
# Define the numba functions (cannot use numba of class function that have self parameter)

# adds a tile to the 2048 games
@njit
def add_tile_numba(board):
    rows, cols=board.shape
    empty_rows, empty_cols = [], []
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0:
                empty_rows.append(r)
                empty_cols.append(c)
    
    n_empty=len(empty_rows)
    if n_empty==0: return

    idx = random.randint(0, n_empty - 1)
    val = 4 if random.random() > 0.9 else 2
    target_r = empty_rows[idx]
    target_c = empty_cols[idx]
    board[target_r, target_c] = val

# numba function to rotate an array for the 2048 board game
@njit
def numba_rotate(board):
    n = board.shape[0]
    m = board.shape[1]
    res = np.zeros((m, n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            res[m - 1 - j, i] = board[i, j]
    return res

# numba function to check if the game is over
# check for zeros, then check for horizontal merges, then check for vertical merges
@njit()
def is_game_over_numba(board):
    rows, cols = board.shape

    for i in range(rows):
        for j in range(cols):
            if board[i, j] == 0:
                return False

    for i in range(rows):
        for j in range(cols - 1):
            if board[i, j] == board[i, j+1]:
                return False

    for i in range(rows - 1):
        for j in range(cols):
            if board[i, j] == board[i+1, j]:
                return False

    return True

# squashes the rows of a 2048 game
# vectorizes the process so we can squash all rows at once
# step function will rotate array first, so we assume that we need to squash left
@guvectorize([(int32[:], int32[:])], '(n)->(n)', nopython=True)
def squash_row(row, result, score_out):
    n=row.shape[0]
    t_idx=0
    score_out[0] = 0

    # zero out result array
    for i in range(n):
        result[i] = 0

    # single pass compress and merge
    write_idx=0
    last_merged=False 
    
    for i in range(n):
        val = row[i]
        
        # Skip empty tiles
        if val != 0:
            # Case A: The current slot in result is empty (Start of array)
            if result[write_idx] == 0:
                result[write_idx] = val
                
            # Case B: Merge!
            # Matches the value at write_idx AND we didn't just create that value via a merge
            elif result[write_idx] == val and not last_merged:
                merged_val = val * 2
                result[write_idx] = merged_val
                # Accumulate score into the output pointer
                score_out[0] += merged_val 
                last_merged = True
                
            # Case C: No Match (Shift to next slot)
            else:
                write_idx += 1
                result[write_idx] = val
                last_merged = False

# Define the actual environment of the game that utilizes the numba functions
class TFE(gym.Env):
    def __init__(self, board_height, board_width):
        metadata = {'render_modes': ['human']}
        self.n, self.m = board_height, board_width

        self.action_space=gym.spaces.Discrete(4) # up, down, left, right
        self.observation_space=gym.spaces.Box(
            low=0, high=np.inf, shape=(self.n, self.m), dtype=np.int32
        )
        self.board = np.zeros((self.n, self.m), dtype=np.int32)
        self.total_score=0
        self.render_mode = 'human'
        self.rotations_needed = {0: 0, 1: 1, 2: 2, 3: 3}

    def get_obs(self):
        # return the current state of the environment
        return self.board

    def get_info(self):
        # return the current score of the game via a dictionary
        return {'score': self.total_score}

    def add_tile(self):
        # specific 2048 logic helper function, will use the current board and add a tile just like the 2048 game
        add_tile_numba(self.board)

    def is_game_over(self):
        return is_game_over_numba(self.board)

    def reset(self, seed=None, options=None) -> Tuple(np.dnarray, dict):
        # resets the environment for a new game, returns the new board and the new information about the environment
        super().reset(seed=seed)
        self.board=np.zeros((self.n, self.m), dtype=np.int32)
        self.total_score=0
        self.add_tile()
        self.add_tile()
        return self.get_obs(), self.get_info

    # core logic of env
    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        # rotate board k times
        k = self.rotations_needed[action]
        working_board = self.board
        for _ in range(k):
            working_board = numba_rotate(working_board)
        
        # squash tiles and calc step score
        result_buffer, row_scores = np.zeros_like(working_board), np.zeros(self.n, dtype=np.int32)
        squash_row(working_board, result_buffer, row_scores)
        step_score=np.sum(row_scores)
        self.total_score += step_score

        # rotate board back
        rotations_back = (4 - k) % 4
        final_board = result_buffer
        for _ in range(rotations_back):
            final_board = numba_rotate(final_board)

        # assign env board, add tiles, 
        self.board = final_board
        self.add_tile()

        # calc reward, return rl tuple
        reward=0
        if step_score > 0:
            reward = step_score.bit_length()-1

        return self.get_obs(), reward, self.is_game_over(), False, self.get_info()

