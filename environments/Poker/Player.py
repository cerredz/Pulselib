class Player:
    """
    Stateful object representing a player at the table.
    We reuse these objects to avoid Garbage Collection overhead.
    """
    def __init__(self, stack_size: int, player_id: int):
        self.id = player_id
        self.stack = stack_size      # Current chips
        self.current_round_bet = 0   # Amount bet in current street
        self.total_invested = 0      # Amount bet in entire hand
        self.status = 'active'       # 'active', 'folded', 'allin'
        self.hand = []               # List of eval7 cards

    def reset_state(self, new_hand, starting_stack=None):
        """
        Wipes the player state for a new hand.
        """
        self.hand = new_hand
        self.current_round_bet = 0
        self.total_invested = 0
        self.status = 'active'
        
        # If we want to reset stacks to 100BB every hand (Standard RL)
        if starting_stack is not None:
            self.stack = starting_stack
        # If starting_stack is None, we keep the stack from the previous game (Session Mode)
