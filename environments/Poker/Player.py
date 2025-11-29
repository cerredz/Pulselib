class Player(object):
    def __init__(self, blinds:int, current_round_bet: int, status: bool):
        self.blinds=blinds
        self.current_round_bet=current_round_bet
        self.status=status