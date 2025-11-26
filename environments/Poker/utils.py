# utility functions for our poker environment
import enum
import numpy as np

class Cards(enum.Enum):
    ACE_OF_CLUBS=1
    TWO_OF_CLUBS=2
    THREE_OF_CLUBS=3
    FOUR_OF_CLUBS=4
    FIVE_OF_CLUBS=5
    SIX_OF_CLUBS=6
    SEVEN_OF_CLUBS=7
    EIGHT_OF_CLUBS=8
    NINE_OF_CLUBS=9
    TEN_OF_CLUBS=10
    JACK_OF_CLUBS=11
    QUEEN_OF_CLUBS=12
    KING_OF_CLUBS=13
    ACE_OF_DIAMONDS=14
    TWO_OF_DIAMONDS=15
    THREE_OF_DIAMONDS=16
    FOUR_OF_DIAMONDS=17
    FIVE_OF_DIAMONDS=18
    SIX_OF_DIAMONDS=19
    SEVEN_OF_DIAMONDS=20
    EIGHT_OF_DIAMONDS=21
    NINE_OF_DIAMONDS=22
    TEN_OF_DIAMONDS=23
    JACK_OF_DIAMONDS=24
    QUEEN_OF_DIAMONDS=25
    KING_OF_DIAMONDS=26
    ACE_OF_HEARTS=27
    TWO_OF_HEARTS=28
    THREE_OF_HEARTS=29
    FOUR_OF_HEARTS=30
    FIVE_OF_HEARTS=31
    SIX_OF_HEARTS=32
    SEVEN_OF_HEARTS=33
    EIGHT_OF_HEARTS=34
    NINE_OF_HEARTS=35
    TEN_OF_HEARTS=36
    JACK_OF_HEARTS=37
    QUEEN_OF_HEARTS=38
    KING_OF_HEARTS=39
    ACE_OF_SPADES=40
    TWO_OF_SPADES=41
    THREE_OF_SPADES=42
    FOUR_OF_SPADES=43
    FIVE_OF_SPADES=44
    SIX_OF_SPADES=45
    SEVEN_OF_SPADES=46
    EIGHT_OF_SPADES=47
    NINE_OF_SPADES=48
    TEN_OF_SPADES=49
    JACK_OF_SPADES=50
    QUEEN_OF_SPADES=51
    KING_OF_SPADES=52

class Deck():
    NUM_SUITS=4
    TOTAL_CARDS=52
    NUM_CARDS_PER_SUIT=13

    def __init__(self):
        self.deck=self.get_deck()
        self.cards=np.array([card for card in Cards])
        
    def get_deck(self,shuffle:bool=True):
        # gets a new deck of cards
        deck=np.array((52, 1), dtype=np.int32)
        for card in Cards:
            deck[card.value-1]=card.value
        if shuffle: deck=self.shuffle(deck)
        return deck # shape: (52, 1)

    def shuffle(self):
        np.random.shuffle(self.deck)
        return self.deck # shape: (52, 1)





