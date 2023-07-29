import abc
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import learnbyplay.games.rules
import torch
import random

class Player(abc.ABC):
    def __init__(self, identifier: str):
        self.identifier = identifier

    @abc.abstractmethod
    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        pass

class RandomPlayer(Player):
    def __init__(self, identifier):
        super(RandomPlayer, self).__init__(identifier)

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        return random.choice(legal_moves)

class ConsolePlayer(Player):
    def __init__(self, identifier):
        super(ConsolePlayer, self).__init__(identifier)

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        the_chosen_move_is_legal = False
        chosen_move = None
        while not the_chosen_move_is_legal:
            print(authority.ToString(state_tsr))
            chosen_move = input(f"Choose a move (legal moves: {legal_moves}): ")
            the_chosen_move_is_legal = chosen_move in legal_moves
        return chosen_move