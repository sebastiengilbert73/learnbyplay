import abc
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import torch
from enum import Enum, auto

class GameStatus(Enum):
    WIN = 1
    LOSS = 2
    DRAW = 3
    NONE = 4

class Authority(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def LegalMoves(self, state_tsr: torch.Tensor, player_identifier: str) -> List[str]:
        pass

    @abc.abstractmethod
    def InitialState(self) -> torch.Tensor:
        pass

    def ToString(self, state_tsr: torch.Tensor) -> str:
        return "To be defined"

    @abc.abstractmethod
    def Move(self, state_tsr: torch.Tensor, move: str, player_identifier: str) -> Tuple[torch.Tensor, GameStatus]:
        pass

    @abc.abstractmethod
    def MaximumNumberOfMoves(self) -> int:  # Returns the maximum number of moves in a game, after which the game is a draw
        pass

    @abc.abstractmethod
    def StateTensorShape(self) -> Tuple[int]:
        pass

    @abc.abstractmethod
    def SwapAgentAndOpponent(self, state_tsr: torch.Tensor) -> torch.Tensor:
        pass