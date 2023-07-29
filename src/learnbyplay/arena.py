import learnbyplay.games.rules
from learnbyplay.player import Player
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import torch
import copy

class Arena:
    def __init__(self, authority: learnbyplay.games.rules.Authority, agent: Player, opponent: Player,
                 agent_starts: bool) -> None:
        self.authority = authority
        self.agent = agent
        self.opponent = opponent
        self.agent_starts = agent_starts
        self.index_to_player = {0: self.agent, 1: self.opponent}
        if not self.agent_starts:
            self.index_to_player = {0: self.opponent, 1: self.agent}

    def RunGame(self):
        state_tsr = self.authority.InitialState()
        state_action_list = []
        number_of_moves = 0
        game_status = learnbyplay.games.rules.GameStatus.NONE

        while (game_status == learnbyplay.games.rules.GameStatus.NONE) and number_of_moves < self.authority.MaximumNumberOfMoves():
            player_ndx = number_of_moves % 2
            player = self.index_to_player[player_ndx]
            chosen_move = player.ChooseMove(self.authority, state_tsr)
            state_action_list.append((copy.deepcopy(state_tsr), chosen_move))

            state_tsr, game_status = self.authority.Move(state_tsr, chosen_move, player.identifier)
            number_of_moves += 1
            # Make sure the win or loss is attributed to the agent
            if game_status == learnbyplay.games.rules.GameStatus.WIN and player is self.opponent:
                game_status = learnbyplay.games.rules.GameStatus.LOSS
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS and player is self.opponent:
                game_status = learnbyplay.games.rules.GameStatus.WIN
            elif number_of_moves == self.authority.MaximumNumberOfMoves():
                game_status = learnbyplay.games.rules.GameStatus.DRAW
        state_action_list.append((copy.deepcopy(state_tsr), None))
        return state_action_list, game_status