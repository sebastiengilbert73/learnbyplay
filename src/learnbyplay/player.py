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

class PositionRegressionPlayer(Player):
    def __init__(self, identifier, neural_net, temperature=1.0, flatten_state=True,
                 acts_as_opponent=False):
        super(PositionRegressionPlayer, self).__init__(identifier)
        self.neural_net = neural_net
        self.neural_net.eval()
        self.temperature = temperature
        self.flatten_state = flatten_state
        self.device = next(self.neural_net.parameters()).device
        self.acts_as_opponent = acts_as_opponent

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority,
                   state_tsr: torch.Tensor) -> str:
        #if must_swap_agent_and_opponent:
        #    state_tsr = authority.SwapAgentAndOpponent(state_tsr)
        move_predicted_return_list = self.PredictReturns(state_tsr, authority)
        legal_moves = []
        corresponding_predicted_returns = []
        highest_predicted_return = -2.0
        champion_move = None
        for move, predicted_return in move_predicted_return_list:
            legal_moves.append(move)
            corresponding_predicted_returns.append(predicted_return)
            if predicted_return > highest_predicted_return:
                highest_predicted_return = predicted_return
                champion_move = move

        if self.temperature <= 0:  # Zero temperature: return the greedy best move
            return champion_move

        corresponding_predicted_temperature_returns_tsr = torch.tensor(corresponding_predicted_returns)/self.temperature
        corresponding_probabilities_tsr = torch.nn.functional.softmax(corresponding_predicted_temperature_returns_tsr, dim=0)
        random_nbr = random.random()
        running_sum = 0
        for move_ndx in range(corresponding_probabilities_tsr.shape[0]):
            running_sum += corresponding_probabilities_tsr[move_ndx].item()
            if running_sum >= random_nbr:
                return legal_moves[move_ndx]

    def PredictReturns(self, state_tsr, authority):
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        move_predicted_return_list = []
        for move_ndx in range(len(legal_moves)):
            move = legal_moves[move_ndx]
            candidate_state_tsr, game_status = authority.Move(
                state_tsr, move, self.identifier
            )
            candidate_state_tsr = candidate_state_tsr.float().to(self.device)
            if self.acts_as_opponent:
                candidate_state_tsr = authority.SwapAgentAndOpponent(candidate_state_tsr)
            if self.flatten_state:
                candidate_state_tsr = candidate_state_tsr.view(-1)
            predicted_return = self.neural_net(candidate_state_tsr.unsqueeze(0)).squeeze().item()
            move_predicted_return_list.append((move, predicted_return))
        return move_predicted_return_list