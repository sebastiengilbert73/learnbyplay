import learnbyplay.games.rules
import torch
import copy
from typing import Dict, List, Any, Set, Tuple, Optional, Union

class Connect4(learnbyplay.games.rules.Authority):
    def __init__(self):
        super(Connect4, self).__init__()
        self.player_identifiers = ['YELLOW', 'RED']
        self.player_to_channel = {'YELLOW': 0, 'RED': 1}  # YELLOW is the agent; RED is the opponent

    def LegalMoves(self, state_tsr: torch.Tensor, player_identifier: str) -> List[str]:
        legal_moves = []
        for col in range(7):
            if state_tsr[:, 0, col].any():
                continue  # The 0th row is occupied
            legal_moves.append(str(col))
        return legal_moves

    def InitialState(self) -> torch.Tensor:
        state_tsr = torch.zeros((2, 6, 7), dtype=torch.uint8)
        return state_tsr

    def ToString(self, state_tsr: torch.Tensor) -> str:
        state_string = '_____________________________\n'
        for row in range(6):
            for col in range(7):
                state_string += '| '
                number_of_occupied_channels = 0
                for player, channel in self.player_to_channel.items():
                    if state_tsr[channel, row, col] == 1:
                        state_string += player[0] + ' '  # 'Y' or 'R'
                        number_of_occupied_channels += 1
                if number_of_occupied_channels > 1:
                    raise ValueError(f"Connect4.ToString(): (row, col) = ({row}, {col}); number_of_occupied_channels = {number_of_occupied_channels}")
                if number_of_occupied_channels == 0:
                    state_string += '  '
            state_string += '|\n'
            state_string += '_____________________________\n'
        state_string += "  0   1   2   3   4   5   6"
        return state_string

    def Move(self, state_tsr: torch.Tensor, move: str, player_identifier: str) -> Tuple[torch.Tensor, learnbyplay.games.rules.GameStatus]:
        chosen_col = int(move)
        if chosen_col < 0 or chosen_col > 6:
            raise ValueError(f"Connect4.Move(): chosen_col ({chosen_col}) is not in the [0, 6] range")
        if state_tsr[:, 0, chosen_col].any():
            raise ValueError(f"Connect4.Move(): The chosen column ({chosen_col}) is already filled")
        if player_identifier == 'Y':
            player_identifier = 'YELLOW'
        if player_identifier == 'R':
            player_identifier = 'RED'

        highest_unoccupied_row = 0
        for candidate_row in range(1, 6):
            if state_tsr[:, candidate_row, chosen_col].any():
                break
            highest_unoccupied_row = candidate_row
        new_state_tsr = copy.deepcopy(state_tsr)

        new_state_tsr[self.player_to_channel[player_identifier], highest_unoccupied_row, chosen_col] = 1

        # Game status
        game_status = learnbyplay.games.rules.GameStatus.NONE
        a_line_is_completed = self.ALineIsCompleted(new_state_tsr, player_identifier)
        if a_line_is_completed:
            game_status = learnbyplay.games.rules.GameStatus.WIN
        else:  # No line was completed
            if torch.nonzero(new_state_tsr).size(0) == 6 * 7:  # If the number of non-zero entries is 6 * 7
                game_status = learnbyplay.games.rules.GameStatus.DRAW

        return new_state_tsr, game_status

    def MaximumNumberOfMoves(self) -> int:
        return 6 * 7

    def StateTensorShape(self) -> Tuple[int]:
        return (2, 6, 7)

    def SwapAgentAndOpponent(self, state_tsr: torch.Tensor) -> torch.Tensor:
        swapped_state_tsr = torch.zeros_like(state_tsr)
        swapped_state_tsr[0, :, :] = state_tsr[1, :, :]
        swapped_state_tsr[1, :, :] = state_tsr[0, :, :]
        return swapped_state_tsr

    def ALineIsCompleted(self, state_tsr, player_identifier):
        channel = self.player_to_channel[player_identifier]
        # Row of 4
        for row in range(6):
            for starting_col in range(4):  # [0, 1, 2, 3]
                if state_tsr[channel, row, starting_col: starting_col + 4].all():
                    return True
        # Column of 4
        for col in range(7):
            for starting_row in range(3):  # [0, 1, 2]
                if state_tsr[channel, starting_row: starting_row + 4, col].all():
                    return True
        # Slash
        for starting_row in range(3, 6):
            for starting_col in range(0, 4):
                number_of_squares = 0
                for z in range(4):
                    row = starting_row - z
                    col = starting_col + z
                    if state_tsr[channel, row, col] > 0:
                        number_of_squares += 1
                    else:
                        break
                if number_of_squares == 4:
                    return True
        # Backslash
        for starting_row in range(0, 3):
            for starting_col in range(0, 4):
                number_of_squares = 0
                for z in range(4):
                    row = starting_row + z
                    col = starting_col + z
                    if state_tsr[channel, row, col] > 0:
                        number_of_squares += 1
                    else:
                        break
                if number_of_squares == 4:
                    return True
        return False