import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.tictactoe_arch as architectures
import random
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    agentIsConsolePlayer,
    agentStarts,
    agentArchitecture,
    agentNeuralNetworkFilepath,
    agentTemperature,
    opponentIsConsolePlayer,
    opponentArchitecture,
    opponentNeuralNetworkFilepath,
    opponentTemperature,
    randomSeed
):
    logging.info("tictactoe_arena.main()")

    random.seed(randomSeed)
    torch.manual_seed(0)

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'
    # Build the agent
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if agentIsConsolePlayer:
        agent = learnbyplay.player.ConsolePlayer(agent_identifier)
    elif agentNeuralNetworkFilepath is not None:
        neural_net = None
        if agentArchitecture == 'SaintAndre_512':
            neural_net = architectures.SaintAndre(
                latent_size=512,
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_arena.main(): Not implemented architecture '{agentArchitecture}'")
        neural_net.load_state_dict(torch.load(agentNeuralNetworkFilepath))
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=neural_net,
            temperature=agentTemperature,
            flatten_state=True,
            acts_as_opponent=False
        )
    # Build the opponent
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)
    if opponentIsConsolePlayer:
        opponent = learnbyplay.player.ConsolePlayer(opponent_identifier)
    elif opponentNeuralNetworkFilepath is not None:
        opponent_neural_net = None
        if opponentArchitecture == 'SaintAndre_512':
            opponent_neural_net = architectures.SaintAndre(
                latent_size=512,
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_arena.main(): Not implemented architecture '{opponentArchitecture}'")
        opponent_neural_net.load_state_dict(torch.load(opponentNeuralNetworkFilepath))
        opponent = learnbyplay.player.PositionRegressionPlayer(
            identifier=opponent_identifier,
            neural_net=opponent_neural_net,
            temperature=opponentTemperature,
            flatten_state=True,
            acts_as_opponent=True
        )
    # Create the arena
    arena = Arena(authority, agent, opponent)
    state_action_list, game_status = arena.RunGame(agentStarts)
    for state_action in state_action_list:
        print(authority.ToString(state_action[0]))
        print(state_action[1])
        print()
    logging.info(f"game_status, from the agent perspective = {game_status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agentIsConsolePlayer', help="The agent plays with the console", action='store_true')
    parser.add_argument('--agentStarts', help="The agent plays first", action='store_true')
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentNeuralNetworkFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--opponentIsConsolePlayer', help="The opponent plays with the console", action='store_true')
    parser.add_argument('--opponentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_512'",
                        default='SaintAndre_512')
    parser.add_argument('--opponentNeuralNetworkFilepath', help="The filepath to the opponent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentTemperature', help="The opponent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    args = parser.parse_args()
    if args.agentNeuralNetworkFilepath.upper() == 'NONE':
        args.agentNeuralNetworkFilepath = None
    if args.opponentNeuralNetworkFilepath.upper() == 'NONE':
        args.opponentNeuralNetworkFilepath = None
    main(
        args.agentIsConsolePlayer,
        args.agentStarts,
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.agentTemperature,
        args.opponentIsConsolePlayer,
        args.opponentArchitecture,
        args.opponentNeuralNetworkFilepath,
        args.opponentTemperature,
        args.randomSeed
    )