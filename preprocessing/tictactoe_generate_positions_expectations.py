import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import os
import random
import torch
import ast
import architectures.tictactoe_arch as architectures

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    numberOfGames,
    gamma,
    randomSeed,
    agentArchitecture,
    agentFilepath,
    opponentArchitecture,
    opponentFilepath,
    epsilon,
    temperature,
    printPositionsAndExpectations
):
    logging.info("tictactoe_generate_positions_expectations.main()")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if agentFilepath is not None:
        agent_neural_net = None
        if agentArchitecture == 'SaintAndre_512':
            agent_neural_net = architectures.SaintAndre(
                latent_size=512,
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_generate_positions_expectations.main(): Not implemented agent architecture '{agentArchitecture}'")
        agent_neural_net.load_state_dict(torch.load(agentFilepath))
        agent_neural_net.to(device)
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier='X',
            neural_net=agent_neural_net,
            temperature=temperature,
            flatten_state=True,
            acts_as_opponent=False,
            epsilon=epsilon
        )

    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)
    if opponentFilepath is not None:
        opponent_neural_net = None
        if opponentArchitecture == 'SaintAndre_512':
            opponent_neural_net = architectures.SaintAndre(
                latent_size=512,
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_generate_positions_expectations.main(): Not implemented opponent architecture '{opponentArchitecture}'")
        opponent_neural_net.load_state_dict(torch.load(opponentFilepath))
        opponent_neural_net.to(device)
        opponent = learnbyplay.player.PositionRegressionPlayer(
            identifier='O',
            neural_net=opponent_neural_net,
            temperature=temperature,
            flatten_state=True,
            acts_as_opponent=True,
            epsilon=epsilon
        )

    arena = Arena(authority, agent, opponent)

    position_expectation_list = arena.GeneratePositionsAndExpectations(number_of_games=numberOfGames,
                                                                       gamma=gamma)
    with open(os.path.join(outputDirectory, "dataset.csv"), "w") as output_file:
        for feature_ndx in range(18):
            output_file.write(f"v{feature_ndx},")
        output_file.write("return\n")
        for position, expectation in position_expectation_list:
            position_vct = position.view(-1)
            for feature_ndx in range(position_vct.shape[0]):
                output_file.write(f"{position_vct[feature_ndx].item()},")
            output_file.write(f"{expectation}\n")

    logging.info(f"Done!")

    if printPositionsAndExpectations:
        for position_expectation in position_expectation_list:
            print(f"{authority.ToString(position_expectation[0])}\n{position_expectation[1]}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_tictactoe_generate_positions_expectations'",
                        default="./output_tictactoe_generate_positions_expectations")
    parser.add_argument('--numberOfGames', help="The number of games. Default: 10000", type=int, default=10000)
    parser.add_argument('--gamma', help="The discount factor. Default: 0.8", type=float, default=0.8)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--agentArchitecture', help="The architecture for the agent. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentArchitecture', help="The architecture for the opponent neural network. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--opponentFilepath', help="The filepath to the opponent neural network. Default: 'None'", default='None')
    parser.add_argument('--epsilon', help="The epsilon parameter, for epsilon-greedy choices. Default: 0.0", type=float, default=0.0)
    parser.add_argument('--temperature', help="The SoftMax temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--printPositionsAndExpectations', help="Print the positions and expectations to the console", action='store_true')
    args = parser.parse_args()
    if args.agentFilepath.upper() == 'NONE':
        args.agentFilepath = None
    if args.opponentFilepath.upper() == 'NONE':
        args.opponentFilepath = None
    main(
        args.outputDirectory,
        args.numberOfGames,
        args.gamma,
        args.randomSeed,
        args.agentArchitecture,
        args.agentFilepath,
        args.opponentArchitecture,
        args.opponentFilepath,
        args.epsilon,
        args.temperature,
        args.printPositionsAndExpectations
    )