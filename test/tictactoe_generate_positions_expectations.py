import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import os
import random
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    numberOfGames,
    gamma,
    randomSeed
):
    logging.info("tictactoe_generate_positions_expectations.main()")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)

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

    #for position_expectation in position_expectation_list:
    #    print(f"{authority.ToString(position_expectation[0])}\n{position_expectation[1]}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_tictactoe_generate_positions_expectations'",
                        default="./output_tictactoe_generate_positions_expectations")
    parser.add_argument('--numberOfGames', help="The number of games. Default: 1000", type=int, default=1000)
    parser.add_argument('--gamma', help="The discount factor. Default: 0.9", type=float, default=0.9)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    args = parser.parse_args()
    main(
        args.outputDirectory,
        args.numberOfGames,
        args.gamma,
        args.randomSeed
    )