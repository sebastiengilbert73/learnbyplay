import argparse
import ast
import logging
import os
import run_multiple_games
import random
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    game,
    agentNeuralNetworkFilepath,
    opponentDirectoriesPrefix,
    outputDirectory,
    architecture,
    epsilons,
    numberOfGames,
    useCpu
):
    logging.info(f"run_one_vs_all.main(): game = {game}")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    opponent_level = 0
    opponent_directory = opponentDirectoriesPrefix + str(opponent_level)

    with open(os.path.join(outputDirectory, "run_one_vs_all.csv"), 'w') as output_file:
        output_file.write(f"opponent_level,wins,losses,draws\n")
        while (os.path.exists(opponent_directory) or opponent_level == 0) :
            opponent_neural_network_filepath = None
            if opponent_level > 0:
                opponent_neural_network_filepath = os.path.join(opponent_directory, architecture + '.pth')
            number_of_wins, number_of_losses, number_of_draws = run_multiple_games.main(
                game=game,
                consolePlayer=False,
                agentArchitecture=architecture,
                agentNeuralNetworkFilepath=agentNeuralNetworkFilepath,
                agentTemperature=0.0,
                agentLookAheadDepth=1,
                opponentArchitecture=architecture,
                opponentNeuralNetworkFilepath=opponent_neural_network_filepath,
                opponentTemperature=0.0,
                opponentLookAheadDepth=1,
                numberOfGames=numberOfGames,
                useCpu=useCpu,
                epsilons=epsilons
            )
            logging.info(f"opponent level {opponent_level}: {number_of_wins}, {number_of_losses}, {number_of_draws}")
            output_file.write(f"{opponent_level},{number_of_wins},{number_of_losses},{number_of_draws}\n")
            opponent_level += 1
            opponent_directory = opponentDirectoriesPrefix + str(opponent_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help="The game")
    parser.add_argument('agentNeuralNetworkFilepath', help="The filepath to the agent neural network")
    parser.add_argument('opponentDirectoriesPrefix', help="The prefix for the directories that contain the opponent neural networks")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_run_one_vs_all'",
                        default='./output_run_one_vs_all')
    parser.add_argument('--architecture',
                        help="The neural network architecture. Default: 'Usb_64_512'",
                        default='Usb_64_512')
    parser.add_argument('--epsilons', help="The epsilon values, for epsilon-greedy. Default: '[0.5, 0.5, 0.1]'", default='[0.5, 0.5, 0.1]')
    parser.add_argument('--numberOfGames', help="The number of games played. Default: 1000", type=int, default=1000)
    parser.add_argument('--useCpu', help="Force using CPU", action='store_true')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)

    args.epsilons = ast.literal_eval(args.epsilons)

    main(
        args.game,
        args.agentNeuralNetworkFilepath,
        args.opponentDirectoriesPrefix,
        args.outputDirectory,
        args.architecture,
        args.epsilons,
        args.numberOfGames,
        args.useCpu,
    )