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
    directoriesPrefix,
    outputDirectory,
    architecture,
    epsilons,
    numberOfGames,
    useCpu
):
    logging.info(f"level_improvements.main(): game = {game}")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    base_level = 0
    base_directory = directoriesPrefix + str(base_level)
    upgraded_directory = directoriesPrefix + str(base_level + 1)

    with open(os.path.join(outputDirectory, "level_improvement.csv"), 'w') as output_file:
        output_file.write(f"upgraded_level,wins,losses,draws\n")
        while (os.path.exists(base_directory) or base_level == 0) and os.path.exists(upgraded_directory):
            #logging.debug(f"directory {directory} exists")
            base_neural_network_filepath = None
            if base_level > 0:
                base_neural_network_filepath = os.path.join(base_directory, architecture + '.pth')
            upgraded_neural_network_filepath = os.path.join(upgraded_directory, architecture + '.pth')
            number_of_upgraded_wins, number_of_upgraded_losses, number_of_draws = run_multiple_games.main(
                game=game,
                consolePlayer=False,
                agentArchitecture=architecture,
                agentNeuralNetworkFilepath=upgraded_neural_network_filepath,
                agentTemperature=0.0,
                agentLookAheadDepth=1,
                opponentArchitecture=architecture,
                opponentNeuralNetworkFilepath=base_neural_network_filepath,
                opponentTemperature=0.0,
                opponentLookAheadDepth=1,
                numberOfGames=numberOfGames,
                useCpu=useCpu,
                epsilons=epsilons
            )
            logging.info(f"upgraded level {base_level + 1}: {number_of_upgraded_wins}, {number_of_upgraded_losses}, {number_of_draws}")
            output_file.write(f"{base_level + 1},{number_of_upgraded_wins},{number_of_upgraded_losses},{number_of_draws}\n")
            base_level += 1
            base_directory = directoriesPrefix + str(base_level)
            upgraded_directory = directoriesPrefix + str(base_level + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help="The game")
    parser.add_argument('directoriesPrefix', help="The prefix for the directories that contain the neural networks")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_level_improvements'",
                        default='./output_level_improvements')
    parser.add_argument('--architecture',
                        help="The neural network architecture. Default: 'SaintAndre_1024'",
                        default='SaintAndre_1024')
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
        args.directoriesPrefix,
        args.outputDirectory,
        args.architecture,
        args.epsilons,
        args.numberOfGames,
        args.useCpu,
    )