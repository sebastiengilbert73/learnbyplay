import argparse
import logging
import os
import tictactoe_run_multiple_games

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    directoriesPrefix,
    outputDirectory,
    agentArchitecture,
    agentTemperature,
    numberOfGames,
    useCpu,
    randomSeed
):
    logging.info("tictactoe_level_comparison.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    level = 1
    directory = directoriesPrefix + str(level)
    logging.debug(f"directory = {directory}")
    while os.path.exists(directory):
        #logging.debug(f"directory {directory} exists")
        agent_neural_network_filepath = os.path.join(directory, agentArchitecture + '.pth')
        number_of_agent_wins, number_of_agent_losses, number_of_draws = tictactoe_run_multiple_games.main(
            consolePlayer=False,
            agentArchitecture=agentArchitecture,
            agentNeuralNetworkFilepath=agent_neural_network_filepath,
            agentTemperature=agentTemperature,
            opponentArchitecture=None,
            opponentNeuralNetworkFilepath=None,
            opponentTemperature=1.0,
            numberOfGames=numberOfGames,
            useCpu=useCpu,
            randomSeed=randomSeed
        )
        logging.info(f"level {level}: {number_of_agent_wins}, {number_of_agent_losses}, {number_of_draws}")

        level += 1
        directory = directoriesPrefix + str(level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directoriesPrefix', help="The prefix for the directories that contain the neural networks")
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_tictactoe_level_comparison'", default='./output_tictactoe_level_comparison')
    parser.add_argument('--agentArchitecture',
                        help="In case of a neural network, the architecture. Default: 'SaintAndre_1024'",
                        default='SaintAndre_1024')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 0.0", type=float, default=0.0)
    parser.add_argument('--numberOfGames', help="The number of games played. Default: 1000", type=int, default=1000)
    parser.add_argument('--useCpu', help="Force using CPU", action='store_true')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    args = parser.parse_args()
    main(
        args.directoriesPrefix,
        args.outputDirectory,
        args.agentArchitecture,
        args.agentTemperature,
        args.numberOfGames,
        args.useCpu,
        args.randomSeed
    )