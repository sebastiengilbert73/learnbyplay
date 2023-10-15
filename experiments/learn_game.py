import argparse
import logging
import ast
import preprocessing.generate_positions_expectations
import train.train_agent
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    game,
    numberOfGames,
    randomSeed,
    architecture,
    epsilons,
    validationProportion,
    numberOfRounds,
    batchSize,
    dropoutRatio,
    learningRate,
    weightDecay,
    numberOfEpochs
):
    logging.info(f"learn_game.main()")

    # Initial round: random matches
    round_dataset_directory = os.path.join(outputDirectory, "datasets_round0")
    preprocessing.generate_positions_expectations.main(
        outputDirectory=round_dataset_directory,
        game=game,
        numberOfGames=numberOfGames,
        gamma=0.01,  # Initial round: only care about the last action
        randomSeed=randomSeed,
        agentArchitecture=architecture,
        agentFilepath=None,
        opponentArchitecture=architecture,
        opponentFilepath=None,
        epsilons=[1.0],
        temperature=0,
        printPositionsAndExpectations=False,
        splitTrainAndValidation=True,
        validationProportion=validationProportion
    )
    round_train_directory = os.path.join(outputDirectory, "train_round0")
    train.train_agent.main(
        datasetFilepath=os.path.join(round_dataset_directory, "train_dataset.csv"),
        validationDatasetFilepath=os.path.join(round_dataset_directory, "validation_dataset.csv"),
        outputDirectory=round_train_directory,
        game=game,
        randomSeed=randomSeed,
        validationRatio=None,
        batchSize=batchSize,
        architecture=architecture,
        dropoutRatio=dropoutRatio,
        useCpu=False,
        learningRate=learningRate,
        weightDecay=weightDecay,
        numberOfEpochs=numberOfEpochs,
        startingNeuralNetworkFilepath=None
    )

    # Loop through the rounds
    for round in range(1, numberOfRounds):
        gamma = 0.1**(1.0/round)
        round_dataset_directory = os.path.join(outputDirectory, f"datasets_round{round}")
        previous_round_train_directory = os.path.join(outputDirectory, f"train_round{round - 1}")
        preprocessing.generate_positions_expectations.main(
            outputDirectory=round_dataset_directory,
            game=game,
            numberOfGames=numberOfGames,
            gamma=gamma,
            randomSeed=round,
            agentArchitecture=architecture,
            agentFilepath=os.path.join(previous_round_train_directory, f"{architecture}.pth"),
            opponentArchitecture=architecture,
            opponentFilepath=os.path.join(previous_round_train_directory, f"{architecture}.pth"),
            epsilons=epsilons,
            temperature=0,
            printPositionsAndExpectations=False,
            splitTrainAndValidation=True,
            validationProportion=validationProportion
        )
        round_train_directory = os.path.join(outputDirectory, f"train_round{round}")
        train.train_agent.main(
            datasetFilepath=os.path.join(round_dataset_directory, "train_dataset.csv"),
            validationDatasetFilepath=os.path.join(round_dataset_directory, "validation_dataset.csv"),
            outputDirectory=round_train_directory,
            game=game,
            randomSeed=round,
            validationRatio=None,
            batchSize=batchSize,
            architecture=architecture,
            dropoutRatio=dropoutRatio,
            useCpu=False,
            learningRate=learningRate,
            weightDecay=weightDecay,
            numberOfEpochs=numberOfEpochs,
            startingNeuralNetworkFilepath=os.path.join(previous_round_train_directory, f"{architecture}.pth")
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_learn_game'",
                        default="./output_learn_game")
    parser.add_argument('--game', help="The game to play. Default: 'connect4'", default='connect4')
    parser.add_argument('--numberOfGames', help="The number of games per round. Default: 10000", type=int, default=10000)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--architecture', help="The architecture for the agent. Default: 'Hdmi_32_64_128'",
                        default='Hdmi_32_64_128')
    parser.add_argument('--epsilons', help="The list of epsilon parameters. Default: '[0.5]'",
                        default='[0.5]')
    parser.add_argument('--validationProportion',
                        help="The validation proportion. Default: 0.2",
                        type=float, default=0.2)
    parser.add_argument('--numberOfRounds', help="The number of rounds. Default: 50", type=int, default=50)
    parser.add_argument('--batchSize', help="The batch size. Default: 64", type=int, default=64)
    parser.add_argument('--dropoutRatio', help="The dropout ratio. Default: 0.7", type=float, default=0.7)
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.0001", type=float, default=0.0001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.00001", type=float, default=0.00001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs per round. Default: 10", type=int, default=10)
    args = parser.parse_args()
    args.epsilons = ast.literal_eval(args.epsilons)
    main(
        args.outputDirectory,
        args.game,
        args.numberOfGames,
        args.randomSeed,
        args.architecture,
        args.epsilons,
        args.validationProportion,
        args.numberOfRounds,
        args.batchSize,
        args.dropoutRatio,
        args.learningRate,
        args.weightDecay,
        args.numberOfEpochs
    )