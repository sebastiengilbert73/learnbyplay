import argparse
import logging
import learnbyplay.games.tictactoe
import learnbyplay.games.sumto100
import learnbyplay.games.connect4
from learnbyplay.arena import Arena
import learnbyplay
import os
import random
import torch
import ast
import architectures.tictactoe_arch as tictactoe_arch
import architectures.sumto100_arch as sumto100_arch
import architectures.connect4_arch as connect4_arch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    game,
    outputDirectory,
    numberOfMatches,
    gamma,
    randomSeed,
    agentArchitecture,
    agentFilepath,
    opponentArchitecture,
    opponentFilepaths,
    epsilons,
    validationProportion
):
    logging.info("generate_positions_expectations_all_predecessors.main()")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    authority = None
    agent_identifier = 'agent'
    opponent_identifier = 'opponent'
    flatten_state = True
    if game == 'tictactoe':
        authority = learnbyplay.games.tictactoe.TicTacToe()
        agent_identifier = 'X'
        opponent_identifier = 'O'
    elif game == 'sumto100':
        authority = learnbyplay.games.sumto100.SumTo100()
    elif game == 'connect4':
        authority = learnbyplay.games.connect4.Connect4()
        agent_identifier = 'YELLOW'
        opponent_identifier = 'RED'
        flatten_state = False
    else:
        raise NotImplementedError(f"generate_positions_expectations_all_predecessors.main(): Not implemented game '{game}'")

    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if agentFilepath is not None:
        agent = CreatePlayer(agentArchitecture, agentFilepath, device, agent_identifier, flatten_state)

    if len(opponentFilepaths) == 0:
        opponentFilepaths = [None]

    number_of_matches_per_opponent = round(numberOfMatches/len(opponentFilepaths))

    logging.info(f"Generating datasets...")
    train_position_expectation_list = []
    validation_position_expectation_list = []
    for opponent_filepath in opponentFilepaths:
        opponent = CreatePlayer(opponentArchitecture, opponent_filepath, device, opponent_identifier, flatten_state)

        arena = Arena(authority, agent, opponent)

        number_of_validation_matches = round(validationProportion * number_of_matches_per_opponent)
        number_of_train_matches = number_of_matches_per_opponent - number_of_validation_matches

        # Generate training dataset
        opponent_train_position_expectation_list = arena.GeneratePositionsAndExpectations(
            number_of_games=number_of_train_matches,
            gamma=gamma,
            epsilons=epsilons)
        train_position_expectation_list += opponent_train_position_expectation_list

        # Generate validation dataset
        opponent_validation_position_expectation_list = arena.GeneratePositionsAndExpectations(
            number_of_games=number_of_validation_matches,
            gamma=gamma,
            epsilons=epsilons)
        validation_position_expectation_list += opponent_validation_position_expectation_list

    number_of_cells = ProductOfElements(authority.InitialState().shape)
    with open(os.path.join(outputDirectory, "train_dataset.csv"), "w") as output_file:
        for feature_ndx in range(number_of_cells):
            output_file.write(f"v{feature_ndx},")
        output_file.write("return\n")
        for position, expectation in train_position_expectation_list:
            position_vct = position.view(-1)
            for feature_ndx in range(position_vct.shape[0]):
                output_file.write(f"{position_vct[feature_ndx].item()},")
            output_file.write(f"{expectation}\n")

    with open(os.path.join(outputDirectory, "validation_dataset.csv"), "w") as output_file:
        for feature_ndx in range(number_of_cells):
            output_file.write(f"v{feature_ndx},")
        output_file.write("return\n")
        for position, expectation in validation_position_expectation_list:
            position_vct = position.view(-1)
            for feature_ndx in range(position_vct.shape[0]):
                output_file.write(f"{position_vct[feature_ndx].item()},")
            output_file.write(f"{expectation}\n")

    logging.info("Done!")

def ProductOfElements(t):
    product = 1
    for i in range(len(t)):
        product *= t[i]
    return product

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

def CreatePlayer(architecture, filepath, device, identifier, flatten_state):
    if filepath is None:
        return learnbyplay.player.RandomPlayer(identifier)

    neural_net = None
    if architecture.startswith('SaintAndre_'):
        chunks = ChunkArchName(architecture)
        neural_net = tictactoe_arch.SaintAndre(
            latent_size=int(chunks[1]),
            dropout_ratio=0.5
        )
    elif architecture.startswith('Coptic_'):
        chunks = ChunkArchName(architecture)
        neural_net = tictactoe_arch.Coptic(
            number_of_channels=int(chunks[1]),
            dropout_ratio=0.5
        )
    elif architecture.startswith('Century21_'):
        chunks = ChunkArchName(architecture)
        neural_net = sumto100_arch.Century21(
            latent_size=int(chunks[1]),
            dropout_ratio=0.5
        )
    elif architecture.startswith('Usb_'):
        chunks = ChunkArchName(architecture)
        neural_net = connect4_arch.Usb(
            number_of_convs=int(chunks[1]),
            latent_size=int(chunks[2]),
            dropout_ratio=0.5
        )
    elif architecture.startswith('Dvi_'):
        chunks = ChunkArchName(architecture)
        neural_net = connect4_arch.Dvi(
            nconv1=int(chunks[1]),
            nconv2=int(chunks[2]),
            latent_size=int(chunks[3]),
            dropout_ratio=0.5
        )
    elif architecture.startswith('Hdmi_'):
        chunks = ChunkArchName(architecture)
        neural_net = connect4_arch.Hdmi(
            nconv1=int(chunks[1]),
            nconv2=int(chunks[2]),
            latent_size=int(chunks[3]),
            dropout_ratio=0.5
        )
    else:
        raise NotImplementedError(f"generate_positions_expectations_all_predecessors.CreatePlayer(): Not implemented architecture '{architecture}'")
    neural_net.load_state_dict(torch.load(filepath))
    neural_net.to(device)
    player = learnbyplay.player.PositionRegressionPlayer(
        identifier=identifier,
        neural_net=neural_net,
        temperature=0,
        flatten_state=flatten_state,
        acts_as_opponent=True,
        epsilon=0
    )

    return player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help="The game to play. Default: 'tictactoe'", default='tictactoe')
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_generate_positions_expectations_all_predecessors'",
                        default="./output_generate_positions_expectations_all_predecessors")
    parser.add_argument('--numberOfMatches', help="The number of matches. Default: 10000", type=int, default=10000)
    parser.add_argument('--gamma', help="The discount factor. Default: 0.8", type=float, default=0.8)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--agentArchitecture', help="The architecture for the agent. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentArchitecture', help="The architecture for the opponent neural network. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--opponentFilepaths', help="The list of filepaths to the opponent neural networks. Default: '[]'", default='[]')
    parser.add_argument('--epsilons', help="The list of epsilon parameters. Default: '[0.5, 0.5, 0.1]'", default='[0.5, 0.5, 0.1]')
    parser.add_argument('--validationProportion', help="The validation proportion. Default: 0.2", type=float, default=0.2)
    args = parser.parse_args()
    if args.agentFilepath.upper() == 'NONE':
        args.agentFilepath = None
    args.opponentFilepaths = ast.literal_eval(args.opponentFilepaths)
    args.epsilons = ast.literal_eval(args.epsilons)
    main(
        args.game,
        args.outputDirectory,
        args.numberOfMatches,
        args.gamma,
        args.randomSeed,
        args.agentArchitecture,
        args.agentFilepath,
        args.opponentArchitecture,
        args.opponentFilepaths,
        args.epsilons,
        args.validationProportion
    )