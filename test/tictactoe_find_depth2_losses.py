import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.tictactoe_arch as architectures
import torch
import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
        agentArchitecture,
        agentNeuralNetworkFilepath,
        numberOfGames,
        useCpu
):
    logging.info("tictactoe_find_depth2_losses.main()")

    device = 'cpu'
    if not useCpu and torch.cuda.is_available():
        device = 'cuda'

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'

    neural_net = None
    if agentArchitecture.startswith('SaintAndre_'):
        chunks = ChunkArchName(agentArchitecture)
        neural_net = architectures.SaintAndre(
            latent_size=int(chunks[1]),
            dropout_ratio=0.5
        )
    elif agentArchitecture.startswith('Coptic_'):
        chunks = ChunkArchName(agentArchitecture)
        neural_net = architectures.Coptic(
            number_of_channels=int(chunks[1]),
            dropout_ratio=0.5
        )
    else:
        raise NotImplementedError(f"tictactoe_find_depth2_losses.main(): Not implemented agent architecture '{agentArchitecture}'")
    neural_net.load_state_dict(torch.load(agentNeuralNetworkFilepath))
    neural_net.to(device)
    agent = learnbyplay.player.PositionRegressionPlayer(
        identifier=agent_identifier,
        neural_net=neural_net,
        temperature=0.0,
        flatten_state=True,
        acts_as_opponent=False,
        look_ahead_depth=2,
        epsilon=0.0
    )
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)

    arena = Arena(authority, agent, opponent)
    number_of_agent_wins, number_of_agent_losses, number_of_draws = arena.RunMultipleGames(
        numberOfGames, epsilons=[0])
    logging.info(f"number_of_agent_wins = {number_of_agent_wins}; number_of_agent_losses = {number_of_agent_losses}; number_of_draws = {number_of_draws}")
    return number_of_agent_wins, number_of_agent_losses, number_of_draws

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agentNeuralNetworkFilepath', help="The filepath to the agent neural network.")
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_1024'", default='SaintAndre_1024')
    parser.add_argument('--numberOfGames', help="The number of games played. Default: 1000", type=int, default=1000)
    parser.add_argument('--useCpu', help="Force using CPU", action='store_true')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)

    main(
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.numberOfGames,
        args.useCpu
    )