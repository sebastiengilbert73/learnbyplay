import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.tictactoe_arch as architectures
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    consolePlayer,
    agentArchitecture,
    agentNeuralNetworkFilepath,
    agentTemperature,
    numberOfGames
):
    logging.info("tictactoe_run_multiple_games.main()")

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if consolePlayer:
        agent = learnbyplay.player.ConsolePlayer(agent_identifier)
    elif agentNeuralNetworkFilepath is not None:
        neural_net = None
        if agentArchitecture == 'SaintAndre_512':
            neural_net = architectures.SaintAndre(
                latent_size=512,
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_run_multiple_games.main(): Not implemented architecture '{agentArchitecture}'")
        neural_net.load_state_dict(torch.load(agentNeuralNetworkFilepath))
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=neural_net,
            temperature=agentTemperature,
            flatten_state=True
        )
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)

    arena = Arena(authority, agent, opponent)
    number_of_agent_wins, number_of_agent_losses, number_of_draws = arena.RunMultipleGames(numberOfGames)
    logging.info(f"number_of_agent_wins = {number_of_agent_wins}; number_of_agent_losses = {number_of_agent_losses}; number_of_draws = {number_of_draws}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--consolePlayer', help="The agent plays with the console", action='store_true')
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentNeuralNetworkFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--numberOfGames', help="The number of games played. Default: 1000", type=int, default=1000)
    args = parser.parse_args()
    if args.agentNeuralNetworkFilepath.upper() == 'NONE':
        args.agentNeuralNetworkFilepath = None
    main(
        args.consolePlayer,
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.agentTemperature,
        args.numberOfGames
    )