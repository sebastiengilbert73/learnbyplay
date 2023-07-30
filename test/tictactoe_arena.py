import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.tictactoe_arch as architectures

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    consolePlayer,
    agentStarts,
    agentArchitecture,
    agentNeuralNetworkFilepath,
    agentTemperature
):
    logging.info("tictactoe_arena.main()")

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
            raise NotImplementedError(f"tictactoe_arena.main(): Not implemented architecture '{agentArchitecture}'")
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=neural_net,
            temperature=agentTemperature,
            flatten_state=True
        )
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)

    arena = Arena(authority, agent, opponent)
    state_action_list, game_status = arena.RunGame(agentStarts)
    for state_action in state_action_list:
        print(authority.ToString(state_action[0]))
        print(state_action[1])
        print()
    logging.info(f"game_status = {game_status}")
    #logging.info(f"state_action_list = {state_action_list}\ngame_status = {game_status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--consolePlayer', help="The agent plays with the console", action='store_true')
    parser.add_argument('--agentStarts', help="The agent plays first", action='store_true')
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentNeuralNetworkFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 1.0", type=float, default=1.0)
    args = parser.parse_args()
    if args.agentNeuralNetworkFilepath.upper() == 'NONE':
        args.agentNeuralNetworkFilepath = None
    main(
        args.consolePlayer,
        args.agentStarts,
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.agentTemperature
    )