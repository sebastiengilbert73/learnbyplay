import argparse
import logging
import learnbyplay.games.connect4
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.connect4_arch as architectures
import random
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    agentIsConsolePlayer,
    agentStarts,
    agentArchitecture,
    agentNeuralNetworkFilepath,
    agentTemperature,
    opponentIsConsolePlayer,
    opponentArchitecture,
    opponentNeuralNetworkFilepath,
    opponentTemperature,
    randomSeed
):
    logging.info("connect4_arena.main()")

    random.seed(randomSeed)
    torch.manual_seed(0)

    authority = learnbyplay.games.connect4.Connect4()
    agent_identifier = 'YELLOW'
    opponent_identifier = 'RED'
    # Build the agent
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if agentIsConsolePlayer:
        agent = learnbyplay.player.ConsolePlayer(agent_identifier)
    elif agentNeuralNetworkFilepath is not None:
        neural_net = None
        if agentArchitecture.startswith('Usb_'):
            chunks = ChunkArchName(agentArchitecture)
            neural_net = architectures.Usb(
                number_of_convs=int(chunks[1]),
                latent_size=int(chunks[2]),
                dropout_ratio=0.5
            )
        elif agentArchitecture.startswith('Dvi_'):
            chunks = ChunkArchName(agentArchitecture)
            neural_net = architectures.Dvi(
                nconv1=int(chunks[1]),
                nconv2=int(chunks[2]),
                latent_size=int(chunks[3]),
                dropout_ratio=0.5
            )
        elif agentArchitecture.startswith('Hdmi_'):
            chunks = ChunkArchName(agentArchitecture)
            neural_net = architectures.Hdmi(
                nconv1=int(chunks[1]),
                nconv2=int(chunks[2]),
                latent_size=int(chunks[3]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"connect4_arena.main(): Not implemented architecture '{agentArchitecture}'")
        neural_net.load_state_dict(torch.load(agentNeuralNetworkFilepath))
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=neural_net,
            temperature=agentTemperature,
            flatten_state=False,
            acts_as_opponent=False
        )

    # Build the opponent
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)
    if opponentIsConsolePlayer:
        opponent = learnbyplay.player.ConsolePlayer(opponent_identifier)
    elif opponentNeuralNetworkFilepath is not None:
        opponent_neural_net = None
        if opponentArchitecture.startswith('Usb_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = architectures.Usb(
                number_of_convs=int(chunks[1]),
                latent_size=int(chunks[2]),
                dropout_ratio=0.5
            )
        elif opponentArchitecture.startswith('Dvi_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = architectures.Dvi(
                nconv1=int(chunks[1]),
                nconv2=int(chunks[2]),
                latent_size=int(chunks[3]),
                dropout_ratio=0.5
            )
        elif opponentArchitecture.startswith('Hdmi_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = architectures.Hdmi(
                nconv1=int(chunks[1]),
                nconv2=int(chunks[2]),
                latent_size=int(chunks[3]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"connect4_arena.main(): Not implemented architecture '{opponentArchitecture}'")
        opponent_neural_net.load_state_dict(torch.load(opponentNeuralNetworkFilepath))
        opponent = learnbyplay.player.PositionRegressionPlayer(
            identifier=opponent_identifier,
            neural_net=opponent_neural_net,
            temperature=opponentTemperature,
            flatten_state=False,
            acts_as_opponent=True
        )

    # Create the arena
    arena = Arena(authority, agent, opponent)
    state_action_list, game_status = arena.RunGame(agentStarts)
    for state_action in state_action_list:
        print(authority.ToString(state_action[0]))
        print(state_action[1])
        print()
    logging.info(f"game_status, from the agent perspective = {game_status}")


def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agentIsConsolePlayer', help="The agent plays with the console", action='store_true')
    parser.add_argument('--agentStarts', help="The agent plays first", action='store_true')
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'Usb_64_512'", default='Usb_64_512')
    parser.add_argument('--agentNeuralNetworkFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--opponentIsConsolePlayer', help="The opponent plays with the console", action='store_true')
    parser.add_argument('--opponentArchitecture', help="In case of a neural network, the architecture. Default: 'Usb_64_512'",
                        default='Usb_64_512')
    parser.add_argument('--opponentNeuralNetworkFilepath', help="The filepath to the opponent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentTemperature', help="The opponent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    args = parser.parse_args()
    if args.agentNeuralNetworkFilepath.upper() == 'NONE':
        args.agentNeuralNetworkFilepath = None
    if args.opponentNeuralNetworkFilepath.upper() == 'NONE':
        args.opponentNeuralNetworkFilepath = None
    main(
        args.agentIsConsolePlayer,
        args.agentStarts,
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.agentTemperature,
        args.opponentIsConsolePlayer,
        args.opponentArchitecture,
        args.opponentNeuralNetworkFilepath,
        args.opponentTemperature,
        args.randomSeed
    )