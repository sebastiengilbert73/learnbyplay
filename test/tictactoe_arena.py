import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    consolePlayer,
    agentStarts
):
    logging.info("tictactoe_arena.main()")

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'O'
    opponent_identifier = 'X'
    if agentStarts:
        agent_identifier = 'X'
        opponent_identifier = 'O'
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if consolePlayer:
        agent = learnbyplay.player.ConsolePlayer(agent_identifier)
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
    args = parser.parse_args()

    main(
        args.consolePlayer,
        args.agentStarts
    )