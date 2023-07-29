import argparse
import logging
import games.tictactoe

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("play_tictactoe.main()")

    tictactoe_authority = games.tictactoe.TicTacToe()
    state_tsr = tictactoe_authority.InitialState()
    print(state_tsr)

    logging.debug(f"tictactoe_authority.ToString(state_tsr) = \n{tictactoe_authority.ToString(state_tsr)}")

    state_tsr = tictactoe_authority.Move(state_tsr, "0 1", 'X')
    print(tictactoe_authority.ToString(state_tsr))
    state_tsr = tictactoe_authority.Move(state_tsr, "1 1", 'O')
    print(tictactoe_authority.ToString(state_tsr))
    legal_moves = tictactoe_authority.LegalMoves(state_tsr)
    print(f"legal_moves = {legal_moves}")

if __name__ == '__main__':
    main()