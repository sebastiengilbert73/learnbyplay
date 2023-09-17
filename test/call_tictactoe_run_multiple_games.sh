#!/bin/bash
python tictactoe_run_multiple_games.py \
	--agentArchitecture=SaintAndre_1024 \
	--agentNeuralNetworkFilepath="C:\Users\sebas\Documents\projects\tutorial_learnbyplay\learn_tictactoe\output_tictactoe_train_agent_level17\SaintAndre_1024.pth" \
	--agentTemperature=0 \
	--agentLookAheadDepth=1 \
	--numberOfGames=1000 \
	