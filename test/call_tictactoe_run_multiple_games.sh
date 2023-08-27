#!/bin/bash
python tictactoe_run_multiple_games.py \
	--agentArchitecture=SaintAndre_1024 \
	--agentNeuralNetworkFilepath="..\experiments\output_tictactoe_train_agent_level5\SaintAndre_1024.pth" \
	--agentTemperature=0 \
	--agentLookAheadDepth=2 \
	--numberOfGames=1000 \
	