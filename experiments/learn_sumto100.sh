#!/bin/bash
# gamma: 0.9 -> 0.95
declare -i NUMBER_OF_GAMES=10000
declare -i NUMBER_OF_EPOCHS=25

cd ..
python preprocessing/generate_positions_expectations.py \
	--outputDirectory=./experiments/output_sumto100_generate_positions_expectations_level0 \
    --game=sumto100 \
    --numberOfGames=$NUMBER_OF_GAMES \
    --gamma=0.95 \
    --randomSeed=1 \
    --agentArchitecture=None \
    --agentFilepath=None \
    --opponentArchitecture=None \
    --opponentFilepath=None \
    --epsilon=0.1 \
    --temperature=0
	
dataset_filepath="./experiments/output_sumto100_generate_positions_expectations_level0/dataset.csv"
	
python train/train_agent.py \
		$dataset_filepath \
		--outputDirectory="./experiments/output_sumto100_train_agent_level1" \
		--game=sumto100 \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=Century21_512 \
		--dropoutRatio=0 \
		--learningRate=0.001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath=None
		
python utilities/sumto100_state_value.py \
	"./experiments/output_sumto100_train_agent_level1/Century21_512.pth" \
	--outputDirectory="./experiments/output_sumto100_state_value_level1" \
	--architecture=Century21_512
	
for level in {1..12}
do
	dataset_filepath="./experiments/output_sumto100_generate_positions_expectations_level${level}/dataset.csv"
	python preprocessing/generate_positions_expectations.py \
		--outputDirectory="./experiments/output_sumto100_generate_positions_expectations_level${level}" \
		--game=sumto100 \
		--numberOfGames=$NUMBER_OF_GAMES \
		--gamma=0.95 \
		--randomSeed=1 \
		--agentArchitecture=Century21_512 \
		--agentFilepath="./experiments/output_sumto100_train_agent_level${level}/Century21_512.pth" \
		--opponentArchitecture=Century21_512 \
		--opponentFilepath="./experiments/output_sumto100_train_agent_level${level}/Century21_512.pth" \
		--epsilon=0.1 \
		--temperature=0
		
	declare -i next_level=$((level + 1))
	python train/train_agent.py \
		"./experiments/output_sumto100_generate_positions_expectations_level${level}/dataset.csv" \
		--outputDirectory="./experiments/output_sumto100_train_agent_level${next_level}" \
		--game=sumto100 \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=Century21_512 \
		--dropoutRatio=0 \
		--learningRate=0.001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath="./experiments/output_sumto100_train_agent_level${level}/Century21_512.pth"
		
	python utilities/sumto100_state_value.py \
		"./experiments/output_sumto100_train_agent_level${next_level}/Century21_512.pth" \
		--outputDirectory="./experiments/output_sumto100_state_value_level${next_level}" \
		--architecture=Century21_512
done