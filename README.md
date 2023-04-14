# IT3105 - AlphaGo Knock-Off (Deep Reinforcement Learning)

This project implements a simpler version of the AlphaGo/AlphaZero system from Deepmind.
The core algorithm consists of estimating a probability distribution over all moves with Monte-Carlo Tree-Search (MCTS).
Many state-distribution pairs form training cases, which are fed to a convolutional deep neural network.
During training, the (hopefully) improving network will be used to guide the rollouts in MCTS, which should yield improving estimations.

Although the core algorithms have been generalized to work with "any" arbitrary turn-based 2-player game, the main focus has been on Hex.
The goal in Hex is to be the first to connect both sides of the board (Left/Right for blue, Top/Bottom for red).

To play against the best trained model, simply run the play.py file (change net_player, player_start to try different variations). 
The user-interface is very limited, and only supports move insertions via console by typing the index of a given square.

The model scored a 100/100 in the Online Hex Tournament (OHT).
(Certificates to play in OHT have been omitted)

# Visualization of a game
### Model playing as red
![](https://github.com/espsjo/it3105/blob/main/Documentation/visualization.gif)
